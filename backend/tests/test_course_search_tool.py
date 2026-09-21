from unittest.mock import MagicMock

import pytest

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


def results(docs, metas):
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs))


@pytest.fixture
def store():
    s = MagicMock()
    s.get_lesson_link.return_value = "https://x/lesson"
    return s


def test_happy_path_formats_and_tracks_sources(store):
    store.search.return_value = results(
        ["chunk a", "chunk b"],
        [{"course_title": "C", "lesson_number": 1}, {"course_title": "C", "lesson_number": 1}],
    )
    tool = CourseSearchTool(store)
    out = tool.execute(query="q")
    assert "[C - Lesson 1]\nchunk a" in out and "chunk b" in out
    assert tool.last_sources == [{"text": "C - Lesson 1", "url": "https://x/lesson"}]  # deduped


def test_args_forwarded(store):
    store.search.return_value = results([], [])
    CourseSearchTool(store).execute(query="q", course_name="MCP", lesson_number=2)
    store.search.assert_called_once_with(query="q", course_name="MCP", lesson_number=2)


def test_error_returned_verbatim(store):
    store.search.return_value = SearchResults.empty("No course found matching 'zzz'")
    assert CourseSearchTool(store).execute(query="q", course_name="zzz") == "No course found matching 'zzz'"


def test_empty_results_message_with_filters(store):
    store.search.return_value = results([], [])
    out = CourseSearchTool(store).execute(query="q", course_name="C", lesson_number=3)
    assert out == "No relevant content found in course 'C' in lesson 3."


def test_no_lesson_number_skips_link_lookup(store):
    store.search.return_value = results(["d"], [{"course_title": "C", "lesson_number": None}])
    tool = CourseSearchTool(store)
    out = tool.execute(query="q")
    assert out.startswith("[C]")
    store.get_lesson_link.assert_not_called()
    assert tool.last_sources == [{"text": "C", "url": None}]


def test_tool_definition():
    d = CourseSearchTool(MagicMock()).get_tool_definition()
    assert d["name"] == "search_course_content"
    assert d["input_schema"]["required"] == ["query"]


def test_tool_manager_dispatch_and_sources(store):
    store.search.return_value = results(["d"], [{"course_title": "C", "lesson_number": 0}])
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(store))
    assert "d" in tm.execute_tool("search_course_content", query="q")
    assert tm.get_last_sources()
    tm.reset_sources()
    assert tm.get_last_sources() == []
    assert "not found" in tm.execute_tool("nope")


# ---- integration against a real VectorStore ----

@pytest.fixture
def real_tool(test_config, docs_dir):
    from rag_system import RAGSystem
    rag = RAGSystem(test_config)
    courses, chunks = rag.add_course_folder(docs_dir)
    assert courses == 1 and chunks > 0
    return CourseSearchTool(rag.vector_store)


def test_real_unfiltered_search(real_tool):
    out = real_tool.execute(query="what is a widget")
    assert "Test Course on Widgets" in out
    assert real_tool.last_sources


def test_real_partial_course_name(real_tool):
    out = real_tool.execute(query="widget interface", course_name="Widgets")
    assert "[Test Course on Widgets" in out


def test_real_lesson_filter(real_tool):
    out = real_tool.execute(query="widget", course_name="Widgets", lesson_number=1)
    assert "Lesson 1" in out and "Lesson 0" not in out


def test_real_lesson_number_as_float(real_tool):
    # Gemini's function-call args deliver JSON numbers as floats
    out = real_tool.execute(query="widget", course_name="Widgets", lesson_number=1.0)
    assert "Lesson 1" in out and not out.startswith("Search error")


def test_real_source_url(real_tool):
    real_tool.execute(query="widget", course_name="Widgets", lesson_number=0)
    assert real_tool.last_sources[0]["url"] == "https://example.com/widgets/0"


def test_sources_accumulate_across_searches_until_reset(store):
    tool = CourseSearchTool(store)
    tm = ToolManager()
    tm.register_tool(tool)
    store.search.return_value = results(["a"], [{"course_title": "C1", "lesson_number": 1}])
    tm.execute_tool("search_course_content", query="q1")
    store.search.return_value = results(["b", "c"], [{"course_title": "C2", "lesson_number": 2},
                                                     {"course_title": "C1", "lesson_number": 1}])
    tm.execute_tool("search_course_content", query="q2")
    assert [s["text"] for s in tm.get_last_sources()] == ["C1 - Lesson 1", "C2 - Lesson 2"]
    tm.reset_sources()
    assert tm.get_last_sources() == []
