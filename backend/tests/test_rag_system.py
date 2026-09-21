from unittest.mock import patch

import pytest

from rag_system import RAGSystem
from tests.conftest import make_response


@pytest.fixture
def rag(test_config, docs_dir, fake_client_factory):
    with patch("ai_generator.genai.Client"):
        r = RAGSystem(test_config)
    r.add_course_folder(docs_dir)
    r._script = lambda responses: setattr(
        r.ai_generator, "client", fake_client_factory(responses)
    )
    return r


def test_add_course_folder_idempotent(rag, docs_dir):
    assert rag.add_course_folder(docs_dir) == (0, 0)
    assert rag.get_course_analytics()["total_courses"] == 1


def test_content_query_uses_tool_and_returns_sources(rag):
    rag._script([
        make_response(calls=[("search_course_content", {"query": "what is a widget"})]),
        make_response(text="A widget is a component."),
    ])
    answer, sources = rag.query("What is a widget?", "s1")
    assert answer == "A widget is a component."
    assert sources and {"text", "url"} <= set(sources[0])
    # the tool result actually sent back to the model contains real content
    second = rag.ai_generator.client.models.generate_content.call_args_list[1].kwargs
    result = second["contents"][2].parts[0].function_response.response["result"]
    assert "Test Course on Widgets" in result


def test_content_query_with_float_lesson_number(rag):
    rag._script([
        make_response(calls=[("search_course_content",
                              {"query": "widget", "course_name": "Widgets", "lesson_number": 1.0})]),
        make_response(text="ok"),
    ])
    rag.query("q")
    second = rag.ai_generator.client.models.generate_content.call_args_list[1].kwargs
    result = second["contents"][2].parts[0].function_response.response["result"]
    assert not result.startswith("Search error"), result


def test_sources_reset_between_queries(rag):
    rag._script([
        make_response(calls=[("search_course_content", {"query": "widget"})]),
        make_response(text="a"),
        make_response(text="general answer"),
    ])
    _, s1 = rag.query("content q")
    _, s2 = rag.query("general q")
    assert s1 and s2 == []


def test_general_query_no_tool(rag):
    rag._script([make_response(text="4")])
    answer, sources = rag.query("2+2?")
    assert (answer, sources) == ("4", [])


def test_session_history_recorded_and_passed(rag):
    rag._script([make_response(text="first"), make_response(text="second")])
    rag.query("one", "sess")
    rag.query("two", "sess")
    cfg = rag.ai_generator.client.models.generate_content.call_args_list[1].kwargs["config"]
    assert "User: one" in cfg.system_instruction and "Assistant: first" in cfg.system_instruction


def test_outline_query(rag):
    rag._script([
        make_response(calls=[("get_course_outline", {"course_name": "Widgets"})]),
        make_response(text="outline"),
    ])
    rag.query("outline of widgets")
    second = rag.ai_generator.client.models.generate_content.call_args_list[1].kwargs
    result = second["contents"][2].parts[0].function_response.response["result"]
    assert "Lesson" not in result or "0. Introduction" in result
    assert "Test Course on Widgets" in result


def test_generator_exception_propagates(rag):
    rag.ai_generator.client = type("C", (), {"models": type("M", (), {
        "generate_content": staticmethod(lambda **k: (_ for _ in ()).throw(RuntimeError("boom")))})()})()
    with pytest.raises(RuntimeError):
        rag.query("q")  # app.py turns this into HTTP 500 -> "query failed"


def test_two_searches_merge_sources_and_reset(rag):
    rag._script([
        make_response(calls=[("get_course_outline", {"course_name": "Widgets"})]),
        make_response(calls=[("search_course_content", {"query": "widget", "lesson_number": 0})]),
        make_response(text="two-step answer"),
        make_response(text="general"),
    ])
    answer, sources = rag.query("What does lesson 0 of the widgets course cover?")
    assert answer == "two-step answer"  # outline round, search round, then forced-text request
    assert len(rag.ai_generator.client.models.generate_content.call_args_list) == 3
    assert [s["text"] for s in sources] == ["Test Course on Widgets - Lesson 0"]
    _, s2 = rag.query("hello")
    assert s2 == []
