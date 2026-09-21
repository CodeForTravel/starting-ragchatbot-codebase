"""Real Gemini + real vector store. Skipped without GEMINI_API_KEY."""

import os

import pytest

from config import config

pytestmark = pytest.mark.skipif(not config.GEMINI_API_KEY, reason="no GEMINI_API_KEY")


def test_live_model_responds():
    from ai_generator import AIGenerator

    g = AIGenerator(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    assert g.generate_response("Say hi in one word.")


def test_live_content_query(tmp_path):
    from rag_system import RAGSystem

    docs = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    cfg = type(config)(
        GEMINI_API_KEY=config.GEMINI_API_KEY, CHROMA_PATH=str(tmp_path / "c")
    )
    rag = RAGSystem(cfg)
    rag.add_course_folder(docs)
    answer, sources = rag.query(
        "What is covered in lesson 0 of the computer use course?"
    )
    assert answer and not answer.startswith("Sorry")
    assert sources


def test_live_outline_then_unscoped_search(tmp_path):
    """Two-round flow on the real API: outline first, then a search that is not scoped to the same course."""
    from rag_system import RAGSystem

    docs = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    cfg = type(config)(
        GEMINI_API_KEY=config.GEMINI_API_KEY, CHROMA_PATH=str(tmp_path / "c")
    )
    rag = RAGSystem(cfg)
    rag.add_course_folder(docs)

    executed = []
    original = rag.tool_manager.execute_tool

    def spy(name, **kwargs):
        executed.append((name, kwargs))
        return original(name, **kwargs)

    rag.tool_manager.execute_tool = spy
    answer, _ = rag.query(
        "Find another course that covers the same topic as lesson 4 of the MCP course."
    )
    names = [n for n, _ in executed]
    assert names[0] == "get_course_outline"
    assert "search_course_content" in names
    assert len(executed) <= 2
    search_args = next(k for n, k in executed if n == "search_course_content")
    assert not search_args.get("course_name"), search_args
    assert answer and not answer.startswith("Sorry")
