import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from config import Config  # noqa: E402

SAMPLE_COURSE = """Course Title: Test Course on Widgets
Course Link: https://example.com/widgets
Course Instructor: Ada Tester

Lesson 0: Introduction
Lesson Link: https://example.com/widgets/0
Widgets are small reusable components. This lesson introduces what a widget is and why widgets matter in modern software.

Lesson 1: Building Widgets
Lesson Link: https://example.com/widgets/1
To build a widget you first define its interface, then implement rendering. Gears and springs are optional accessories.
"""


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "widgets.txt").write_text(SAMPLE_COURSE)
    return str(d)


@pytest.fixture
def test_config(tmp_path):
    return Config(GEMINI_API_KEY="fake-key", CHROMA_PATH=str(tmp_path / "chroma"))


def make_response(text=None, calls=None):
    """Fake Gemini response. calls: list of (name, args) tuples."""
    function_calls = [SimpleNamespace(name=n, args=a) for n, a in (calls or [])] or None
    content = SimpleNamespace(role="model", parts=[])
    return SimpleNamespace(
        text=text,
        function_calls=function_calls,
        candidates=[SimpleNamespace(content=content)],
    )


@pytest.fixture
def fake_client_factory():
    """Returns a function building a fake genai client with scripted responses."""
    def build(responses):
        client = MagicMock()
        client.models.generate_content.side_effect = list(responses)
        return client
    return build


# ---------------------------------------------------------------------------
# API test fixtures
#
# backend/app.py builds a real RAGSystem and mounts ../frontend at import time,
# so it can't be imported in tests. build_test_app() mirrors its endpoints on a
# fresh FastAPI app around an injected RAG system; keep the two in sync.
# ---------------------------------------------------------------------------

from typing import List, Optional  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class Source(BaseModel):
    text: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


class ClearSessionResponse(BaseModel):
    success: bool


def build_test_app(rag_system) -> FastAPI:
    app = FastAPI(title="Course Materials RAG System (test)")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/sessions/{session_id}", response_model=ClearSessionResponse)
    async def clear_session(session_id: str):
        try:
            rag_system.session_manager.clear_session(session_id)
            return ClearSessionResponse(success=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Stand-in for the static frontend mount at "/"
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return "<html><body>Course Materials RAG System</body></html>"

    return app


@pytest.fixture
def sample_sources():
    return [{"text": "Test Course on Widgets - Lesson 1", "url": "https://example.com/widgets/1"}]


@pytest.fixture
def mock_rag_system(sample_sources):
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "session-1"
    rag.query.return_value = ("A widget is a component.", sample_sources)
    rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course on Widgets"],
    }
    return rag


@pytest.fixture
def client(mock_rag_system):
    return TestClient(build_test_app(mock_rag_system))
