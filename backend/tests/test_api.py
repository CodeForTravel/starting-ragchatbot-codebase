import pytest

from tests.conftest import build_test_app
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api


class TestQueryEndpoint:
    def test_query_without_session_creates_one(self, client, mock_rag_system, sample_sources):
        r = client.post("/api/query", json={"query": "What is a widget?"})
        assert r.status_code == 200
        assert r.json() == {
            "answer": "A widget is a component.",
            "sources": sample_sources,
            "session_id": "session-1",
        }
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is a widget?", "session-1")

    def test_query_with_session_reuses_it(self, client, mock_rag_system):
        r = client.post("/api/query", json={"query": "hi", "session_id": "abc"})
        assert r.status_code == 200
        assert r.json()["session_id"] == "abc"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("hi", "abc")

    def test_query_source_url_optional(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("ok", [{"text": "no link"}])
        r = client.post("/api/query", json={"query": "q"})
        assert r.json()["sources"] == [{"text": "no link", "url": None}]

    def test_query_empty_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("General answer", [])
        assert client.post("/api/query", json={"query": "q"}).json()["sources"] == []

    @pytest.mark.parametrize("body", [{}, {"session_id": "abc"}, {"query": 123}, {"query": None}])
    def test_query_invalid_body_is_422(self, client, body):
        assert client.post("/api/query", json=body).status_code == 422

    def test_query_non_json_body_is_422(self, client):
        assert client.post("/api/query", content="not json").status_code == 422

    def test_query_rag_failure_is_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("boom")
        r = client.post("/api/query", json={"query": "q"})
        assert r.status_code == 500
        assert r.json()["detail"] == "boom"

    def test_query_get_not_allowed(self, client):
        assert client.get("/api/query").status_code == 405


class TestCoursesEndpoint:
    def test_returns_stats(self, client):
        r = client.get("/api/courses")
        assert r.status_code == 200
        assert r.json() == {"total_courses": 1, "course_titles": ["Test Course on Widgets"]}

    def test_no_courses(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {"total_courses": 0, "course_titles": []}
        assert client.get("/api/courses").json() == {"total_courses": 0, "course_titles": []}

    def test_failure_is_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("db down")
        r = client.get("/api/courses")
        assert r.status_code == 500
        assert r.json()["detail"] == "db down"

    def test_post_not_allowed(self, client):
        assert client.post("/api/courses").status_code == 405


class TestSessionEndpoint:
    def test_clear_session(self, client, mock_rag_system):
        r = client.delete("/api/sessions/abc")
        assert r.status_code == 200
        assert r.json() == {"success": True}
        mock_rag_system.session_manager.clear_session.assert_called_once_with("abc")

    def test_clear_session_failure_is_500(self, client, mock_rag_system):
        mock_rag_system.session_manager.clear_session.side_effect = RuntimeError("nope")
        assert client.delete("/api/sessions/abc").status_code == 500


class TestRootEndpoint:
    def test_root_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")

    def test_unknown_route_404(self, client):
        assert client.get("/api/nope").status_code == 404


def test_endpoints_with_real_rag_system(test_config, docs_dir, fake_client_factory):
    """End-to-end through real RAGSystem with only the Gemini client faked."""
    from unittest.mock import patch

    from rag_system import RAGSystem
    from tests.conftest import make_response

    with patch("ai_generator.genai.Client"):
        rag = RAGSystem(test_config)
    rag.add_course_folder(docs_dir)
    rag.ai_generator.client = fake_client_factory([
        make_response(calls=[("search_course_content", {"query": "widget"})]),
        make_response(text="A widget is a component."),
    ])
    c = TestClient(build_test_app(rag))

    assert c.get("/api/courses").json()["course_titles"] == ["Test Course on Widgets"]
    body = c.post("/api/query", json={"query": "What is a widget?"}).json()
    assert body["answer"] == "A widget is a component."
    assert body["sources"] and body["session_id"]
