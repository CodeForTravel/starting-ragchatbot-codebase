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
