"""External-behavior tests: API requests made, tools executed, text returned."""

from unittest.mock import MagicMock, patch

import pytest

from ai_generator import AIGenerator
from search_tools import CourseOutlineTool, CourseSearchTool
from tests.conftest import make_response

TOOLS = [
    CourseSearchTool(MagicMock()).get_tool_definition(),
    CourseOutlineTool(MagicMock()).get_tool_definition(),
]
SEARCH = "search_course_content"
OUTLINE = "get_course_outline"


@pytest.fixture
def gen(fake_client_factory):
    def build(responses):
        with patch("ai_generator.genai.Client") as C:
            C.return_value = fake_client_factory(responses)
            return AIGenerator("k", "some-model")

    return build


@pytest.fixture
def tm():
    m = MagicMock()
    m.execute_tool.return_value = "tool output"
    return m


def requests_of(g):
    return g.client.models.generate_content.call_args_list


def is_text_only(call):
    cfg = call.kwargs["config"]
    return (
        cfg.tool_config is not None
        and cfg.tool_config.function_calling_config.mode == "NONE"
    )


def roles(call):
    return [c.role for c in call.kwargs["contents"]]


def response_parts(call):
    return [
        p.function_response
        for c in call.kwargs["contents"]
        for p in c.parts
        if getattr(p, "function_response", None)
    ]


def test_build_config_valid_with_real_sdk():
    cfg = AIGenerator("k", "m")._build_config("sys", TOOLS)
    assert [d.name for d in cfg.tools[0].function_declarations] == [SEARCH, OUTLINE]
    assert cfg.automatic_function_calling.disable is True
    assert cfg.temperature == 0


def test_no_tool_call_single_request(gen, tm):
    g = gen([make_response(text="Paris")])
    assert g.generate_response("capital?", tools=TOOLS, tool_manager=tm) == "Paris"
    assert len(requests_of(g)) == 1
    tm.execute_tool.assert_not_called()


def test_one_tool_round_then_answer(gen, tm):
    g = gen(
        [
            make_response(calls=[(SEARCH, {"query": "w", "course_name": "W"})]),
            make_response(text="final answer"),
        ]
    )
    assert g.generate_response("Q", tools=TOOLS, tool_manager=tm) == "final answer"
    tm.execute_tool.assert_called_once_with(SEARCH, query="w", course_name="W")
    reqs = requests_of(g)
    assert len(reqs) == 2
    assert not is_text_only(reqs[1])  # model may still chain another tool
    assert roles(reqs[1]) == ["user", "model", "user"]
    assert response_parts(reqs[1])[0].response == {"result": "tool output"}


def test_two_sequential_rounds_preserve_context(gen, tm):
    tm.execute_tool.side_effect = ["Lesson 4: Servers", "Other course info"]
    g = gen(
        [
            make_response(calls=[(OUTLINE, {"course_name": "X"})]),
            make_response(calls=[(SEARCH, {"query": "Servers"})]),
            make_response(text="complete answer"),
        ]
    )
    assert g.generate_response("Q", tools=TOOLS, tool_manager=tm) == "complete answer"

    assert [c.args for c in tm.execute_tool.call_args_list] == [(OUTLINE,), (SEARCH,)]
    assert [c.kwargs for c in tm.execute_tool.call_args_list] == [
        {"course_name": "X"},
        {"query": "Servers"},
    ]
    reqs = requests_of(g)
    assert len(reqs) == 3
    assert [is_text_only(r) for r in reqs] == [False, False, True]
    assert roles(reqs[1]) == ["user", "model", "user"]
    assert roles(reqs[2]) == ["user", "model", "user", "model", "user"]
    # round-1 result is still visible in the final request, in order
    assert [r.name for r in response_parts(reqs[2])] == [OUTLINE, SEARCH]
    assert response_parts(reqs[2])[0].response == {"result": "Lesson 4: Servers"}


def test_cap_of_two_rounds(gen, tm):
    g = gen(
        [
            make_response(calls=[(SEARCH, {"query": "a"})]),
            make_response(calls=[(SEARCH, {"query": "b"})]),
            make_response(calls=[(SEARCH, {"query": "c"})], text="best effort"),
        ]
    )
    out = g.generate_response("Q", tools=TOOLS, tool_manager=tm)
    assert out == "best effort"  # text of the forced final request; its call is ignored
    assert tm.execute_tool.call_count == 2
    reqs = requests_of(g)
    assert len(reqs) == 3 and is_text_only(reqs[2])


def test_parallel_calls_are_one_round(gen, tm):
    g = gen(
        [
            make_response(calls=[(SEARCH, {"query": "a"}), (SEARCH, {"query": "b"})]),
            make_response(calls=[(OUTLINE, {"course_name": "X"})]),
            make_response(text="done"),
        ]
    )
    assert g.generate_response("Q", tools=TOOLS, tool_manager=tm) == "done"
    assert tm.execute_tool.call_count == 3  # a second round was still allowed
    reqs = requests_of(g)
    assert len(response_parts(reqs[1])) == 2
    assert (
        len(reqs[1].kwargs["contents"][-1].parts) == 2
    )  # both results in a single user turn


def test_tool_exception_ends_loop_gracefully(gen, tm):
    tm.execute_tool.side_effect = RuntimeError("db down")
    g = gen(
        [
            make_response(calls=[(SEARCH, {"query": "a"})]),
            make_response(text="Sorry, search is unavailable."),
        ]
    )
    out = g.generate_response("Q", tools=TOOLS, tool_manager=tm)
    assert out == "Sorry, search is unavailable."
    reqs = requests_of(g)
    assert len(reqs) == 2 and is_text_only(reqs[1])
    assert "db down" in response_parts(reqs[1])[0].response["error"]


def test_tool_exception_in_round_two(gen, tm):
    tm.execute_tool.side_effect = ["ok", RuntimeError("boom")]
    g = gen(
        [
            make_response(calls=[(OUTLINE, {"course_name": "X"})]),
            make_response(calls=[(SEARCH, {"query": "a"})]),
            make_response(text="partial answer"),
        ]
    )
    assert g.generate_response("Q", tools=TOOLS, tool_manager=tm) == "partial answer"
    reqs = requests_of(g)
    assert len(reqs) == 3 and is_text_only(reqs[2])
    assert len(response_parts(reqs[2])) == 2


def test_failure_in_one_parallel_call_still_answers_all(gen, tm):
    tm.execute_tool.side_effect = [RuntimeError("bad"), "fine"]
    g = gen(
        [
            make_response(calls=[(SEARCH, {"query": "a"}), (SEARCH, {"query": "b"})]),
            make_response(text="answer"),
        ]
    )
    assert g.generate_response("Q", tools=TOOLS, tool_manager=tm) == "answer"
    reqs = requests_of(g)
    parts = response_parts(reqs[1])
    assert (
        len(parts) == 2
        and "error" in parts[0].response
        and parts[1].response == {"result": "fine"}
    )
    assert is_text_only(reqs[1])


def test_tool_call_with_none_args(gen, tm):
    g = gen([make_response(calls=[(OUTLINE, None)]), make_response(text="ok")])
    assert g.generate_response("Q", tools=TOOLS, tool_manager=tm) == "ok"
    tm.execute_tool.assert_called_once_with(OUTLINE)


def test_silent_model_after_tool_round_retries_then_falls_back(gen, tm):
    g = gen(
        [
            make_response(calls=[(SEARCH, {"query": "q"})]),
            make_response(text=""),  # round-2 request: no calls, no text
            make_response(text=""),
            make_response(text=None),
        ]
    )  # forced final + its retry
    out = g.generate_response("Q", tools=TOOLS, tool_manager=tm)
    assert out.startswith("Sorry")
    assert len(requests_of(g)) == 4


def test_no_tool_manager_returns_text_without_crashing(gen):
    g = gen([make_response(calls=[(SEARCH, {"query": "q"})], text="hello")])
    assert g.generate_response("Q", tools=TOOLS, tool_manager=None) == "hello"
    assert len(requests_of(g)) == 1


def test_history_in_system_prompt(gen):
    g = gen([make_response(text="hi")])
    g.generate_response("Q", conversation_history="User: a\nAssistant: b")
    cfg = requests_of(g)[0].kwargs["config"]
    assert "Previous conversation:\nUser: a" in cfg.system_instruction


def test_system_prompt_allows_two_calls():
    p = AIGenerator.SYSTEM_PROMPT
    assert "One tool call per query maximum" not in p
    assert "2 sequential" in p
