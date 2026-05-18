from mini_swe_env.async_grpo.rollout_worker import (
    _get_tools,
    _has_answer_call,
    _is_terminal_non_tool_response,
    _make_chat_response,
    _normalize_tool_calls,
)


def test_normalize_tool_calls_serializes_arguments_to_json_string() -> None:
    calls = _normalize_tool_calls(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "answer", "arguments": {}},
            }
        ]
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "answer"
    assert calls[0]["function"]["arguments"] == "{}"


def test_terminal_detection_requires_stop_and_no_tool_calls() -> None:
    terminal = _make_chat_response(
        {"role": "assistant", "content": "Done."},
        model="qwen",
        finish_reason="stop",
    )
    assert _is_terminal_non_tool_response(terminal) is True

    with_tools = _make_chat_response(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "answer", "arguments": "{}"},
                }
            ],
        },
        model="qwen",
        finish_reason="stop",
    )
    assert _is_terminal_non_tool_response(with_tools) is False
    assert _has_answer_call(with_tools) is True


def test_get_tools_from_intercept_or_body() -> None:
    tool_schema = {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "submit",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert _get_tools({"tools": [tool_schema]}) == [tool_schema]
    assert _get_tools({"body": {"tools": [tool_schema]}}) == [tool_schema]
    assert _get_tools({}) is None
