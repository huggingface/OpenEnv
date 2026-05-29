import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.mini_swe_env.async_grpo.rollout_worker import (
    _OMITTED_TOOL_OUTPUT_MARKER,
    _clamp_max_completion_tokens,
    _compute_group_advantages,
    _coerce_token_ids,
    _extract_xml_tool_calls,
    _extract_chat_choice_logprobs,
    _extract_prompt_token_ids,
    _fit_messages_to_context_window,
    _get_tools,
    _has_answer_call,
    _is_context_window_error,
    _is_retriable_rollout_error,
    _is_terminal_non_tool_response,
    _make_chat_response,
    _normalize_chat_choice_message,
    _normalize_tool_calls,
    _retry_completion_tokens_from_context_error,
    _truncate_messages_for_prompt_budget,
    _truncate_text_middle,
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


def test_coerce_token_ids_and_prompt_token_ids_are_int_lists() -> None:
    assert _coerce_token_ids([1, "2", 3]) == [1, 2, 3]
    assert _coerce_token_ids(["bad"]) == []
    assert _extract_prompt_token_ids({"prompt_token_ids": [4, "5"]}) == [4, 5]


def test_extract_chat_choice_logprobs_pads_missing_values() -> None:
    choice = {
        "logprobs": {
            "content": [
                {"logprob": -0.1},
                {},
                {"logprob": -0.3},
            ]
        }
    }
    assert _extract_chat_choice_logprobs(choice, expected_len=4) == pytest.approx(
        [-0.1, 0.0, -0.3, 0.0]
    )


def test_extract_xml_tool_calls_recovers_qwen_style_blocks() -> None:
    content, tool_calls = _extract_xml_tool_calls(
        'Working...\n<tool_call>\n{"name": "answer", "arguments": {}}\n</tool_call>\nDone.'
    )
    assert content == "Working...\n\nDone."
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "answer"
    assert tool_calls[0]["function"]["arguments"] == "{}"


def test_normalize_chat_choice_message_parses_xml_tool_calls_from_content() -> None:
    message = _normalize_chat_choice_message(
        tokenizer=object(),
        choice={
            "message": {
                "role": "assistant",
                "content": '<tool_call>\n{"name": "answer", "arguments": {}}\n</tool_call>',
                "tool_calls": [],
            }
        },
        completion_ids=[],
    )
    assert message["content"] == ""
    assert message["tool_calls"][0]["function"]["name"] == "answer"
    assert message["tool_calls"][0]["function"]["arguments"] == "{}"


def test_clamp_max_completion_tokens_reserves_context_margin() -> None:
    assert _clamp_max_completion_tokens(
        prompt_len=2561,
        requested=1536,
        max_model_len=4096,
    ) == 1519


def test_retry_completion_tokens_from_context_error_parses_vllm_message() -> None:
    error_text = (
        "This model's maximum context length is 4096 tokens. However, you "
        "requested 1536 output tokens and your prompt contains at least 2561 "
        "input tokens, for a total of at least 4097 tokens."
    )
    assert _retry_completion_tokens_from_context_error(error_text) == (
        1519,
        4096,
        2561,
    )


def test_truncate_text_middle_preserves_edges() -> None:
    text = "A" * 80 + "B" * 80
    truncated, changed = _truncate_text_middle(text, max_chars=64)
    assert changed is True
    assert len(truncated) <= 64
    assert truncated.startswith("A" * 10)
    assert truncated.endswith("B" * 8)


def test_truncate_messages_for_prompt_budget_only_trims_long_tool_output() -> None:
    messages, tool_truncations, assistant_truncations = (
        _truncate_messages_for_prompt_budget(
            [
                {"role": "user", "content": "short"},
                {"role": "tool", "content": "x" * 200},
                {"role": "assistant", "content": "ok"},
            ],
            max_tool_message_chars=64,
            max_assistant_message_chars=64,
        )
    )
    assert tool_truncations == 1
    assert assistant_truncations == 0
    assert messages[0]["content"] == "short"
    assert len(messages[1]["content"]) <= 64
    assert messages[2]["content"] == "ok"


def test_fit_messages_to_context_window_omits_oldest_tool_output_when_needed() -> None:
    def render_prompt_ids(
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
    ) -> list[int]:
        total = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
        return list(range(total))

    messages, prompt_ids = _fit_messages_to_context_window(
        messages=[
            {"role": "user", "content": "task"},
            {"role": "tool", "content": "x" * 220},
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": "y" * 220},
        ],
        tools=None,
        render_prompt_ids=render_prompt_ids,
        requested_completion_tokens=40,
        max_model_len=116,
        max_tool_message_chars=64,
        min_tool_message_chars=32,
        max_assistant_message_chars=32,
        min_assistant_message_chars=16,
    )
    assert len(prompt_ids) <= 60
    assert messages[1]["content"] == _OMITTED_TOOL_OUTPUT_MARKER
    assert messages[3]["content"] == _OMITTED_TOOL_OUTPUT_MARKER


def test_context_window_errors_are_not_marked_retriable() -> None:
    exc = RuntimeError(
        "vllm 400: This model's maximum context length is 8192 tokens. "
        "However, you requested 1 output tokens and your prompt contains "
        "at least 8192 input tokens."
    )
    assert _is_context_window_error(exc) is True
    assert _is_retriable_rollout_error(exc) is False


def test_transient_sandbox_errors_are_marked_retriable() -> None:
    exc = RuntimeError("HF sandbox tunnel failed: 429 Too Many Requests")
    assert _is_context_window_error(exc) is False
    assert _is_retriable_rollout_error(exc) is True


def test_compute_group_advantages_zscores_rewards() -> None:
    advantages, reward_mean, reward_std = _compute_group_advantages(
        [0.0, 0.0, 1.0, 1.0]
    )
    assert reward_mean == pytest.approx(0.5)
    assert reward_std == pytest.approx(0.5)
    assert advantages == pytest.approx([-1.0, -1.0, 1.0, 1.0], abs=1e-6)


def test_compute_group_advantages_returns_zero_for_constant_rewards() -> None:
    advantages, reward_mean, reward_std = _compute_group_advantages(
        [1.0, 1.0, 1.0, 1.0]
    )
    assert reward_mean == pytest.approx(1.0)
    assert reward_std == pytest.approx(0.0)
    assert advantages == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-6)
