"""Synthetic SSE: capture non-streaming, reply streaming.

Every coding harness streams. That is not a preference we can talk them out of -- opencode, codex and
claude-code all drive their UI off token deltas -- and a harness that asks for SSE and receives a
plain JSON body does not error. Its stream parser simply yields nothing, so opencode reports
`step-finish reason:"unknown"` with zero tokens and no message, having received a *perfectly valid*
tool call. Capture looks flawless from our side and the agent does nothing. That failure cost real
debugging time, hence this module.

Meanwhile capture wants the opposite: one complete response, because token ids and logprobs arrive
whole and reassembling them from deltas is error-prone in exactly the way that silently corrupts
training data.

So we do both. Fetch non-streaming upstream, store that for capture, then replay the complete
response to the client as a synthetic SSE stream. The client cannot tell the difference; we never
parse deltas.

The per-dialect machinery is reused from Polar's transformers (`create_stream_state` /
`transform_stream_chunk`), which are dependency-clean. Only the small formatting helpers are ported
here, because in Polar they live in `server.py` next to its node/dispatcher layer.
"""

from __future__ import annotations

import json
from typing import Any

from .detection import APIType
from .dialects.base import BaseTransformer

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Without this an intermediate proxy (nginx, cloudflared) may buffer the whole stream and hand
    # it over at once, which defeats the point for the client even though the bytes are correct.
    "X-Accel-Buffering": "no",
}


def _format_typed_events(events: list[dict[str, Any]]) -> str:
    """Anthropic and OpenAI-Responses both use named events (`event: <type>`)."""
    return "".join(
        f"event: {event.get('type', 'unknown')}\ndata: {json.dumps(event, default=str)}\n\n"
        for event in events
    )


def _format_data_only(chunk: dict[str, Any]) -> str:
    """OpenAI chat-completions and Google use bare `data:` lines."""
    return f"data: {json.dumps(chunk, default=str)}\n\n"


def format_events(api_type: APIType, events: list[dict[str, Any]]) -> str:
    """Format EVERY event. Dropping any of them truncates the stream.

    This used to emit only `events[0]` for the data-only dialects, which is invisible for
    chat-completions (we synthesise exactly one chunk) but silently truncates Google. gemini-cli calls
    `:streamGenerateContent?alt=sse`, whose stream state emits several events, and receiving only the
    first produced:

        Error: Incomplete JSON segment at the end
            at ApiClient.processStreamResponse_1 (@google/gemini-cli/...)

    A 200 with a truncated body, which is the failure shape this whole layer keeps running into.
    """
    if api_type in (APIType.ANTHROPIC, APIType.OPENAI_RESPONSES):
        return _format_typed_events(events)
    return "".join(_format_data_only(event) for event in events)


def format_chunk(
    api_type: APIType,
    transformer: BaseTransformer,
    chunk: dict[str, Any],
    original_request: dict[str, Any],
    *,
    is_first: bool,
) -> str:
    """Fallback for transformers with no stream-state machine: one-shot chunk transform."""
    transformed = transformer.transform_stream_chunk(
        chunk, original_request, is_first=is_first
    )
    if api_type == APIType.ANTHROPIC:
        return _format_typed_events(transformed)
    if api_type == APIType.OPENAI_RESPONSES:
        events = (
            transformed
            if isinstance(transformed, list)
            else ([transformed] if transformed else [])
        )
        return _format_typed_events(events)
    return _format_data_only(transformed)


def response_to_chunk(response: dict[str, Any]) -> dict[str, Any]:
    """Repackage a complete chat completion as a single `chat.completion.chunk` delta.

    One chunk carrying everything, rather than a plausible-looking token-by-token replay. The client
    only needs a well-formed stream, and faking granularity would invent timing information we do not
    have. Tool calls have to be re-indexed into delta form: streaming clients accumulate
    `tool_calls[i].function.arguments` across chunks, so the `index` field is required even when
    there is exactly one chunk to accumulate.
    """
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    tool_calls_delta = [
        {
            "index": i,
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "function": {
                "name": (tc.get("function") or {}).get("name", ""),
                "arguments": (tc.get("function") or {}).get("arguments", ""),
            },
        }
        for i, tc in enumerate(message.get("tool_calls") or [])
    ]

    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("content") is not None:
        delta["content"] = message["content"]
    # Reasoning models put thinking here; harnesses that render it expect it in the delta.
    for key in ("reasoning_content", "reasoning"):
        if message.get(key) is not None:
            delta["reasoning_content"] = message[key]
            break
    if tool_calls_delta:
        delta["tool_calls"] = tool_calls_delta

    return {
        "id": response.get("id"),
        "object": "chat.completion.chunk",
        "created": response.get("created"),
        "model": response.get("model"),
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": choice.get("finish_reason")}
        ],
        # Clients that sent stream_options.include_usage expect this; we dropped the option upstream
        # (vLLM rejects it with stream=False) but the response carries usage anyway, so honour it.
        "usage": response.get("usage"),
    }


async def replay(
    api_type: APIType,
    transformer: BaseTransformer,
    response: dict[str, Any],
    original_request: dict[str, Any],
):
    """Async generator yielding the SSE body for one complete upstream response."""
    chunk = response_to_chunk(response)
    stream_state = transformer.create_stream_state(original_request)

    if stream_state is not None:
        # Dialects with real state machines (Anthropic, Responses, Google) emit a sequence of
        # lifecycle events -- message_start, content_block_delta, message_stop and friends -- and the
        # client will reject a stream that skips them, so finalize() is not optional.
        events = stream_state.process_chunk(chunk, is_first=True)
        if events:
            yield format_events(api_type, events)
        final_events = stream_state.finalize()
        if final_events:
            yield format_events(api_type, final_events)
    else:
        output = format_chunk(
            api_type, transformer, chunk, original_request, is_first=True
        )
        if output:
            yield output

    if api_type == APIType.OPENAI_CHAT:
        # Only chat-completions uses this sentinel. The typed-event dialects signal completion with
        # their own terminal event, and an extra [DONE] there is a parse error.
        yield "data: [DONE]\n\n"
