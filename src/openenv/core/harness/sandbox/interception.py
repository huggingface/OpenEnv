# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transparent OpenAI-compatible forwarding proxy with logprob capture.

The proxy is a small FastAPI app that agent CLIs (OpenCode, Claude Code,
Codex, Pi, etc.) talk to instead of the upstream LLM endpoint. It:

1. Forwards every ``POST /v1/chat/completions`` request to the real upstream
   URL, injecting ``logprobs=true`` and ``top_logprobs=N`` so the upstream
   returns per-token logprobs.
2. Captures each ``(request, response, logprobs)`` triple to a JSON-lines
   trace file.
3. Returns the upstream response to the agent verbatim (minus the ``logprobs``
   field, which we strip so the CLI never sees anything unexpected).

The proxy is stateless beyond the trace file. One proxy instance runs per
session, normally inside the sandbox on ``localhost:7000``.

Run standalone::

    UPSTREAM_API_KEY=... python -m openenv.core.harness.sandbox.interception \\
        --upstream-url https://vllm.example/v1 \\
        --trace /tmp/trace.jsonl \\
        --port 7000
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import socket
import threading
import time
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse


CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_LOG = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    """Runtime configuration for one :class:`InterceptionProxy`."""

    upstream_url: str
    upstream_api_key: str = "intercepted"
    trace_path: str = "/tmp/opencode-proxy-trace.jsonl"
    host: str = "127.0.0.1"
    port: int = 7000
    top_logprobs: int = 5
    request_timeout_s: float = 600.0
    # Cap ``max_tokens`` before forwarding. OpenCode historically asks for very
    # large values (e.g. 32000) that exceed gpt-4o-mini's 16384 cap; capping
    # here avoids spurious upstream 400s without requiring the caller to know
    # per-model limits.
    max_tokens_cap: int | None = 16384
    # Disable Qwen-style reasoning/thinking by injecting
    # ``chat_template_kwargs.enable_thinking=false`` into forwarded requests.
    disable_thinking: bool = False
    # Override the ``model`` field on every forwarded request. Some opencode
    # builds emit a stripped model id (e.g. ``Qwen3.5-4B`` instead of the
    # ``Qwen/Qwen3.5-4B`` the upstream serves) for their internal
    # title-generation call. Setting this to the exact upstream model id
    # bypasses that mismatch.
    model_override: str | None = None


@dataclass
class TurnRecord:
    """One intercepted turn, written to the trace file as JSON-lines."""

    turn: int
    request: dict[str, Any]
    response: dict[str, Any]
    logprobs: list[dict[str, Any]] | None
    completion_tokens: list[str]
    completion_token_ids: list[int]
    per_token_logps: list[float]
    finish_reason: str | None
    latency_s: float
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)


def _build_app(cfg: ProxyConfig) -> FastAPI:
    """Construct the FastAPI app that serves one proxy session."""

    state: dict[str, Any] = {"turn": 0, "lock": asyncio.Lock()}

    # HTTP client reused across requests. ``None`` auth header — we let each
    # request carry its own ``Authorization`` populated from ``upstream_api_key``.
    client = httpx.AsyncClient(timeout=cfg.request_timeout_s)
    trace_file = open(cfg.trace_path, "a", buffering=1)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        try:
            yield
        finally:
            await client.aclose()
            trace_file.close()

    app = FastAPI(title="opencode-interception-proxy", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(CHAT_COMPLETIONS_PATH)
    async def chat_completions(request: Request) -> Response:
        raw_body = await request.body()
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400, content={"error": "invalid json body"}
            )

        forwarded_body = _prepare_forwarded_body(body, cfg)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.upstream_api_key}",
        }
        upstream_url = _resolve_upstream_url(cfg.upstream_url)

        async with state["lock"]:
            state["turn"] += 1
            turn_idx = state["turn"]

        if forwarded_body.get("stream"):
            return await _proxy_streaming(
                client=client,
                upstream_url=upstream_url,
                headers=headers,
                forwarded_body=forwarded_body,
                original_body=body,
                trace_file=trace_file,
                turn_idx=turn_idx,
            )
        return await _proxy_unary(
            client=client,
            upstream_url=upstream_url,
            headers=headers,
            forwarded_body=forwarded_body,
            original_body=body,
            trace_file=trace_file,
            turn_idx=turn_idx,
        )

    return app


def _prepare_forwarded_body(body: dict[str, Any], cfg: ProxyConfig) -> dict[str, Any]:
    """Return the body we actually send upstream.

    - Injects ``logprobs=true`` + ``top_logprobs`` so the upstream emits
      per-token logprobs.
    - Caps ``max_tokens`` / ``max_completion_tokens`` to ``max_tokens_cap``.
    - For models that reject ``max_tokens`` (e.g. gpt-5.x), translates to
      ``max_completion_tokens``.
    """
    forwarded = copy.deepcopy(body)
    forwarded.setdefault("logprobs", True)
    forwarded.setdefault("top_logprobs", cfg.top_logprobs)

    # GPT-5.x and newer: ``max_tokens`` is rejected; must use
    # ``max_completion_tokens``. Detect via model string so we don't break
    # gpt-4.x or vLLM-hosted models that accept ``max_tokens``.
    model = str(forwarded.get("model", ""))
    needs_translation = _model_uses_max_completion_tokens(model)
    if needs_translation and "max_tokens" in forwarded:
        value = forwarded.pop("max_tokens")
        forwarded.setdefault("max_completion_tokens", value)

    if cfg.max_tokens_cap is not None:
        for key in ("max_tokens", "max_completion_tokens"):
            value = forwarded.get(key)
            if isinstance(value, int) and value > cfg.max_tokens_cap:
                forwarded[key] = cfg.max_tokens_cap

    if cfg.disable_thinking:
        # vLLM applies chat_template_kwargs to the tokenizer's chat template
        # for Qwen3/Qwen3.5 models, turning off <think>...</think> generation.
        extra = forwarded.setdefault("chat_template_kwargs", {})
        extra.setdefault("enable_thinking", False)

    if cfg.model_override:
        forwarded["model"] = cfg.model_override

    return forwarded


def _model_uses_max_completion_tokens(model: str) -> bool:
    """Heuristic: ``True`` for models that reject ``max_tokens``."""
    # Strip a provider prefix opencode may have prepended (e.g. "intercepted/").
    bare = model.split("/", 1)[-1].lower()
    return bare.startswith(("gpt-5", "o1", "o3", "o4"))


def _resolve_upstream_url(upstream: str) -> str:
    """Build the fully qualified chat-completions URL from a base URL."""
    base = upstream.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}{CHAT_COMPLETIONS_PATH}"


async def _proxy_unary(
    *,
    client: httpx.AsyncClient,
    upstream_url: str,
    headers: dict[str, str],
    forwarded_body: dict[str, Any],
    original_body: dict[str, Any],
    trace_file: Any,
    turn_idx: int,
) -> Response:
    start = time.time()
    upstream_response = await client.post(
        upstream_url, content=json.dumps(forwarded_body), headers=headers
    )
    latency = time.time() - start
    try:
        response_json = upstream_response.json()
    except Exception:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get(
                "content-type", "application/json"
            ),
        )

    record = _build_turn_record(
        turn_idx=turn_idx,
        request_body=forwarded_body,
        response_json=response_json,
        latency_s=latency,
    )
    trace_file.write(record.to_json() + "\n")
    sanitized = _strip_logprobs(response_json)
    return JSONResponse(content=sanitized, status_code=upstream_response.status_code)


async def _proxy_streaming(
    *,
    client: httpx.AsyncClient,
    upstream_url: str,
    headers: dict[str, str],
    forwarded_body: dict[str, Any],
    original_body: dict[str, Any],
    trace_file: Any,
    turn_idx: int,
) -> Response:
    """Forward an SSE stream while accumulating the full response.

    Opens the upstream stream and inspects the status. On non-2xx, reads the
    full body (an error JSON, not SSE) and returns it to the caller as a
    regular JSON response — previously we silently emitted an empty
    ``text/event-stream`` which opencode interpreted as an empty assistant
    turn. Both the error body and the latency are written to the trace file
    so debugging a broken rollout doesn't require another round-trip.
    """

    start = time.time()

    # Open the stream outside the generator so we can branch on status before
    # committing to a streaming response shape.
    upstream_cm = client.stream(
        "POST",
        upstream_url,
        content=json.dumps(forwarded_body),
        headers=headers,
    )
    upstream = await upstream_cm.__aenter__()

    if upstream.status_code >= 400:
        # Upstream responded with an error body (not SSE). Read it fully and
        # return as a non-streaming JSON payload.
        error_bytes = await upstream.aread()
        await upstream_cm.__aexit__(None, None, None)
        latency = time.time() - start
        try:
            error_json = json.loads(error_bytes.decode() or "{}")
        except Exception:
            error_json = {"error": error_bytes.decode(errors="replace")[:4000]}
        record = _build_turn_record(
            turn_idx=turn_idx,
            request_body=forwarded_body,
            response_json={
                "choices": [],
                "usage": None,
                "upstream_status": upstream.status_code,
                "upstream_error": error_json,
            },
            latency_s=latency,
        )
        trace_file.write(record.to_json() + "\n")
        _LOG.warning(
            "proxy turn %s: upstream %s: %s",
            turn_idx,
            upstream.status_code,
            str(error_json)[:400],
        )
        return JSONResponse(content=error_json, status_code=upstream.status_code)

    async def _stream() -> Any:
        accumulated: dict[str, Any] = {
            "content_by_idx": {},
            "tool_calls_by_idx": {},
            "finish_by_idx": {},
            "logprobs_by_idx": {},
        }
        last_chunk: dict[str, Any] = {}
        try:
            async for line in upstream.aiter_lines():
                if not line:
                    yield "\n"
                    continue
                yield line + "\n"
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                last_chunk = chunk
                _accumulate_stream_chunk(chunk, accumulated)
        finally:
            await upstream_cm.__aexit__(None, None, None)

        latency = time.time() - start
        response_json = _assemble_streamed_response(last_chunk, accumulated)
        record = _build_turn_record(
            turn_idx=turn_idx,
            request_body=forwarded_body,
            response_json=response_json,
            latency_s=latency,
        )
        trace_file.write(record.to_json() + "\n")

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _accumulate_stream_chunk(chunk: dict[str, Any], acc: dict[str, Any]) -> None:
    for choice in chunk.get("choices", []) or []:
        idx = choice.get("index", 0)
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content:
            acc["content_by_idx"].setdefault(idx, []).append(content)
        # HF-Router's Qwen thinking mode streams the chain-of-thought under a
        # separate ``reasoning`` field (per Together/Scaleway). Accumulate it
        # so the assembled response surfaces it — otherwise it's dropped and
        # proxy_turn observability is lost for thinking-mode rollouts.
        reasoning = delta.get("reasoning")
        if reasoning:
            acc.setdefault("reasoning_by_idx", {}).setdefault(idx, []).append(reasoning)
        for tc in delta.get("tool_calls") or []:
            tc_idx = tc.get("index", 0)
            bucket = acc["tool_calls_by_idx"].setdefault(
                (idx, tc_idx),
                {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tc.get("id"):
                bucket["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                bucket["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                bucket["function"]["arguments"] += fn["arguments"]
        if choice.get("finish_reason"):
            acc["finish_by_idx"][idx] = choice["finish_reason"]
        lp = choice.get("logprobs") or {}
        content_lp = lp.get("content")
        if content_lp:
            acc["logprobs_by_idx"].setdefault(idx, []).extend(content_lp)


def _assemble_streamed_response(
    last_chunk: dict[str, Any], acc: dict[str, Any]
) -> dict[str, Any]:
    indices = sorted(
        set(acc["content_by_idx"])
        | set(acc["finish_by_idx"])
        | {k[0] for k in acc["tool_calls_by_idx"]}
        | set(acc["logprobs_by_idx"])
        | {0}
    )
    choices: list[dict[str, Any]] = []
    for idx in indices:
        tool_calls = [
            acc["tool_calls_by_idx"][k]
            for k in sorted(acc["tool_calls_by_idx"])
            if k[0] == idx
        ]
        message: dict[str, Any] = {"role": "assistant"}
        content = "".join(acc["content_by_idx"].get(idx, []))
        if content:
            message["content"] = content
        reasoning = "".join((acc.get("reasoning_by_idx") or {}).get(idx, []))
        if reasoning:
            message["reasoning"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        choice: dict[str, Any] = {
            "index": idx,
            "message": message,
            "finish_reason": acc["finish_by_idx"].get(idx),
        }
        if acc["logprobs_by_idx"].get(idx):
            choice["logprobs"] = {"content": acc["logprobs_by_idx"][idx]}
        choices.append(choice)
    return {
        "id": last_chunk.get("id", ""),
        "object": "chat.completion",
        "model": last_chunk.get("model", ""),
        "choices": choices,
        "usage": last_chunk.get("usage"),
    }


def _build_turn_record(
    *,
    turn_idx: int,
    request_body: dict[str, Any],
    response_json: dict[str, Any],
    latency_s: float,
) -> TurnRecord:
    """Extract per-token logprobs into a normalized :class:`TurnRecord`."""

    choice = (response_json.get("choices") or [{}])[0]
    logprobs_field = choice.get("logprobs") or {}
    content_lp = logprobs_field.get("content") or []

    tokens: list[str] = []
    token_ids: list[int] = []
    per_token_logps: list[float] = []
    for entry in content_lp:
        tokens.append(entry.get("token", ""))
        # OpenAI returns no raw token ids; vLLM returns them as ``token_id``.
        token_id = entry.get("token_id")
        if token_id is not None:
            token_ids.append(int(token_id))
        lp = entry.get("logprob")
        if lp is not None:
            per_token_logps.append(float(lp))

    return TurnRecord(
        turn=turn_idx,
        request=request_body,
        response=response_json,
        logprobs=content_lp,
        completion_tokens=tokens,
        completion_token_ids=token_ids,
        per_token_logps=per_token_logps,
        finish_reason=choice.get("finish_reason"),
        latency_s=latency_s,
    )


def _strip_logprobs(response_json: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the response with ``choices[*].logprobs`` removed."""

    out = dict(response_json)
    choices = out.get("choices")
    if isinstance(choices, list):
        out["choices"] = [
            {k: v for k, v in (ch or {}).items() if k != "logprobs"}
            for ch in choices
        ]
    return out


# ---------------------------------------------------------------------------
# Standalone runner (used inside the sandbox)
# ---------------------------------------------------------------------------


def serve(cfg: ProxyConfig) -> None:
    """Start the proxy and block (for use as the sandbox-side entry point)."""

    app = _build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


class InterceptionProxy:
    """Thread-backed controller for running the proxy locally.

    Used by unit tests and by any in-process driver that wants a short-lived
    proxy on the local machine. Inside a sandbox we invoke :func:`serve`
    directly via ``python -m opencode_env.interception``.
    """

    def __init__(self, cfg: ProxyConfig) -> None:
        self._cfg = cfg
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self._cfg.host}:{self._cfg.port}/v1"

    @property
    def config(self) -> ProxyConfig:
        return self._cfg

    def start(self) -> None:
        app = _build_app(self._cfg)
        config = uvicorn.Config(
            app,
            host=self._cfg.host,
            port=self._cfg.port,
            log_level="warning",
            lifespan="on",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._run_server, daemon=True
        )
        self._thread.start()
        # Wait for the server to accept connections.
        deadline = time.time() + 10
        while time.time() < deadline:
            if _port_open(self._cfg.host, self._cfg.port):
                self._ready.set()
                return
            time.sleep(0.05)
        raise RuntimeError("InterceptionProxy failed to start within 10s")

    def _run_server(self) -> None:
        assert self._server is not None
        self._server.run()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def __enter__(self) -> "InterceptionProxy":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _port_open(host: str, port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


# ---------------------------------------------------------------------------
# Trace reader (used by the session to pull captured turns back)
# ---------------------------------------------------------------------------


def read_trace(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read a proxy trace file into a list of dicts."""

    trace: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return trace
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        trace.append(json.loads(line))
    return trace


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(prog="opencode_env.interception")
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument(
        "--upstream-api-key",
        default=None,
        help=(
            "Upstream API key. Prefer OPENCODE_UPSTREAM_API_KEY so the key "
            "does not appear in process argv."
        ),
    )
    parser.add_argument("--trace", default="/tmp/opencode-proxy-trace.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--max-tokens-cap",
        type=int,
        default=None,
        help="Clamp max_tokens/max_completion_tokens on forwarded requests.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Inject chat_template_kwargs.enable_thinking=false (Qwen3/Qwen3.5).",
    )
    parser.add_argument(
        "--model-override",
        default=None,
        help="Rewrite the `model` field on every forwarded request.",
    )
    args = parser.parse_args()
    upstream_api_key = (
        args.upstream_api_key
        or os.environ.get("OPENCODE_UPSTREAM_API_KEY")
        or os.environ.get("UPSTREAM_API_KEY")
        or "intercepted"
    )

    cfg = ProxyConfig(
        upstream_url=args.upstream_url,
        upstream_api_key=upstream_api_key,
        trace_path=args.trace,
        host=args.host,
        port=args.port,
        top_logprobs=args.top_logprobs,
        request_timeout_s=args.request_timeout,
        max_tokens_cap=args.max_tokens_cap,
        disable_thinking=args.disable_thinking,
        model_override=args.model_override,
    )
    serve(cfg)


if __name__ == "__main__":
    main()
