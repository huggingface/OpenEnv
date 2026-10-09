# SPDX-License-Identifier: BSD-3-Clause

"""Recording relays: every call out of the agent zone crosses one of these.

Three relays share one HTTP proxy core:

- the env relay forwards any HTTP request or websocket to the env server's Unix
  socket, unchanged, and records it;
- the model proxy swaps the harness's placeholder key for the real one and
  records each request with the assistant text it produced;
- a service relay forwards to a hidden service's Unix socket and records it.

Callers see the upstream's status, headers (minus hop-by-hop ones) and bytes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from .trace import TraceSealedError

BODY_LIMIT = 64 * 1024
"""Recorded bodies are cut to this many bytes and flagged `truncated`."""

HTTP_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

# Set by the client library or by the websocket handshake, never forwarded.
_REQUEST_DROP = HOP_BY_HOP | {"host", "content-length"}
_WS_DROP = _REQUEST_DROP | {
    "authorization",
    "sec-websocket-key",
    "sec-websocket-version",
    "sec-websocket-extensions",
    "sec-websocket-protocol",
}
_INFRA_HEADERS = frozenset({"server", "via", "x-powered-by"})

JsonHook = Callable[[dict], Awaitable[None]]


def _append(recorder, kind: str, source: str, data: dict, **where) -> bool:
    """Record one event. `False` once the trace is sealed: the call must not pass."""
    try:
        recorder.append(kind, source, data, **where)
    except TraceSealedError:
        return False
    return True


def episode_unavailable(**headers: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": "episode_unavailable"}}, status_code=503, headers=headers
    )


def _clean(headers: Iterable[tuple[bytes, bytes]], drop: frozenset[str] | set[str]):
    return [(k, v) for k, v in headers if k.decode("latin-1").lower() not in drop]


def _body_field(raw: bytes, truncated: bool = False) -> dict[str, Any]:
    """`{"body": ..., "truncated": ...}` for a trace record: JSON if it parses."""
    truncated = truncated or len(raw) > BODY_LIMIT
    raw = raw[:BODY_LIMIT]
    text = raw.decode("utf-8", errors="replace")
    body: Any = text
    if not truncated and text:
        with contextlib.suppress(ValueError):
            body = json.loads(text)
    return {"body": body, "truncated": truncated}


def _parse_json(payload: str | bytes) -> Any:
    try:
        return json.loads(payload)
    except (ValueError, TypeError):
        return None


def _target(scope: dict) -> str:
    """The request target exactly as the caller sent it."""
    raw = scope.get("raw_path") or scope["path"].encode()
    target = raw.decode("latin-1")
    if scope.get("query_string"):
        target += "?" + scope["query_string"].decode("latin-1")
    return target


class _Tee:
    """Async iterator over an upstream body that keeps a copy for the trace."""

    def __init__(
        self,
        upstream: httpx.Response,
        limit: int | None,
        done: Callable[[bytes, bool], Awaitable[None] | None],
    ):
        self._upstream = upstream
        self._limit = limit
        self._done = done
        self._buf = bytearray()
        self._truncated = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._upstream.aiter_raw():
                if self._limit is None or len(self._buf) < self._limit:
                    self._buf += chunk
                    if self._limit is not None and len(self._buf) > self._limit:
                        del self._buf[self._limit :]
                        self._truncated = True
                else:
                    self._truncated = True
                yield chunk
        finally:
            await self._upstream.aclose()
            result = self._done(bytes(self._buf), self._truncated)
            if asyncio.iscoroutine(result):
                await result


async def _forward(
    client: httpx.AsyncClient,
    request: Request,
    body: bytes,
    *,
    drop: frozenset[str] | set[str],
    add: list[tuple[bytes, bytes]] = (),
    strip_response: frozenset[str] | set[str] = HOP_BY_HOP,
    extra_response: list[tuple[bytes, bytes]] = (),
    limit: int | None = BODY_LIMIT,
    done: Callable[[int, bytes, bool, str], Awaitable[None] | None],
) -> Response:
    """Send `request` upstream and stream the answer back, teeing it to `done`."""
    headers = _clean(request.headers.raw, drop) + list(add)
    upstream_request = client.build_request(
        request.method, _target(request.scope), headers=headers, content=body
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError:
        result = done(502, b"", False, "")
        if asyncio.iscoroutine(result):
            await result
        return JSONResponse(
            {"error": {"code": "upstream_unavailable"}},
            status_code=502,
            headers=dict(extra_response),
        )
    content_type = upstream.headers.get("content-type", "")
    tee = _Tee(
        upstream,
        limit,
        lambda buf, truncated: done(upstream.status_code, buf, truncated, content_type),
    )
    response = StreamingResponse(tee, status_code=upstream.status_code)
    response.raw_headers = _clean(upstream.headers.raw, strip_response) + list(
        extra_response
    )
    return response


def _upstream_client(uds: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=uds),
        base_url="http://env",
        timeout=httpx.Timeout(None, connect=10.0),
    )


class EnvRelay:
    """The env relay's endpoints, reusable by apps that put a guard in front.

    Args:
        upstream_uds (`str`):
            The env server's Unix socket.
        recorder (`TraceRecorder`):
            Where records go.
        zone (`str`, *optional*, defaults to `"agent"`):
            Zone named on every record.
        container (`str`, *optional*, defaults to `"env"`):
            Container named on every record.
        on_ws_out (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called with every JSON object the env server sends on `/ws`.
        on_ws_in (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called with every JSON object a caller sends on `/ws`.
        on_http_response (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called with the `env.response` record data of every non-MCP request.
    """

    def __init__(
        self,
        upstream_uds: str,
        recorder,
        *,
        zone: str = "agent",
        container: str = "env",
        on_ws_out: JsonHook | None = None,
        on_ws_in: JsonHook | None = None,
        on_http_response: JsonHook | None = None,
    ):
        self.upstream_uds = upstream_uds
        self.recorder = recorder
        self.zone = zone
        self.container = container
        self.on_ws_out = on_ws_out
        self.on_ws_in = on_ws_in
        self.on_http_response = on_http_response
        self.client = _upstream_client(upstream_uds)

    def _record(self, kind: str, data: dict) -> bool:
        return _append(
            self.recorder,
            kind,
            "env_relay",
            data,
            zone=self.zone,
            container=self.container,
        )

    async def http(self, request: Request) -> Response:
        body = await request.body()
        path = request.url.path
        is_mcp = request.method == "POST" and path.rstrip("/") == "/mcp"
        if is_mcp:
            call = _parse_json(body)
            if isinstance(call, dict):
                data = {k: call.get(k) for k in ("method", "params", "id")}
            else:
                data = {"batch": call} if isinstance(call, list) else _body_field(body)
            recorded = self._record("mcp.call", data)
        else:
            recorded = self._record(
                "env.request",
                {"method": request.method, "path": path, **_body_field(body)},
            )
        if not recorded:
            return episode_unavailable()

        async def done(status: int, buf: bytes, truncated: bool, _ct: str) -> None:
            if is_mcp:
                result = _parse_json(buf) if not truncated else None
                data: dict[str, Any] = {"status": status}
                if isinstance(result, dict):
                    data.update({k: result.get(k) for k in ("id", "result", "error")})
                else:
                    data.update(_body_field(buf, truncated))
                self._record("mcp.result", data)
                return
            data = {
                "method": request.method,
                "path": path,
                "status": status,
                **_body_field(buf, truncated),
            }
            self._record("env.response", data)
            if self.on_http_response is not None:
                await self.on_http_response(data)

        return await _forward(
            self.client,
            request,
            body,
            drop=_REQUEST_DROP | {"authorization"},
            done=done,
        )

    async def websocket(self, websocket: WebSocket) -> None:
        from websockets.asyncio.client import unix_connect
        from websockets.exceptions import ConnectionClosed, InvalidHandshake

        path = websocket.url.path
        is_gym = path.rstrip("/") == "/ws"
        protocols = [
            p.strip()
            for p in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if p.strip()
        ]
        headers = [
            (k.decode("latin-1"), v.decode("latin-1"))
            for k, v in _clean(websocket.headers.raw, _WS_DROP)
        ]
        try:
            upstream = await unix_connect(
                self.upstream_uds,
                uri="ws://env" + _target(websocket.scope),
                additional_headers=headers,
                subprotocols=protocols or None,
                max_size=None,
                compression=None,
                open_timeout=10,
                ping_interval=None,
            )
        except (OSError, InvalidHandshake, asyncio.TimeoutError):
            await websocket.close(code=1011)
            return

        await websocket.accept(subprotocol=upstream.subprotocol)
        meta = {"path": path}

        async def client_to_upstream() -> None:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    await upstream.close(code=message.get("code") or 1000)
                    return
                payload = message.get("text")
                if payload is None:
                    payload = message.get("bytes") or b""
                parsed = _parse_json(payload)
                if not self._record("ws.in", {**meta, **_frame(payload, parsed)}):
                    raise _Sealed
                if is_gym and self.on_ws_in is not None and isinstance(parsed, dict):
                    await self.on_ws_in(parsed)
                await upstream.send(payload)

        async def upstream_to_client() -> None:
            try:
                async for payload in upstream:
                    parsed = _parse_json(payload)
                    if not self._record("ws.out", {**meta, **_frame(payload, parsed)}):
                        raise _Sealed
                    if isinstance(payload, str):
                        await websocket.send_text(payload)
                    else:
                        await websocket.send_bytes(payload)
                    if (
                        is_gym
                        and self.on_ws_out is not None
                        and isinstance(parsed, dict)
                    ):
                        await self.on_ws_out(parsed)
            except ConnectionClosed:
                pass
            code = upstream.close_code or 1000
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.close(code=code, reason=upstream.close_reason or "")

        tasks = [
            asyncio.ensure_future(client_to_upstream()),
            asyncio.ensure_future(upstream_to_client()),
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in done:
                exc = task.exception()
                if isinstance(exc, _Sealed):
                    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                        await websocket.close(code=1008, reason="episode_unavailable")
                elif exc is not None and not isinstance(
                    exc, (ConnectionClosed, WebSocketDisconnect, RuntimeError)
                ):
                    raise exc
        finally:
            await upstream.close()

    async def aclose(self) -> None:
        await self.client.aclose()


class _Sealed(Exception):
    """A frame could not be recorded because the trace is sealed."""


def _frame(payload: str | bytes, parsed: Any) -> dict[str, Any]:
    if parsed is not None:
        return {"json": parsed}
    if isinstance(payload, bytes):
        return {"bytes": len(payload)}
    return _body_field(payload.encode())


def env_relay_app(
    upstream_uds: str,
    recorder,
    *,
    zone: str = "agent",
    container: str = "env",
    on_ws_out: JsonHook | None = None,
    on_ws_in: JsonHook | None = None,
    on_http_response: JsonHook | None = None,
) -> Starlette:
    """Relay any request or websocket to the env server, recording each call.

    `POST /mcp` is recorded as `mcp.call` and `mcp.result`; other requests as
    `env.request` and `env.response`; websocket frames as `ws.in` and `ws.out`.
    The caller's `Authorization` header is never forwarded.

    Args:
        upstream_uds (`str`):
            The env server's Unix socket.
        recorder (`TraceRecorder`):
            Where records go.
        zone (`str`, *optional*, defaults to `"agent"`):
            Zone named on every record.
        container (`str`, *optional*, defaults to `"env"`):
            Container named on every record.
        on_ws_out (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called with every JSON object the env server sends on `/ws`, for
            example to notice `done: true`.
        on_ws_in (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called with every JSON object a caller sends on `/ws`.
        on_http_response (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called with the `env.response` record data of every non-MCP request.

    Returns:
        `Starlette`: The relay. `app.state.relay.aclose()` closes its client.
    """
    relay = EnvRelay(
        upstream_uds,
        recorder,
        zone=zone,
        container=container,
        on_ws_out=on_ws_out,
        on_ws_in=on_ws_in,
        on_http_response=on_http_response,
    )
    app = Starlette(
        routes=[
            Route("/{path:path}", relay.http, methods=HTTP_METHODS),
            WebSocketRoute("/{path:path}", relay.websocket),
        ]
    )
    app.state.relay = relay
    return app


# --- model proxy ------------------------------------------------------------

REQUEST_ID_HEADER = "x-openenvd-request-id"


def _sse_data(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        if line.startswith("data:"):
            value = line[5:]
            out.append(value[1:] if value.startswith(" ") else value)
    return out


def extract_text(
    provider: str, body: str, *, stream: bool
) -> tuple[str | None, str | None]:
    """Best-effort `(text, thinking)` from a model response body.

    Args:
        provider (`str`):
            `anthropic` (Messages API) or `openai` (chat completions).
        body (`str`):
            The JSON body, or the concatenated SSE `data:` payloads.
        stream (`bool`):
            Whether `body` holds SSE payloads, one per line.

    Returns:
        `tuple`: The assistant text and thinking text; `None` where absent.
    """
    text: list[str] = []
    thinking: list[str] = []
    events = (
        [_parse_json(line) for line in body.splitlines() if line and line != "[DONE]"]
        if stream
        else [_parse_json(body)]
    )
    for event in events:
        if not isinstance(event, dict):
            continue
        if provider == "anthropic":
            if stream:
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text.append(delta.get("text") or "")
                    elif delta.get("type") == "thinking_delta":
                        thinking.append(delta.get("thinking") or "")
            else:
                for block in event.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text.append(block.get("text") or "")
                    elif block.get("type") == "thinking":
                        thinking.append(block.get("thinking") or "")
        else:
            choices = event.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                continue
            part = choices[0].get("delta" if stream else "message") or {}
            content = part.get("content")
            if isinstance(content, list):
                content = "".join(
                    c.get("text") or "" for c in content if isinstance(c, dict)
                )
            if content:
                text.append(content)
            reasoning = part.get("reasoning_content") or part.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                thinking.append(reasoning)
    return ("".join(text) if text else None, "".join(thinking) if thinking else None)


def model_proxy_app(
    upstream_base_url: str,
    real_key: str,
    recorder,
    *,
    provider: str = "anthropic",
    placeholder_key: str = "openenvd-placeholder",
) -> Starlette:
    """Proxy model calls, swapping the harness's placeholder key for the real one.

    A request must carry `placeholder_key` (as `x-api-key` or a bearer token);
    anything else gets a 401 and never reaches the provider. Responses stream
    back as they arrive and are recorded once complete. The caller gets the
    record's `request_id` in the `x-openenvd-request-id` header. The real key is
    never recorded.

    Args:
        upstream_base_url (`str`):
            The provider's base URL, for example `https://api.anthropic.com`.
        real_key (`str`):
            The provider credential.
        recorder (`TraceRecorder`):
            Where records go.
        provider (`str`, *optional*, defaults to `"anthropic"`):
            `anthropic` (sent as `x-api-key`) or `openai` (sent as a bearer token).
        placeholder_key (`str`, *optional*, defaults to `"openenvd-placeholder"`):
            The only key the harness holds.

    Returns:
        `Starlette`: The proxy. `app.state.aclose()` closes its client.
    """
    if provider not in ("anthropic", "openai"):
        raise ValueError(f"unknown model provider {provider!r}")
    client = httpx.AsyncClient(
        base_url=upstream_base_url, timeout=httpx.Timeout(None, connect=10.0)
    )
    if provider == "anthropic":
        auth = [(b"x-api-key", real_key.encode())]
    else:
        auth = [(b"authorization", b"Bearer " + real_key.encode())]
    drop = _REQUEST_DROP | {"authorization", "x-api-key", "accept-encoding"}
    expected = placeholder_key.encode()

    def presented(request: Request) -> bytes:
        key = request.headers.get("x-api-key")
        if key is None:
            scheme, _, token = request.headers.get("authorization", "").partition(" ")
            key = token if scheme.lower() == "bearer" else ""
        return key.encode()

    def record(kind: str, data: dict) -> bool:
        return _append(recorder, kind, "model_proxy", data)

    async def proxy(request: Request) -> Response:
        if not hmac.compare_digest(presented(request), expected):
            return JSONResponse(
                {"error": {"type": "authentication_error", "message": "invalid key"}},
                status_code=401,
            )
        request_id = uuid.uuid4().hex
        body = await request.body()
        if not record(
            "model.request",
            {
                "request_id": request_id,
                "provider": provider,
                "method": request.method,
                "path": request.url.path,
                **_body_field(body),
            },
        ):
            return episode_unavailable()

        def done(status: int, buf: bytes, _truncated: bool, content_type: str) -> None:
            raw = buf.decode("utf-8", errors="replace")
            stream = "text/event-stream" in content_type
            if stream:
                raw = "\n".join(_sse_data(raw))
            text, thinking = extract_text(provider, raw, stream=stream)
            record(
                "model.response",
                {
                    "request_id": request_id,
                    "status": status,
                    "stream": stream,
                    "text": text,
                    "thinking": thinking,
                    "raw": raw,
                },
            )

        return await _forward(
            client,
            request,
            body,
            drop=drop,
            add=auth + [(b"accept-encoding", b"identity")],
            extra_response=[(REQUEST_ID_HEADER.encode(), request_id.encode())],
            limit=None,
            done=done,
        )

    app = Starlette(routes=[Route("/{path:path}", proxy, methods=HTTP_METHODS)])
    app.state.aclose = client.aclose
    return app


# --- service relay ----------------------------------------------------------


def service_relay_app(service: str, upstream_uds: str, recorder) -> Starlette:
    """Relay HTTP calls to a hidden service, recording each one.

    Response headers that could reveal infrastructure (`server`, `via`,
    `x-powered-by`, `x-forwarded-*`) are dropped.

    Args:
        service (`str`):
            The service's name, recorded on every record.
        upstream_uds (`str`):
            The service's Unix socket.
        recorder (`TraceRecorder`):
            Where records go.

    Returns:
        `Starlette`: The relay. `app.state.aclose()` closes its client.
    """
    client = _upstream_client(upstream_uds)

    def record(kind: str, data: dict) -> bool:
        return _append(
            recorder, kind, "service_relay", data, zone="services", container=service
        )

    class _Strip(frozenset):
        def __contains__(self, name: object) -> bool:
            return (
                super().__contains__(name)
                or isinstance(name, str)
                and name.startswith("x-forwarded-")
            )

    strip = _Strip(HOP_BY_HOP | _INFRA_HEADERS)

    async def proxy(request: Request) -> Response:
        body = await request.body()
        meta = {"service": service, "method": request.method, "path": request.url.path}
        if not record("service.request", {**meta, **_body_field(body)}):
            return episode_unavailable()

        def done(status: int, buf: bytes, truncated: bool, _ct: str) -> None:
            record(
                "service.response",
                {**meta, "status": status, **_body_field(buf, truncated)},
            )

        return await _forward(
            client,
            request,
            body,
            drop=_REQUEST_DROP | {"authorization"},
            strip_response=strip,
            done=done,
        )

    app = Starlette(routes=[Route("/{path:path}", proxy, methods=HTTP_METHODS)])
    app.state.aclose = client.aclose
    return app


# --- serving ----------------------------------------------------------------


class _Server(uvicorn.Server):
    """A uvicorn server that leaves signal handling to openenvd."""

    serve_task: asyncio.Task | None = None

    @contextlib.contextmanager
    def capture_signals(self):
        yield


async def _start(config: uvicorn.Config) -> _Server:
    server = _Server(config)
    server.serve_task = asyncio.ensure_future(server.serve())
    while not server.started:
        if server.serve_task.done():
            try:
                server.serve_task.result()
            except SystemExit as exc:
                raise OSError(f"server failed to start ({exc.code})") from None
            raise OSError("server exited before it started")
        await asyncio.sleep(0.005)
    return server


async def serve_asgi_on_unix(app, path: str) -> uvicorn.Server:
    """Start serving `app` on a Unix socket in the running loop.

    Args:
        app (`ASGIApp`):
            The application.
        path (`str`):
            Socket path. A stale socket there is replaced. Mode is set to `0o666`.

    Returns:
        `uvicorn.Server`: The started server. Stop it with [`stop_server`].
    """
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    config = uvicorn.Config(
        app,
        uds=path,
        log_level="warning",
        lifespan="off",
        server_header=False,
        date_header=False,
    )
    server = await _start(config)
    os.chmod(path, 0o666)
    return server


async def serve_asgi_on_tcp(app, host: str, port: int) -> uvicorn.Server:
    """Start serving `app` on a TCP port in the running loop.

    Args:
        app (`ASGIApp`):
            The application.
        host (`str`):
            Address to bind.
        port (`int`):
            Port to bind; `0` picks a free one (see `server.servers[0].sockets`).

    Returns:
        `uvicorn.Server`: The started server. Stop it with [`stop_server`].
    """
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        lifespan="off",
        server_header=False,
        date_header=False,
    )
    return await _start(config)


async def stop_server(server: uvicorn.Server) -> None:
    """Ask a server started here to exit and wait until it has."""
    server.should_exit = True
    task = getattr(server, "serve_task", None)
    if task is not None:
        await task
