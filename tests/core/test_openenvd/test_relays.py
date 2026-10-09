# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import shutil
import stat
import tempfile

import httpx
import pytest
from openenv.core.openenvd.relays import (
    env_relay_app,
    extract_text,
    model_proxy_app,
    REQUEST_ID_HEADER,
    serve_asgi_on_tcp,
    serve_asgi_on_unix,
    service_relay_app,
    stop_server,
)
from openenv.core.openenvd.trace import TraceRecorder
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket
from websockets.asyncio.client import unix_connect
from websockets.exceptions import ConnectionClosed

KEY = b"k" * 32
REAL_KEY = "sk-real-secret-0123456789"


@pytest.fixture
def sockdir():
    # AF_UNIX paths are limited to ~104 bytes; pytest's tmp_path is too long on macOS.
    d = tempfile.mkdtemp(prefix="oe", dir="/tmp")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def recorder(tmp_path):
    rec = TraceRecorder(tmp_path / "trace.jsonl", KEY, append_only=False)
    yield rec
    rec.close()


def kinds(recorder):
    return [r.kind for r in recorder._records]


def records(recorder, kind):
    return [r.data for r in recorder._records if r.kind == kind]


# --- a stand-in env server ---------------------------------------------------

seen: dict = {}


async def hello(request: Request):
    seen["authorization"] = request.headers.get("authorization")
    seen["query"] = request.url.query
    response = JSONResponse({"hello": "world", "q": request.url.query})
    response.raw_headers.append((b"set-cookie", b"a=1"))
    response.raw_headers.append((b"set-cookie", b"b=2"))
    response.raw_headers.append((b"x-custom", b"yes"))
    return response


async def mcp(request: Request):
    call = await request.json()
    return JSONResponse(
        {"jsonrpc": "2.0", "id": call["id"], "result": {"tools": [{"name": "echo"}]}}
    )


async def missing(request: Request):
    return JSONResponse({"detail": "Not Found"}, status_code=404)


async def stream(request: Request):
    async def gen():
        for i in range(3):
            yield f"chunk-{i};".encode()

    return StreamingResponse(gen(), media_type="text/plain")


async def big(request: Request):
    return Response(b"x" * (70 * 1024), media_type="text/plain")


async def ws_echo(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        if message.get("bytes") is not None:
            await websocket.send_bytes(message["bytes"][::-1])
            continue
        frame = json.loads(message["text"])
        if frame.get("type") == "bye":
            await websocket.close(code=4001, reason="bye")
            return
        await websocket.send_text(
            json.dumps(
                {"type": "observation", "data": {"done": frame.get("type") == "step"}}
            )
        )


upstream_app = Starlette(
    routes=[
        Route("/hello", hello, methods=["GET"]),
        Route("/mcp", mcp, methods=["POST"]),
        Route("/missing", missing),
        Route("/stream", stream),
        Route("/big", big),
        WebSocketRoute("/ws", ws_echo),
    ]
)


@pytest.fixture
async def upstream(sockdir):
    path = os.path.join(sockdir, "env.sock")
    server = await serve_asgi_on_unix(upstream_app, path)
    yield path
    await stop_server(server)


@pytest.fixture
async def relay(sockdir, upstream, recorder):
    frames = []

    async def on_ws_out(frame):
        frames.append(frame)

    app = env_relay_app(upstream, recorder, on_ws_out=on_ws_out)
    path = os.path.join(sockdir, "relay.sock")
    server = await serve_asgi_on_unix(app, path)
    yield path, frames
    await stop_server(server)
    await app.state.relay.aclose()


def uds_client(path):
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=path), base_url="http://unit"
    )


def visible_headers(response):
    return [
        (k, v)
        for k, v in response.headers.multi_items()
        if k not in ("connection", "transfer-encoding")
    ]


# --- env relay ----------------------------------------------------------------


async def test_http_relay_is_byte_identical_and_drops_authorization(upstream, relay):
    relay_path, _ = relay
    async with uds_client(upstream) as direct, uds_client(relay_path) as via:
        a = await direct.get("/hello?x=1%2F2")
        b = await via.get("/hello?x=1%2F2", headers={"authorization": "Bearer t"})
    assert seen["authorization"] is None
    assert seen["query"] == "x=1%2F2"
    assert (a.status_code, a.content) == (b.status_code, b.content)
    assert visible_headers(a) == visible_headers(b)
    assert b.headers.get_list("set-cookie") == ["a=1", "b=2"]


async def test_streamed_and_error_responses_pass_through(relay, recorder):
    relay_path, _ = relay
    async with uds_client(relay_path) as via:
        streamed = await via.get("/stream")
        missing_ = await via.get("/missing")
    assert streamed.content == b"chunk-0;chunk-1;chunk-2;"
    assert missing_.status_code == 404
    responses = records(recorder, "env.response")
    assert responses[0]["body"] == "chunk-0;chunk-1;chunk-2;"
    assert responses[1]["status"] == 404
    assert responses[1]["body"] == {"detail": "Not Found"}
    assert records(recorder, "env.request")[0]["path"] == "/stream"


async def test_large_bodies_are_truncated_in_the_trace_only(relay, recorder):
    relay_path, _ = relay
    async with uds_client(relay_path) as via:
        response = await via.get("/big")
    assert len(response.content) == 70 * 1024
    record = records(recorder, "env.response")[0]
    assert record["truncated"] is True
    assert len(record["body"]) == 64 * 1024


async def test_mcp_calls_are_recorded_as_mcp(relay, recorder):
    relay_path, _ = relay
    call = {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {}}
    async with uds_client(relay_path) as via:
        response = await via.post("/mcp", json=call)
    assert response.json()["result"]["tools"] == [{"name": "echo"}]
    assert records(recorder, "mcp.call") == [
        {"method": "tools/list", "params": {}, "id": 7}
    ]
    result = records(recorder, "mcp.result")[0]
    assert result["id"] == 7 and result["status"] == 200
    assert result["result"] == {"tools": [{"name": "echo"}]}
    assert "env.request" not in kinds(recorder)


async def test_websocket_relay_round_trips_and_records(relay, recorder):
    relay_path, frames = relay
    async with unix_connect(relay_path, uri="ws://unit/ws") as ws:
        await ws.send(json.dumps({"type": "reset"}))
        assert json.loads(await ws.recv()) == {
            "type": "observation",
            "data": {"done": False},
        }
        await ws.send(json.dumps({"type": "step", "data": {}}))
        assert json.loads(await ws.recv())["data"]["done"] is True
        await ws.send(b"\x01\x02")
        assert await ws.recv() == b"\x02\x01"
        await ws.send(json.dumps({"type": "bye"}))
        with pytest.raises(ConnectionClosed):
            await ws.recv()
    assert ws.close_code == 4001
    assert [f["data"]["done"] for f in frames] == [False, True]
    ins = records(recorder, "ws.in")
    assert ins[0] == {"path": "/ws", "json": {"type": "reset"}}
    assert ins[2] == {"path": "/ws", "bytes": 2}
    assert records(recorder, "ws.out")[1]["json"]["data"]["done"] is True


async def test_sealed_trace_stops_calls_before_they_reach_the_env(relay, recorder):
    relay_path, _ = relay
    recorder.seal("done")
    seen.pop("authorization", None)
    async with uds_client(relay_path) as via:
        response = await via.get("/hello")
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "episode_unavailable"}}
    assert "authorization" not in seen


async def test_unreachable_env_is_a_502_without_details(sockdir, recorder):
    app = env_relay_app(os.path.join(sockdir, "nothing.sock"), recorder)
    path = os.path.join(sockdir, "relay.sock")
    server = await serve_asgi_on_unix(app, path)
    try:
        async with uds_client(path) as via:
            response = await via.get("/health")
    finally:
        await stop_server(server)
    assert response.status_code == 502
    assert response.json() == {"error": {"code": "upstream_unavailable"}}
    assert "nothing.sock" not in response.text


async def test_unix_listener_is_world_connectable_and_replaces_stale_socket(sockdir):
    path = os.path.join(sockdir, "s.sock")
    open(path, "w").close()
    server = await serve_asgi_on_unix(upstream_app, path)
    try:
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o666
        assert stat.S_ISSOCK(os.stat(path).st_mode)
    finally:
        await stop_server(server)


# --- model proxy ---------------------------------------------------------------

provider_seen: list = []

ANTHROPIC_JSON = {
    "type": "message",
    "content": [
        {"type": "thinking", "thinking": "let me think"},
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": " there"},
    ],
}
ANTHROPIC_SSE = [
    {"type": "message_start", "message": {}},
    {
        "type": "content_block_delta",
        "delta": {"type": "thinking_delta", "thinking": "hm"},
    },
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "!"}},
    {"type": "message_stop"},
]
OPENAI_SSE = [
    {"choices": [{"delta": {"reasoning_content": "why"}}]},
    {"choices": [{"delta": {"content": "Yo"}}]},
    {"choices": [{"delta": {"content": "!"}}]},
]


def sse(events, done=False):
    body = "".join(f"event: x\ndata: {json.dumps(e)}\n\n" for e in events)
    return body + ("data: [DONE]\n\n" if done else "")


async def provider(request: Request):
    provider_seen.append(dict(request.headers))
    body = await request.json()
    if request.url.path == "/v1/messages":
        if body.get("stream"):

            async def gen():
                for event in ANTHROPIC_SSE:
                    yield f"event: x\ndata: {json.dumps(event)}\n\n".encode()

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(ANTHROPIC_JSON)
    if body.get("stream"):
        return Response(sse(OPENAI_SSE, done=True), media_type="text/event-stream")
    return JSONResponse(
        {"choices": [{"message": {"content": "Sure", "reasoning_content": "r"}}]}
    )


@pytest.fixture
async def provider_url():
    app = Starlette(routes=[Route("/{p:path}", provider, methods=["POST"])])
    server = await serve_asgi_on_tcp(app, "127.0.0.1", 0)
    port = server.servers[0].sockets[0].getsockname()[1]
    provider_seen.clear()
    yield f"http://127.0.0.1:{port}"
    await stop_server(server)


async def proxy(sockdir, base, recorder, provider_name):
    app = model_proxy_app(base, REAL_KEY, recorder, provider=provider_name)
    path = os.path.join(sockdir, f"{provider_name}.sock")
    return app, path, await serve_asgi_on_unix(app, path)


@pytest.mark.parametrize(
    "headers",
    [{}, {"x-api-key": "wrong"}, {"authorization": f"Bearer {REAL_KEY}x"}],
)
async def test_model_proxy_rejects_anything_but_the_placeholder(
    sockdir, provider_url, recorder, headers
):
    app, path, server = await proxy(sockdir, provider_url, recorder, "anthropic")
    try:
        async with uds_client(path) as c:
            response = await c.post("/v1/messages", json={}, headers=headers)
    finally:
        await stop_server(server)
        await app.state.aclose()
    assert response.status_code == 401
    assert provider_seen == []
    assert kinds(recorder) == []


async def test_model_proxy_anthropic_injects_key_and_records_text(
    sockdir, provider_url, recorder, tmp_path
):
    app, path, server = await proxy(sockdir, provider_url, recorder, "anthropic")
    try:
        async with uds_client(path) as c:
            response = await c.post(
                "/v1/messages",
                json={"model": "m", "messages": []},
                headers={"x-api-key": "openenvd-placeholder"},
            )
    finally:
        await stop_server(server)
        await app.state.aclose()
    assert response.json() == ANTHROPIC_JSON
    assert provider_seen[0]["x-api-key"] == REAL_KEY
    assert "authorization" not in provider_seen[0]
    request_id = response.headers[REQUEST_ID_HEADER]
    req = records(recorder, "model.request")[0]
    resp = records(recorder, "model.response")[0]
    assert req["request_id"] == resp["request_id"] == request_id
    assert req["body"] == {"model": "m", "messages": []}
    assert resp["status"] == 200 and resp["stream"] is False
    assert (resp["text"], resp["thinking"]) == ("Hello there", "let me think")
    assert json.loads(resp["raw"]) == ANTHROPIC_JSON
    assert REAL_KEY not in (tmp_path / "trace.jsonl").read_text()


async def test_model_proxy_streams_anthropic_sse(sockdir, provider_url, recorder):
    app, path, server = await proxy(sockdir, provider_url, recorder, "anthropic")
    try:
        async with uds_client(path) as c:
            response = await c.post(
                "/v1/messages",
                json={"stream": True},
                headers={"authorization": "Bearer openenvd-placeholder"},
            )
    finally:
        await stop_server(server)
        await app.state.aclose()
    assert response.text == sse(ANTHROPIC_SSE)
    resp = records(recorder, "model.response")[0]
    assert resp["stream"] is True
    assert (resp["text"], resp["thinking"]) == ("Hi!", "hm")
    assert resp["raw"] == "\n".join(json.dumps(e) for e in ANTHROPIC_SSE)


async def test_model_proxy_openai_uses_bearer_and_extracts(
    sockdir, provider_url, recorder
):
    app, path, server = await proxy(sockdir, provider_url, recorder, "openai")
    try:
        async with uds_client(path) as c:
            plain = await c.post(
                "/v1/chat/completions",
                json={},
                headers={"authorization": "Bearer openenvd-placeholder"},
            )
            streamed = await c.post(
                "/v1/chat/completions",
                json={"stream": True},
                headers={"authorization": "Bearer openenvd-placeholder"},
            )
    finally:
        await stop_server(server)
        await app.state.aclose()
    assert plain.status_code == streamed.status_code == 200
    assert provider_seen[0]["authorization"] == f"Bearer {REAL_KEY}"
    first, second = records(recorder, "model.response")
    assert (first["text"], first["thinking"]) == ("Sure", "r")
    assert (second["text"], second["thinking"]) == ("Yo!", "why")


@pytest.mark.parametrize(
    "provider_name,body,stream,expected",
    [
        ("anthropic", "not json", False, (None, None)),
        (
            "anthropic",
            json.dumps({"content": [{"type": "text", "text": "a"}]}),
            False,
            ("a", None),
        ),
        ("openai", json.dumps({"choices": []}), False, (None, None)),
        (
            "openai",
            json.dumps(
                {"choices": [{"message": {"content": [{"type": "text", "text": "p"}]}}]}
            ),
            False,
            ("p", None),
        ),
        ("openai", "[DONE]", True, (None, None)),
    ],
)
def test_extract_text_is_best_effort(provider_name, body, stream, expected):
    assert extract_text(provider_name, body, stream=stream) == expected


def test_model_proxy_rejects_unknown_provider(recorder):
    with pytest.raises(ValueError):
        model_proxy_app("http://x", "k", recorder, provider="gemini")


# --- service relay -------------------------------------------------------------


async def svc(request: Request):
    body = await request.body()
    response = JSONResponse({"echo": body.decode()[:10], "len": len(body)})
    response.raw_headers += [
        (b"server", b"nginx/1.2"),
        (b"via", b"1.1 internal-proxy"),
        (b"x-forwarded-for", b"10.0.0.1"),
        (b"x-powered-by", b"php"),
        (b"x-kept", b"1"),
    ]
    return response


async def test_service_relay_records_and_hides_infrastructure_headers(
    sockdir, recorder
):
    upstream_path = os.path.join(sockdir, "svc.sock")
    up = await serve_asgi_on_unix(
        Starlette(routes=[Route("/{p:path}", svc, methods=["POST"])]), upstream_path
    )
    app = service_relay_app("db", upstream_path, recorder)
    path = os.path.join(sockdir, "svc-relay.sock")
    server = await serve_asgi_on_unix(app, path)
    try:
        async with uds_client(path) as c:
            response = await c.post("/q", content=b"y" * (65 * 1024))
    finally:
        await stop_server(server)
        await stop_server(up)
        await app.state.aclose()
    assert response.json() == {"echo": "y" * 10, "len": 65 * 1024}
    for name in ("server", "via", "x-forwarded-for", "x-powered-by"):
        assert name not in response.headers
    assert response.headers["x-kept"] == "1"
    req = records(recorder, "service.request")[0]
    assert (req["service"], req["method"], req["path"]) == ("db", "POST", "/q")
    assert req["truncated"] is True
    resp = records(recorder, "service.response")[0]
    assert resp["status"] == 200 and resp["truncated"] is False
    assert resp["body"]["len"] == 65 * 1024
    assert {r.source for r in recorder._records} == {"service_relay"}
