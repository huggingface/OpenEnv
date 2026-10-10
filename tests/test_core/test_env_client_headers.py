# SPDX-License-Identifier: BSD-3-Clause

"""Clients send `headers` on every connection, so they can reach a private Space."""

import socket
import threading
import time

import pytest
import uvicorn
from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient

echo_app = pytest.importorskip("echo_env.server.app").app

HEADERS = {"Authorization": "Bearer secret"}
AUTH_HEADER = (b"authorization", b"Bearer secret")


def require_token(app):
    """Reject HTTP requests and WebSocket upgrades without the token, like a private Space."""

    async def guarded(scope, receive, send):
        if scope["type"] == "lifespan" or AUTH_HEADER in scope["headers"]:
            await app(scope, receive, send)
        elif scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b""})

    return guarded


@pytest.fixture
def server_url():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(require_token(echo_app), log_level="warning")
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def test_websocket_without_headers_is_rejected(server_url):
    with pytest.raises(ConnectionError):
        async with GenericEnvClient(base_url=server_url) as env:
            await env.reset()


async def test_generic_client_sends_headers(server_url):
    async with GenericEnvClient(base_url=server_url, headers=HEADERS) as env:
        await env.reset()
        child = await env.new_session()
        await child.reset()


def test_sync_client_sends_headers(server_url):
    with GenericEnvClient(base_url=server_url, headers=HEADERS).sync() as env:
        env.reset()


async def test_mcp_tool_client_sends_headers(server_url):
    async with MCPToolClient(base_url=server_url, headers=HEADERS) as env:
        await env.reset()
        assert await env.list_tools()
