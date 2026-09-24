# SPDX-License-Identifier: BSD-3-Clause
"""Exercise the real worker transport without claiming OS isolation on macOS."""

import asyncio
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_real_worker_shares_mcp_and_orchestration_state(tmp_path):
    socket_dir = tempfile.TemporaryDirectory(prefix="oed-", dir="/tmp")
    path = Path(socket_dir.name) / "w.sock"
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(path))
    sock.listen()
    root = Path(__file__).resolve().parents[2]
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "openenv.core.openenvd.worker",
        str(sock.fileno()),
        "echo_env.server.echo_environment:EchoEnvironment",
        "openenv.core.env_server.mcp_types:CallToolAction",
        pass_fds=(sock.fileno(),),
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src") + os.pathsep + str(root / "envs"),
        },
    )
    sock.close()

    async def request(operation, data):
        reader, writer = await asyncio.open_unix_connection(str(path))
        try:
            writer.write(
                json.dumps({"operation": operation, "data": data}).encode() + b"\n"
            )
            await writer.drain()
            response = json.loads(await asyncio.wait_for(reader.readline(), 15))
            assert "error" not in response, response
            return response["result"]
        finally:
            writer.close()
            await writer.wait_closed()

    try:
        reset = await request("reset", {"seed": 7})
        assert reset["done"] is False
        tools = await request(
            "mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        assert "echo_message" in [tool["name"] for tool in tools["result"]["tools"]]
        result = await request(
            "mcp",
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "echo_message", "arguments": {"message": "hello"}},
            },
        )
        assert "hello" in json.dumps(result)
        await request(
            "step",
            {
                "type": "call_tool",
                "tool_name": "echo_message",
                "arguments": {"message": "shared episode"},
            },
        )
        state = await request("state", {})
        assert state["step_count"] > 0
        unsupported = await request(
            "mcp", {"jsonrpc": "2.0", "id": 3, "method": "reset", "params": {}}
        )
        assert unsupported["error"]["code"] == -32601
    finally:
        proc.terminate()
        await proc.wait()
        socket_dir.cleanup()


@pytest.mark.asyncio
async def test_local_agent_listener_has_no_privileged_routes():
    import httpx

    socket_dir = tempfile.TemporaryDirectory(prefix="oed-", dir="/tmp")
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(Path(socket_dir.name) / "w.sock"))
    sock.listen()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    root = Path(__file__).resolve().parents[2]
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "openenv.core.openenvd.worker",
        str(sock.fileno()),
        "echo_env.server.echo_environment:EchoEnvironment",
        "openenv.core.env_server.mcp_types:CallToolAction",
        pass_fds=(sock.fileno(),),
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src") + os.pathsep + str(root / "envs"),
            "OPENENVD_AGENT_POLICY": json.dumps(
                {"principal": "agent", "tools": ["echo_message"]}
            ),
            "OPENENVD_AGENT_PORT": str(port),
        },
    )
    sock.close()
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            for _ in range(100):
                try:
                    response = await client.post(
                        "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
                    )
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(0.05)
            else:
                pytest.fail("agent listener did not start")
            assert [tool["name"] for tool in response.json()["result"]["tools"]] == [
                "echo_message"
            ]
            for path in ("/reset", "/state", "/mcp/grader", "/observe", "/ws"):
                assert (await client.post(path, json={})).status_code == 404
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "reset"},
                },
            )
            assert response.json()["error"]["code"] == -32602
    finally:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), 5)
        socket_dir.cleanup()
