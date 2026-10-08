# SPDX-License-Identifier: BSD-3-Clause
"""Real protected stdio transport; portable tests do not claim OS isolation."""

import asyncio
import json
import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = ROOT / "src/openenv/core/openenvd/_worker_bootstrap.py"
CONFIG = {
    "factory": "echo_env.server.echo_environment:EchoEnvironment",
    "action_class": "openenv.core.env_server.mcp_types:CallToolAction",
    "agent_policy": None,
}


class Worker:
    def __init__(self, process):
        self.process = process
        self.events = []

    async def send(self, frame):
        self.process.stdin.write(json.dumps(frame).encode() + b"\n")
        await self.process.stdin.drain()

    async def response(self):
        while True:
            line = await asyncio.wait_for(self.process.stdout.readline(), 15)
            assert line, "worker disconnected"
            response = json.loads(line)
            if "event" in response:
                self.events.append(response["event"])
                continue
            assert "error" not in response, response
            return response["result"]

    async def request(self, operation, data):
        await self.send({"operation": operation, "data": data})
        return await self.response()


@asynccontextmanager
async def running_worker(worker_python, *, config=None, extra_env=None):
    python, _ = worker_python
    process = await asyncio.create_subprocess_exec(
        str(python),
        "-I",
        "-S",
        "-c",
        BOOTSTRAP.read_text(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **(extra_env or {})},
    )
    worker = Worker(process)
    try:
        await worker.send(config or CONFIG)
        # Initialization events must follow this frame so the daemon can
        # distinguish startup success from unsolicited workload observations.
        line = await asyncio.wait_for(process.stdout.readline(), 15)
        if not line:
            pytest.fail((await process.stderr.read()).decode())
        assert json.loads(line) == {"result": {"ready": True}}
        yield worker
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_real_worker_shares_mcp_and_orchestration_state(worker_python):
    async with running_worker(worker_python) as worker:
        reset = await worker.request("reset", {"seed": 7})
        assert reset["done"] is False
        tools = await worker.request(
            "mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        assert "echo_message" in [tool["name"] for tool in tools["result"]["tools"]]
        result = await worker.request(
            "mcp",
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "echo_message", "arguments": {"message": "hello"}},
            },
        )
        assert "hello" in json.dumps(result)
        await worker.request(
            "step",
            {
                "type": "call_tool",
                "tool_name": "echo_message",
                "arguments": {"message": "shared episode"},
            },
        )
        state = await worker.request("state", {})
        assert state["step_count"] > 0
        unsupported = await worker.request(
            "mcp", {"jsonrpc": "2.0", "id": 3, "method": "reset", "params": {}}
        )
        assert unsupported["error"]["code"] == -32601
    assert worker.process.returncode == 0


@pytest.mark.asyncio
async def test_local_agent_listener_has_no_privileged_routes(worker_python):
    import httpx

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    config = {
        **CONFIG,
        "agent_policy": {"principal": "agent", "tools": ["echo_message"]},
    }
    async with running_worker(
        worker_python, config=config, extra_env={"OPENENVD_AGENT_PORT": str(port)}
    ) as worker:
        await worker.request("reset", {"episode_id": "local-episode"})
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            response = await client.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
            )
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
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "echo_message",
                        "arguments": {"message": "local harness"},
                    },
                },
            )
            assert "local harness" in response.text
        state = await worker.request("state", {})
        assert state["episode_id"] == "local-episode"
        assert any(event["type"] == "tool_call" for event in worker.events)


@pytest.mark.asyncio
async def test_imports_and_subprocesses_cannot_use_control_stdio(worker_python):
    _, modules = worker_python
    (modules / "noisy_environment.py").write_text(
        """
import os
import subprocess
import sys

from echo_env.server.echo_environment import EchoEnvironment
from openenv.core.openenvd.harness import HarnessEventSink

print('environment import diagnostics', flush=True)
subprocess.run([sys.executable, '-I', '-c', '''
import os, sys
assert sys.stdin.read() == ''
for fd in range(3, 256):
    try:
        os.fstat(fd)
    except OSError:
        continue
    raise AssertionError('inherited descriptor: ' + str(fd))
print('{"result":{"forged":true}}', flush=True)
'''], check=True, close_fds=False)
HarnessEventSink()({'type': 'import_event'})
"""
    )
    async with running_worker(
        worker_python,
        config={**CONFIG, "factory": "noisy_environment:EchoEnvironment"},
    ) as worker:
        assert (await worker.request("reset", {}))["done"] is False
        assert worker.events == [{"type": "import_event"}]
    diagnostics = (await worker.process.stderr.read()).decode()
    assert "environment import diagnostics" in diagnostics
    assert '{"result":{"forged":true}}' in diagnostics
    assert worker.process.returncode == 0
