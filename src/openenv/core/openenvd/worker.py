# SPDX-License-Identifier: BSD-3-Clause
"""Private environment worker, reachable only through an inherited socket.

The supervisor owns the socket's containing directory (0700). No TCP listener
or abstract Unix socket exposes lifecycle controls inside the workload namespace.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import socket
import sys
from contextlib import AsyncExitStack

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from openenv.core.env_server.http_server import _make_json_serializable
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    ListToolsAction,
)
from openenv.core.env_server.serialization import (
    deserialize_action,
    serialize_observation,
)

from .harness import HarnessEventSink
from .mcp import http_rpc, mcp_handler, socket_rpc
from .policy import SurfacePolicy


async def serve(fd: int, factory_name: str, action_name: str) -> None:
    def resolve(name):
        module, attribute = name.split(":", 1)
        value = importlib.import_module(module)
        for component in attribute.split("."):
            value = getattr(value, component)
        return value

    event_fd = os.environ.get("OPENENVD_EVENT_FD")
    if event_fd is not None:
        os.set_inheritable(int(event_fd), False)
    env = resolve(factory_name)()
    action_cls = resolve(action_name)
    lock = asyncio.Lock()

    async def invoke(method, *args, **kwargs):
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        return await asyncio.to_thread(method, *args, **kwargs)

    async def dispatch(request):
        operation = request["operation"]
        data = request.get("data", {})
        if operation == "ready":
            return {"ready": True}
        if operation == "reset":
            return serialize_observation(await env.reset_async(**data))
        if operation == "step":
            return serialize_observation(
                await env.step_async(deserialize_action(data, action_cls))
            )
        if operation == "state":
            return _make_json_serializable(env.state)
        if operation == "mcp":
            rpc = JsonRpcRequest.model_validate(data)
            if rpc.method == "tools/list":
                client = getattr(env, "mcp_client", None)
                if client:
                    result = {
                        "tools": [
                            _make_json_serializable(t)
                            for t in await client.list_tools()
                        ]
                    }
                else:
                    observation = await env.step_async(ListToolsAction())
                    result = {"tools": [t.model_dump() for t in observation.tools]}
            elif rpc.method == "tools/call":
                client = getattr(env, "mcp_client", None)
                if client:
                    result = _make_json_serializable(
                        await client.call_tool(
                            name=rpc.params["name"],
                            arguments=rpc.params.get("arguments", {}),
                        )
                    )
                else:
                    result = _make_json_serializable(
                        await env.step_async(
                            CallToolAction(
                                tool_name=rpc.params["name"],
                                arguments=rpc.params.get("arguments", {}),
                            )
                        )
                    )
            else:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.METHOD_NOT_FOUND,
                    f"Method not found: {rpc.method}",
                    request_id=rpc.id,
                ).model_dump()
            return JsonRpcResponse.success(
                result=result, request_id=rpc.id
            ).model_dump()
        raise ValueError("unknown worker operation")

    def local_agent_app(policy):
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        sink = HarnessEventSink() if event_fd is not None else None

        async def agent_dispatch(data):
            async with lock:
                response = await dispatch({"operation": "mcp", "data": data})
            if sink and data["method"] == "tools/call":
                sink(
                    {
                        "type": "tool_call",
                        "payload": {"request": data, "response": response},
                    }
                )
            return response

        rpc = mcp_handler(policy, agent_dispatch)

        @app.post("/mcp")
        async def http_mcp(request: Request):
            return await http_rpc(request, rpc)

        @app.websocket("/mcp")
        async def websocket_mcp(websocket: WebSocket):
            await socket_rpc(websocket, rpc)

        return app

    async def handle(reader, writer):
        try:
            while line := await reader.readline():
                try:
                    async with lock:
                        result = await dispatch(json.loads(line))
                    response = {"result": result}
                except Exception:
                    response = {"error": "environment operation failed"}
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    sock = socket.socket(fileno=fd)
    sock.set_inheritable(False)
    async with AsyncExitStack() as stack:
        session = getattr(env, "mcp_session", None)
        if session:
            await stack.enter_async_context(session())
        server = await asyncio.start_unix_server(
            handle, sock=sock, limit=16 * 1024 * 1024
        )
        async with server:
            try:
                policy_json = os.environ.get("OPENENVD_AGENT_POLICY")
                if policy_json:
                    policy = SurfacePolicy.model_validate_json(policy_json)
                    local_server = uvicorn.Server(
                        uvicorn.Config(
                            local_agent_app(policy),
                            host="127.0.0.1",
                            port=int(os.environ.get("OPENENVD_AGENT_PORT", "8000")),
                            log_level="warning",
                            access_log=False,
                        )
                    )
                    tasks = [
                        asyncio.create_task(server.serve_forever()),
                        asyncio.create_task(local_server.serve()),
                    ]
                    try:
                        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    await server.serve_forever()
            finally:
                await invoke(env.close)


if __name__ == "__main__":
    asyncio.run(serve(int(sys.argv[1]), sys.argv[2], sys.argv[3]))
