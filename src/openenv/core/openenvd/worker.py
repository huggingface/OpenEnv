# SPDX-License-Identifier: BSD-3-Clause
"""Environment worker controlled by the daemon's protected SSH stdio stream.

The bootstrap saves the original stdin/stdout as private noninheritable
descriptors before importing this module. No local listener exposes lifecycle
controls. The optional loopback MCP listener contains agent tools only.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
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
from .policy import Principal, SurfacePolicy


async def serve(control_read: int, control_write: int) -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=16 * 1024 * 1024)
    input_transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(control_read, "rb")
    )
    output_transport, output_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, os.fdopen(control_write, "wb")
    )
    writer = asyncio.StreamWriter(output_transport, output_protocol, None, loop)
    output_lock = asyncio.Lock()

    async def send(frame):
        async with output_lock:
            writer.write(json.dumps(frame).encode() + b"\n")
            await writer.drain()

    event_read, event_write = os.pipe()
    os.set_inheritable(event_read, False)
    os.set_inheritable(event_write, False)
    os.set_blocking(event_write, False)
    os.environ["OPENENVD_EVENT_FD"] = str(event_write)
    events = asyncio.StreamReader(limit=65536)
    event_transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(events), os.fdopen(event_read, "rb")
    )
    ready = asyncio.Event()

    async def forward_events():
        while line := await events.readline():
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("invalid harness event")
            await ready.wait()
            await send({"event": event})

    tasks = [asyncio.create_task(forward_events())]
    env = None
    try:
        config = json.loads(await reader.readline())
        if not isinstance(config, dict) or set(config) != {
            "factory",
            "action_class",
            "agent_policy",
        }:
            raise ValueError("invalid worker configuration")
        policy = (
            SurfacePolicy.model_validate(config["agent_policy"])
            if config["agent_policy"] is not None
            else None
        )
        if policy is not None and policy.principal != Principal.AGENT:
            raise ValueError("local surface requires an agent policy")
        env, action_cls = create_environment(config["factory"], config["action_class"])
        await run_environment(env, action_cls, policy, reader, send, ready, tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if env is not None:
            await invoke(env.close)
        os.environ.pop("OPENENVD_EVENT_FD", None)
        os.close(event_write)
        event_transport.close()
        input_transport.close()
        writer.close()


def create_environment(factory_name, action_name):
    def resolve(name):
        module, attribute = name.split(":", 1)
        value = importlib.import_module(module)
        for component in attribute.split("."):
            value = getattr(value, component)
        return value

    return resolve(factory_name)(), resolve(action_name)


async def invoke(method, *args, **kwargs):
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    return await asyncio.to_thread(method, *args, **kwargs)


async def run_environment(env, action_cls, policy, reader, send, ready, tasks):
    lock = asyncio.Lock()

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
        sink = HarnessEventSink()

        async def agent_dispatch(data):
            async with lock:
                response = await dispatch({"operation": "mcp", "data": data})
            if data["method"] == "tools/call":
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

    async def handle():
        while line := await reader.readline():
            try:
                async with lock:
                    result = await dispatch(json.loads(line))
                response = {"result": result}
            except Exception:
                response = {"error": "environment operation failed"}
            await send(response)

    async with AsyncExitStack() as stack:
        session = getattr(env, "mcp_session", None)
        if session:
            await stack.enter_async_context(session())
        try:
            if policy is not None:
                local_server = uvicorn.Server(
                    uvicorn.Config(
                        local_agent_app(policy),
                        host="127.0.0.1",
                        port=int(os.environ.get("OPENENVD_AGENT_PORT", "8000")),
                        log_level="warning",
                        access_log=False,
                    )
                )
                server_task = asyncio.create_task(local_server.serve())
                tasks.append(server_task)
                while not local_server.started:
                    if server_task.done():
                        await server_task
                        raise RuntimeError("agent listener did not start")
                    await asyncio.sleep(0.01)
            await send({"result": {"ready": True}})
            ready.set()
            tasks.append(asyncio.create_task(handle()))
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
            # Stop dispatch before closing the environment's shared MCP session.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main(control_read: int, control_write: int) -> None:
    """Run only after the standalone bootstrap has protected the control channel."""
    asyncio.run(serve(control_read, control_write))
