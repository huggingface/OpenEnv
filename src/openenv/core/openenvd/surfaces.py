# SPDX-License-Identifier: BSD-3-Clause
"""Policy-scoped agent, grader, orchestrator, and observer transports."""

from __future__ import annotations

import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from openenv.core.env_server.mcp_types import JsonRpcResponse

from .mcp import http_rpc, mcp_handler, socket_rpc
from .policy import Principal
from .runtime import Runtime

_GRADER_TOOLS = {
    "grader.read_file": {
        "description": "Read a policy-permitted file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "grader.fs_diff": {
        "description": "Workspace changes since reset",
        "inputSchema": {
            "type": "object",
            "properties": {"since": {"type": "string", "enum": ["reset"]}},
            "additionalProperties": False,
        },
    },
    "grader.run_oracle": {
        "description": "Execute the configured privileged oracle",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "grader.get_full_state": {
        "description": "Inspect environment state",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "grader.get_trajectory": {
        "description": "Read the full episode observation history",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}
_STREAM_KIND = {
    "harness_events": "harness_event",
    "fs_diff": "fs_change",
    "process": "process",
    "network": "network",
    "resource": "resource",
}


def create_surface_app(runtime: Runtime, tokens: dict[Principal, str]) -> FastAPI:
    """Bind each privileged surface to its own operator-provided credential.

    Agent transport has no additional headers or credentials. Credentials for
    grader, observer and orchestrator must be distinct and never reach children.
    """
    policies = runtime.config.surfaces
    tokens = {Principal(key): value for key, value in tokens.items()}
    required = set(policies) - {Principal.AGENT}
    if any(
        not tokens.get(p)
        or not tokens[p].isascii()
        or not tokens[p].isprintable()
        or " " in tokens[p]
        for p in required
    ):
        raise ValueError("each privileged surface requires an ASCII bearer token")
    if len({tokens[p] for p in required}) != len(required):
        raise ValueError("principal tokens must be distinct")

    @asynccontextmanager
    async def lifespan(app):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.runtime = runtime

    def authorized(headers, principal):
        scheme, _, token = headers.get("authorization", "").partition(" ")
        return (
            principal in policies
            and scheme.lower() == "bearer"
            and hmac.compare_digest(token.encode(), tokens.get(principal, "").encode())
        )

    async def dispatch(principal, data):
        policy = policies[principal]
        if data["method"] == "tools/list":
            response = await runtime.request("mcp", data)
            if "error" in response and response["error"] is not None:
                return response
            tools = response["result"]["tools"]
            tools = [tool for tool in tools if not tool["name"].startswith("grader.")]
            if principal == Principal.GRADER:
                tools += [
                    {"name": name, **schema}
                    for name, schema in _GRADER_TOOLS.items()
                    if name != "grader.run_oracle" or policy.allow_privileged_exec
                ]
            response["result"]["tools"] = tools
            return response
        name = data["params"]["name"]
        arguments = data["params"].get("arguments", {})
        if name.startswith("grader."):
            if principal != Principal.GRADER or name not in _GRADER_TOOLS:
                raise PermissionError("unknown privileged tool")
            if name == "grader.read_file":
                if set(arguments) != {"path"} or not isinstance(arguments["path"], str):
                    raise ValueError("path is required")
                result = await asyncio.to_thread(runtime.read_file, arguments["path"])
            elif name == "grader.fs_diff":
                if (
                    set(arguments) - {"since"}
                    or arguments.get("since", "reset") != "reset"
                ):
                    raise ValueError("since must be reset")
                result = [
                    change
                    for change in await asyncio.to_thread(runtime.workspace.diff)
                    if policy.permits_read(Path(change["path"]))
                ]
            else:
                if arguments:
                    raise ValueError("this tool accepts no arguments")
                if name == "grader.run_oracle":
                    result = await runtime.run_oracle()
                elif name == "grader.get_full_state":
                    result = await runtime.request("state")
                else:
                    result = runtime.collector.trajectory()
            return JsonRpcResponse.success(
                result={
                    "content": [{"type": "text", "text": json.dumps(result)}],
                    "isError": False,
                },
                request_id=data.get("id"),
            ).model_dump()
        return await runtime.request("mcp", data)

    handlers = {
        principal: mcp_handler(policy, lambda data, p=principal: dispatch(p, data))
        for principal, policy in policies.items()
    }

    async def authorized_http(request, principal):
        if principal not in policies:
            raise HTTPException(404)
        if principal != Principal.AGENT and not authorized(request.headers, principal):
            raise HTTPException(401, headers={"WWW-Authenticate": "Bearer"})
        return await http_rpc(request, handlers[principal])

    @app.post("/mcp")
    async def agent_http(request: Request):
        return await authorized_http(request, Principal.AGENT)

    @app.post("/mcp/grader")
    async def grader_http(request: Request):
        return await authorized_http(request, Principal.GRADER)

    async def authorized_socket(websocket, principal):
        if principal not in policies or (
            principal != Principal.AGENT
            and not authorized(websocket.headers, principal)
        ):
            await websocket.close(code=1008)
            return
        await socket_rpc(websocket, handlers[principal])

    @app.websocket("/mcp")
    async def agent_socket(websocket: WebSocket):
        await authorized_socket(websocket, Principal.AGENT)

    @app.websocket("/mcp/grader")
    async def grader_socket(websocket: WebSocket):
        await authorized_socket(websocket, Principal.GRADER)

    @app.websocket("/ws")
    async def orchestrator_socket(websocket: WebSocket):
        principal = Principal.ORCHESTRATOR
        if (
            not authorized(websocket.headers, principal)
            or not policies[principal].allow_lifecycle
        ):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                try:
                    message = await websocket.receive_json()
                    operation = message["type"]
                    if operation == "reset":
                        result = await runtime.reset(message.get("data", {}))
                    elif operation in {"step", "state"}:
                        result = await runtime.request(
                            operation, message.get("data", {})
                        )
                    elif operation == "close":
                        await runtime.stop()
                        await websocket.close()
                        return
                    else:
                        raise ValueError("unknown operation")
                    await websocket.send_json(
                        {
                            "type": "state" if operation == "state" else "observation",
                            "data": result,
                        }
                    )
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    OSError,
                ) as error:
                    blocked = "workspace restore refused" in str(
                        error
                    ) or "did not exit" in str(error)
                    await websocket.send_json(
                        {
                            "type": "error",
                            "data": {
                                "message": (
                                    "teardown incomplete: workload cgroup did not empty; "
                                    "workspace restore refused"
                                    if blocked
                                    else "Environment operation failed"
                                ),
                                "code": "execution_error",
                            },
                        }
                    )
        except WebSocketDisconnect:
            pass

    @app.websocket("/observe")
    async def observe(websocket: WebSocket):
        principal = Principal.OBSERVER
        if not authorized(websocket.headers, principal):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        cursor = 0
        collector = runtime.collector
        allowed = {_STREAM_KIND[name] for name in policies[principal].stream}
        try:
            while True:
                if collector is not runtime.collector:
                    collector, cursor = runtime.collector, 0
                collector.changed.clear()
                for event in collector.events[cursor:]:
                    if event.type.value in allowed:
                        await websocket.send_json(event.model_dump(mode="json"))
                    cursor = event.seq + 1
                try:
                    await asyncio.wait_for(collector.changed.wait(), 1)
                except asyncio.TimeoutError:
                    # Receive disconnects even when there are no events.
                    try:
                        message = await asyncio.wait_for(websocket.receive(), 0.01)
                        if message["type"] == "websocket.disconnect":
                            return
                    except asyncio.TimeoutError:
                        pass
        except WebSocketDisconnect:
            pass

    @app.get("/health")
    async def health():
        return {
            "status": "ok"
            if runtime.proc and runtime.proc.returncode is None
            else "unavailable"
        }

    return app
