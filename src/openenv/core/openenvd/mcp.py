# SPDX-License-Identifier: BSD-3-Clause
"""Shared MCP validation and transport handling; callers own authorization."""

import json
from uuid import uuid4

from fastapi import WebSocketDisconnect
from openenv.core.env_server.mcp_types import (
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
)


def error(code, message, request_id=None):
    return JsonRpcResponse.error_response(
        code, message, request_id=request_id
    ).model_dump()


def mcp_handler(policy, dispatch):
    """Bind one surface's policy and session handles to its operation handler.

    Sessions are transport handles for the same episode, never lifecycle controls.
    Each surface gets its own registry, including local and remote agent surfaces.
    """
    sessions = set()

    async def rpc(data):
        request_id = data.get("id") if isinstance(data, dict) else None
        if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
            request_id = None
        try:
            request = JsonRpcRequest.model_validate(data)
            params = request.params
            if not isinstance(params, dict):
                raise ValueError("invalid parameters")
            if request.method == "openenv/session/create":
                if len(sessions) >= 1024:
                    raise ValueError("session capacity exceeded")
                session_id = str(uuid4())
                sessions.add(session_id)
                return JsonRpcResponse.success(
                    {"session_id": session_id}, request_id=request.id
                ).model_dump()
            session_id = params.get("session_id")
            if session_id is not None and (
                not isinstance(session_id, str) or session_id not in sessions
            ):
                raise ValueError("unknown session")
            if request.method == "openenv/session/close":
                if session_id is None:
                    raise ValueError("session_id is required")
                sessions.remove(session_id)
                return JsonRpcResponse.success(
                    {"session_id": session_id, "closed": True}, request_id=request.id
                ).model_dump()
            if request.method == "tools/call":
                name, arguments = params.get("name", ""), params.get("arguments", {})
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise ValueError("invalid arguments")
                if not policy.permits_tool(name):
                    return error(
                        JsonRpcErrorCode.INVALID_PARAMS, "Tool not found", request_id
                    )
            elif request.method != "tools/list":
                return error(
                    JsonRpcErrorCode.METHOD_NOT_FOUND,
                    f"Method not found: {request.method}",
                    request_id,
                )
            response = await dispatch(data)
            if request.method == "tools/list" and not response.get("error"):
                response["result"]["tools"] = [
                    tool
                    for tool in response["result"]["tools"]
                    if policy.permits_tool(tool["name"])
                ]
            return response
        except (ValueError, TypeError):
            return error(
                JsonRpcErrorCode.INVALID_PARAMS, "Invalid parameters", request_id
            )
        except PermissionError:
            return error(
                JsonRpcErrorCode.INVALID_PARAMS, "Operation not permitted", request_id
            )
        except Exception:
            return error(
                JsonRpcErrorCode.INTERNAL_ERROR,
                "Environment operation failed",
                request_id,
            )

    return rpc


async def http_rpc(request, rpc):
    try:
        data = await request.json()
    except ValueError:
        return error(JsonRpcErrorCode.PARSE_ERROR, "Parse error")
    return await rpc(data)


async def socket_rpc(websocket, rpc):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_json()
            except ValueError:
                await websocket.send_json(
                    error(JsonRpcErrorCode.PARSE_ERROR, "Parse error")
                )
                continue
            await websocket.send_text(json.dumps(await rpc(data)))
    except WebSocketDisconnect:
        pass
