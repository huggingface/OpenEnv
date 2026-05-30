# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side interception server for trainer-owned generation.

The :class:`InterceptionServer` runs on the trainer node, outside any
sandbox. Each sandbox's agent is pointed at::

    http://<host>:<port>/rollout/<rollout_id>/v1

When the agent makes an LLM call it blocks at this server. The training
loop calls :meth:`~InterceptionServer.register_rollout` to get a queue,
``await queue.get()`` to dequeue the pending request, runs its own vLLM
forward pass, then calls :func:`deliver_response` to unblock the agent.

The caller is responsible for making the server reachable from the sandbox.
For Docker sandboxes on the same machine, ``host.docker.internal:<port>``
works. For remote sandboxes (E2B, HF Sandbox), set up your own tunnel
(ngrok, frp, public IP, VPN) and pass the URL as
``interception_base_url``.

Usage — training loop::

    server = InterceptionServer(port=8765, tool_name_allowlist={"answer"})
    await server.start()

    # Make the server reachable — your responsibility.
    # Docker: base_url = f"http://host.docker.internal:{server.port}"
    # Remote: base_url = your_tunnel_or_public_url

    request_queue = server.register_rollout(rollout_id)
    # Agent runs with OPENAI_BASE_URL = f"{base_url}/rollout/{rollout_id}/v1"

    while True:
        request_id = await asyncio.to_thread(request_queue.get, timeout=...)
        intercept = server.get_intercept(request_id)
        if intercept is None:
            continue
        response = await vllm.generate(intercept["messages"], ...)
        await deliver_response(intercept, response)

    server.unregister_rollout(rollout_id)
    await server.stop()
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import queue as _queue_mod
import re
import secrets
import threading
import time
import uuid
from typing import Any, Awaitable, Callable

from openenv.core.env_server.mcp_types import RESERVED_TOOL_NAMES

from aiohttp import web


_log = logging.getLogger(__name__)

_KEEPALIVE_INTERVAL_S = 3.0
_MAX_REQUEST_BODY = 16 * 1024 * 1024
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class InterceptionServer:
    """Async HTTP server that gates every LLM call from sandboxed agents.

    One shared instance handles all concurrent rollouts. Each rollout is
    identified by a ``rollout_id`` in the URL path.
    """

    def __init__(
        self,
        port: int = 0,
        secret: str | None = None,
        host: str = "127.0.0.1",
        tool_name_allowlist: set[str] | None = None,
    ) -> None:
        self.port = port
        self.host = host
        self.secret = secret or secrets.token_urlsafe(32)
        if not self.secret.strip():
            raise ValueError("InterceptionServer secret must not be blank.")
        normalized_allowlist: set[str] = set()
        for raw_name in tool_name_allowlist or set():
            name = raw_name.strip()
            if not name:
                raise ValueError("tool_name_allowlist must not include blank names")
            if not _TOOL_NAME_RE.fullmatch(name):
                raise ValueError(
                    "tool_name_allowlist entries must match "
                    f"^[A-Za-z0-9_-]{{1,64}}$ (got {raw_name!r})"
                )
            normalized_allowlist.add(name)
        self._tool_name_allowlist = frozenset(normalized_allowlist)
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._lock = asyncio.Lock()
        self._state_lock = threading.RLock()
        self.active_rollouts: dict[str, dict[str, Any]] = {}
        self.intercepts: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        async with self._lock:
            if self._app is not None:
                return
            app = web.Application(client_max_size=_MAX_REQUEST_BODY)
            app.router.add_post(
                "/rollout/{rollout_id}/v1/chat/completions",
                self._handle_chat_completions,
            )
            app.router.add_post(
                "/rollout/{rollout_id}/v1/tools/{tool_name}",
                self._handle_tool_call,
            )
            app.router.add_post(
                "/rollout/{rollout_id}/exit",
                self._handle_exit,
            )
            app.router.add_post(
                "/rollout/{rollout_id}/v1/exit",
                self._handle_exit,
            )
            app.router.add_get("/health", self._handle_health)
            runner = web.AppRunner(app)
            await runner.setup()
            if self.host == "0.0.0.0":
                _log.warning("InterceptionServer exposed on all interfaces (0.0.0.0).")
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            if self.port == 0:
                server = getattr(site, "_server", None)
                sockets = getattr(server, "sockets", None) if server else None
                if sockets:
                    self.port = sockets[0].getsockname()[1]
            if self.port == 0:
                raise RuntimeError("Failed to resolve OS-assigned port")
            self._app = app
            self._runner = runner
            self._site = site
            _log.info("InterceptionServer listening on :%d", self.port)

    async def stop(self) -> None:
        async with self._lock:
            if self._runner is None:
                return
            with self._state_lock:
                intercepts = list(self.intercepts.values())
                self.intercepts.clear()
                self.active_rollouts.clear()
            for intercept in intercepts:
                fut: asyncio.Future | None = intercept.get("response_future")
                if fut and not fut.done():
                    fut.cancel()
                cq: asyncio.Queue | None = intercept.get("chunk_queue")
                if cq is not None:
                    try:
                        cq.put_nowait(None)
                    except asyncio.QueueFull:
                        pass
            try:
                await self._runner.cleanup()
            except RuntimeError:
                pass
            self._runner = None
            self._site = None
            self._app = None

    def register_rollout(
        self,
        rollout_id: str,
        state: dict[str, Any] | None = None,
    ) -> _queue_mod.Queue[str]:
        request_queue: _queue_mod.Queue[str] = _queue_mod.Queue()
        with self._state_lock:
            self.active_rollouts[rollout_id] = {
                "request_id_queue": request_queue,
                "state": state,
                "tool_handlers": {},
                "tool_defs": {},
            }
            active = len(self.active_rollouts)
        _log.info(
            "interception_rollout_registered rollout_id=%s active_rollouts=%d",
            rollout_id,
            active,
        )
        return request_queue

    def unregister_rollout(self, rollout_id: str) -> None:
        with self._state_lock:
            matching_ids = [
                request_id
                for request_id, intercept in self.intercepts.items()
                if intercept.get("rollout_id") == rollout_id
            ]
            matching_intercepts = [self.intercepts[i] for i in matching_ids]
            for request_id in matching_ids:
                del self.intercepts[request_id]
            removed = self.active_rollouts.pop(rollout_id, None) is not None
            active = len(self.active_rollouts)
            pending = len(self.intercepts)

        for intercept in matching_intercepts:
            fut: asyncio.Future | None = intercept.get("response_future")
            if fut and not fut.done():
                fut.cancel()
            cq: asyncio.Queue | None = intercept.get("chunk_queue")
            if cq is not None:
                try:
                    cq.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        _log.info(
            "interception_rollout_unregistered rollout_id=%s removed=%s "
            "active_rollouts=%d pending_intercepts=%d",
            rollout_id,
            removed,
            active,
            pending,
        )

    def get_intercept(self, request_id: str) -> dict[str, Any] | None:
        with self._state_lock:
            return self.intercepts.get(request_id)

    def stats(self) -> dict[str, int]:
        """Return lightweight runtime counters for health/debug views."""
        with self._state_lock:
            return {
                "active_rollouts": len(self.active_rollouts),
                "pending_intercepts": len(self.intercepts),
            }

    def register_tool_handler(
        self,
        rollout_id: str,
        tool_name: str,
        handler: ToolHandler,
        *,
        tool_definition: dict[str, Any] | None = None,
    ) -> None:
        """Register a host-side tool handler for a rollout.

        The handler is called by ``POST /rollout/{rollout_id}/v1/tools/{tool_name}``
        with a JSON payload containing ``arguments``.

        Optionally provide ``tool_definition`` (OpenAI tool schema). Registered
        schemas are injected into intercepted chat-completion requests for the
        rollout when the incoming request does not already include the tool.

        Only tool names explicitly configured in ``tool_name_allowlist`` are
        accepted. Control-plane names (``reset``, ``step``, ``state``,
        ``close``) are always rejected to preserve the dual API boundary.
        """
        normalized_name = self._validate_tool_registration(
            tool_name,
            tool_definition=tool_definition,
        )

        with self._state_lock:
            context = self.active_rollouts.get(rollout_id)
            if context is None:
                raise KeyError(f"rollout not found: {rollout_id}")
            handlers: dict[str, ToolHandler] = context["tool_handlers"]
            handlers[normalized_name] = handler
            if tool_definition is not None:
                tool_defs: dict[str, dict[str, Any]] = context["tool_defs"]
                tool_defs[normalized_name] = tool_definition

    def unregister_tool_handler(self, rollout_id: str, tool_name: str) -> None:
        with self._state_lock:
            context = self.active_rollouts.get(rollout_id)
            if context is None:
                return
            handlers: dict[str, ToolHandler] = context.get("tool_handlers", {})
            handlers.pop(tool_name, None)
            tool_defs: dict[str, dict[str, Any]] = context.get("tool_defs", {})
            tool_defs.pop(tool_name, None)

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> str | None:
        if not isinstance(tool, dict):
            return None
        function = tool.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        return name if isinstance(name, str) and name else None

    def _validate_tool_registration(
        self,
        tool_name: str,
        *,
        tool_definition: dict[str, Any] | None,
    ) -> str:
        normalized = tool_name.strip()
        if not normalized:
            raise ValueError("tool_name must not be blank")
        if not _TOOL_NAME_RE.fullmatch(normalized):
            raise ValueError(
                f"tool_name must match ^[A-Za-z0-9_-]{{1,64}}$ (got {tool_name!r})"
            )
        if normalized.lower() in RESERVED_TOOL_NAMES:
            raise ValueError(
                "Interception tool name is reserved for infrastructure/control "
                f"APIs: {normalized!r}"
            )
        if normalized not in self._tool_name_allowlist:
            raise ValueError(
                "Interception tool name is not in the configured allowlist: "
                f"{normalized!r}"
            )

        if tool_definition is not None:
            definition_name = self._tool_name(tool_definition)
            if definition_name is None:
                raise ValueError(
                    "tool_definition must be an OpenAI tool schema with function.name"
                )
            if definition_name != normalized:
                raise ValueError(
                    "tool_definition.function.name must exactly match tool_name "
                    f"({definition_name!r} != {normalized!r})"
                )

        return normalized

    def _merge_rollout_tools(
        self,
        tools: Any,
        tool_defs: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        merged: list[dict[str, Any]] = []
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    merged.append(tool)

        existing = {
            name for item in merged if (name := self._tool_name(item)) is not None
        }
        for name, tool in tool_defs.items():
            if name in existing:
                continue
            merged.append(tool)

        return merged or None

    def _authorized(self, request: web.Request) -> bool:
        auth = request.headers.get("Authorization", "")
        api_key = request.headers.get("x-api-key", "")
        return hmac.compare_digest(
            auth, f"Bearer {self.secret}"
        ) or hmac.compare_digest(api_key, self.secret)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", **self.stats()})

    async def _handle_exit(self, request: web.Request) -> web.Response:
        """Handle agent process exit notification.

        Called by the sandbox entrypoint after the agent process exits.
        Enqueues a sentinel ``None`` on the rollout's request queue so that
        ``next_request()`` returns immediately instead of waiting for the
        full timeout.
        """
        rollout_id = request.match_info["rollout_id"]
        with self._state_lock:
            rollout = self.active_rollouts.get(rollout_id)
        if rollout is None:
            return web.json_response({"status": "ignored", "reason": "unknown rollout_id"})

        queue = rollout.get("request_id_queue")
        if queue is not None:
            try:
                queue.put_nowait(None)  # sentinel: signals "agent exited"
            except Exception:
                pass

        _log.info(
            "interception_exit_signal rollout_id=%s",
            rollout_id,
        )
        return web.json_response({"status": "ok"})

    async def _handle_tool_call(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        rollout_id = request.match_info["rollout_id"]
        tool_name = request.match_info["tool_name"]
        with self._state_lock:
            context = self.active_rollouts.get(rollout_id)
            if context is None:
                return web.json_response({"error": "rollout not found"}, status=404)
            handlers: dict[str, ToolHandler] = context.get("tool_handlers", {})
            handler = handlers.get(tool_name)
            if handler is None:
                return web.json_response({"error": "tool not found"}, status=404)

        try:
            body = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"invalid JSON: {exc}"}, status=400)

        arguments_raw: Any
        if isinstance(body, dict) and "arguments" in body:
            arguments_raw = body.get("arguments")
        else:
            arguments_raw = body

        if arguments_raw is None:
            arguments = {}
        elif isinstance(arguments_raw, dict):
            arguments = arguments_raw
        else:
            return web.json_response(
                {"error": "tool arguments must be a JSON object"},
                status=400,
            )

        try:
            response = await handler(arguments)
        except Exception:
            _log.exception(
                "tool handler failed (rollout=%s, tool=%s)",
                rollout_id,
                tool_name,
            )
            return web.json_response({"error": "tool execution failed"}, status=500)

        if not isinstance(response, dict):
            return web.json_response(
                {"error": "tool handler must return a JSON object"},
                status=500,
            )
        return web.json_response(response)

    async def _handle_chat_completions(
        self, request: web.Request
    ) -> web.StreamResponse | web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        rollout_id = request.match_info["rollout_id"]
        with self._state_lock:
            context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response({"error": "rollout not found"}, status=404)

        try:
            body = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"invalid JSON: {exc}"}, status=400)

        tool_defs: dict[str, dict[str, Any]] = dict(context.get("tool_defs", {}))
        merged_tools = self._merge_rollout_tools(body.get("tools"), tool_defs)
        if merged_tools is not None:
            body["tools"] = merged_tools

        is_streaming = bool(body.get("stream"))
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        chunk_queue: asyncio.Queue | None = asyncio.Queue() if is_streaming else None

        intercept: dict[str, Any] = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": body.get("messages"),
            "model": body.get("model"),
            "tools": body.get("tools"),
            "stream": is_streaming,
            "chunk_queue": chunk_queue,
            "response_future": asyncio.get_running_loop().create_future(),
            "body": body,
        }
        with self._state_lock:
            context = self.active_rollouts.get(rollout_id)
            if context is None:
                return web.json_response({"error": "rollout not found"}, status=404)
            self.intercepts[request_id] = intercept
            request_queue: _queue_mod.Queue[str] = context["request_id_queue"]
        request_queue.put_nowait(request_id)

        if is_streaming:
            return await self._stream_response(request, intercept)

        try:
            response_dict = await intercept["response_future"]
        except asyncio.CancelledError:
            return web.json_response({"error": "rollout cancelled"}, status=499)
        except Exception:
            _log.exception("interception request %s failed", request_id)
            return web.json_response({"error": "internal error"}, status=500)
        finally:
            with self._state_lock:
                self.intercepts.pop(request_id, None)

        return web.json_response(response_dict)

    async def _stream_response(
        self, request: web.Request, intercept: dict[str, Any]
    ) -> web.StreamResponse:
        chunk_queue: asyncio.Queue = intercept["chunk_queue"]
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        get_task: asyncio.Task | None = None
        try:
            while True:
                if get_task is None:
                    get_task = asyncio.create_task(chunk_queue.get())
                done, _ = await asyncio.wait({get_task}, timeout=_KEEPALIVE_INTERVAL_S)
                if get_task not in done:
                    await resp.write(b": keepalive\n\n")
                    continue
                chunk = get_task.result()
                get_task = None
                if chunk is None:
                    await resp.write(b"data: [DONE]\n\n")
                    break
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await asyncio.sleep(0)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            if get_task and not get_task.done():
                get_task.cancel()
            fut: asyncio.Future | None = intercept.get("response_future")
            if fut and not fut.done():
                fut.cancel()
            request_id = intercept.get("request_id")
            if isinstance(request_id, str):
                with self._state_lock:
                    self.intercepts.pop(request_id, None)
        try:
            await resp.write_eof()
        except Exception:
            pass
        return resp


def _resolve_future_threadsafe(future: asyncio.Future, value: Any) -> None:
    """Set a future's result from any thread.

    ``asyncio.Future`` is not thread-safe: calling ``set_result`` from a
    thread that is not running the future's event loop can silently fail
    to wake the coroutine awaiting it.  This helper detects cross-loop
    calls and uses ``call_soon_threadsafe`` to schedule the resolution on
    the correct loop.
    """
    if future.done():
        return
    loop = future.get_loop()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        future.set_result(value)
    else:
        loop.call_soon_threadsafe(future.set_result, value)


def _put_queue_threadsafe(q: asyncio.Queue, item: Any) -> None:
    """Put an item on an asyncio.Queue from any thread."""
    loop = getattr(q, "_loop", None)
    if loop is None:
        # Fallback: put_nowait which is simpler. Let QueueFull propagate —
        # silently dropping items would cause hard-to-debug streaming issues.
        q.put_nowait(item)
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        q.put_nowait(item)
    else:
        loop.call_soon_threadsafe(q.put_nowait, item)


async def deliver_response(
    intercept: dict[str, Any], response_dict: dict[str, Any]
) -> None:
    """Unblock the agent's HTTP handler with ``response_dict``.

    For non-streaming requests, resolves the future directly.
    For streaming requests, synthesizes SSE chunks from the complete
    response and signals EOF.

    Thread-safe: can be called from any thread, not just the event loop
    that owns the future/queue.  This is required because the rollout
    worker may run ``deliver_response`` from its own ``asyncio.run()``
    in a daemon thread while the ``InterceptionServer``'s aiohttp
    handler awaits the future on a different loop.
    """
    is_streaming = intercept.get("stream", False)
    chunk_queue: asyncio.Queue | None = intercept.get("chunk_queue")
    future: asyncio.Future | None = intercept.get("response_future")

    if not is_streaming:
        if future:
            _resolve_future_threadsafe(future, response_dict)
        return

    if chunk_queue is None:
        raise RuntimeError("chunk_queue missing on streaming intercept")

    choices = response_dict.get("choices") or []
    for choice in choices:
        msg = choice.get("message") or {}
        content_chunk = {
            "id": response_dict.get("id", ""),
            "object": "chat.completion.chunk",
            "created": response_dict.get("created", int(time.time())),
            "model": response_dict.get("model", ""),
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "delta": {
                        "role": "assistant",
                        "content": msg.get("content"),
                        "tool_calls": msg.get("tool_calls"),
                    },
                    "finish_reason": None,
                }
            ],
        }
        _put_queue_threadsafe(chunk_queue, content_chunk)
        finish_chunk = {
            "id": response_dict.get("id", ""),
            "object": "chat.completion.chunk",
            "created": response_dict.get("created", int(time.time())),
            "model": response_dict.get("model", ""),
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "delta": {},
                    "finish_reason": choice.get("finish_reason"),
                }
            ],
        }
        _put_queue_threadsafe(chunk_queue, finish_chunk)

    _put_queue_threadsafe(chunk_queue, None)
    if future:
        _resolve_future_threadsafe(future, response_dict)


__all__ = [
    "InterceptionServer",
    "deliver_response",
]
