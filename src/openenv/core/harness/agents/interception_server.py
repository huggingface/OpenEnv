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

    server = InterceptionServer(port=8765)
    await server.start()

    # Make the server reachable — your responsibility.
    # Docker: base_url = f"http://host.docker.internal:{server.port}"
    # Remote: base_url = your_tunnel_or_public_url

    queue = server.register_rollout(rollout_id)
    # Agent runs with OPENAI_BASE_URL = f"{base_url}/rollout/{rollout_id}/v1"

    while True:
        request_id = await asyncio.wait_for(queue.get(), timeout=...)
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
import secrets
import threading
import time
import uuid
from typing import Any

from aiohttp import web


_log = logging.getLogger(__name__)

_KEEPALIVE_INTERVAL_S = 3.0
_MAX_REQUEST_BODY = 16 * 1024 * 1024


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
    ) -> None:
        self.port = port
        self.host = host
        self.secret = secret or secrets.token_urlsafe(32)
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
    ) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        with self._state_lock:
            self.active_rollouts[rollout_id] = {
                "request_id_queue": queue,
                "state": state,
            }
        return queue

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
            self.active_rollouts.pop(rollout_id, None)

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

    def get_intercept(self, request_id: str) -> dict[str, Any] | None:
        with self._state_lock:
            return self.intercepts.get(request_id)

    def _authorized(self, request: web.Request) -> bool:
        auth = request.headers.get("Authorization", "")
        api_key = request.headers.get("x-api-key", "")
        return hmac.compare_digest(
            auth, f"Bearer {self.secret}"
        ) or hmac.compare_digest(api_key, self.secret)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

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
            request_queue: asyncio.Queue = context["request_id_queue"]
        await request_queue.put(request_id)

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


async def deliver_response(
    intercept: dict[str, Any], response_dict: dict[str, Any]
) -> None:
    """Unblock the agent's HTTP handler with ``response_dict``.

    For non-streaming requests, resolves the future directly.
    For streaming requests, synthesizes SSE chunks from the complete
    response and signals EOF.
    """
    is_streaming = intercept.get("stream", False)
    chunk_queue: asyncio.Queue | None = intercept.get("chunk_queue")
    future: asyncio.Future | None = intercept.get("response_future")

    if not is_streaming:
        if future and not future.done():
            future.set_result(response_dict)
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
        await chunk_queue.put(content_chunk)
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
        await chunk_queue.put(finish_chunk)

    await chunk_queue.put(None)
    if future and not future.done():
        future.set_result(response_dict)


__all__ = [
    "InterceptionServer",
    "deliver_response",
]
