# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio

import aiohttp
import pytest

from openenv.core.harness.agents.interception_server import (
    InterceptionServer,
    deliver_response,
)


@pytest.mark.asyncio
async def test_interception_server_rejects_unauthorized_requests() -> None:
    server = InterceptionServer(port=0, secret="secret-token")
    await server.start()
    try:
        async with aiohttp.ClientSession() as client:
            resp = await client.post(
                f"http://127.0.0.1:{server.port}/rollout/r1/v1/chat/completions",
                json={"messages": []},
            )
            assert resp.status == 401
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_interception_server_returns_404_for_unknown_rollout() -> None:
    server = InterceptionServer(port=0, secret="secret-token")
    await server.start()
    try:
        async with aiohttp.ClientSession() as client:
            resp = await client.post(
                f"http://127.0.0.1:{server.port}/rollout/missing/v1/chat/completions",
                headers={"Authorization": "Bearer secret-token"},
                json={"messages": []},
            )
            assert resp.status == 404
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_interception_server_non_stream_roundtrip_cleans_intercept() -> None:
    server = InterceptionServer(port=0, secret="secret-token")
    await server.start()
    queue = server.register_rollout("r1")
    try:
        async with aiohttp.ClientSession() as client:
            request_task = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{server.port}/rollout/r1/v1/chat/completions",
                    headers={"Authorization": "Bearer secret-token"},
                    json={
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                )
            )
            request_id = await asyncio.wait_for(queue.get(), timeout=1.0)
            intercept = server.get_intercept(request_id)
            assert intercept is not None

            await deliver_response(
                intercept,
                {
                    "id": "resp-1",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

            resp = await request_task
            assert resp.status == 200
            payload = await resp.json()
            assert payload["id"] == "resp-1"

            # Request entries should not leak after completion.
            assert server.get_intercept(request_id) is None
    finally:
        server.unregister_rollout("r1")
        await server.stop()


@pytest.mark.asyncio
async def test_interception_server_unregister_rollout_cancels_pending_request() -> None:
    server = InterceptionServer(port=0, secret="secret-token")
    await server.start()
    queue = server.register_rollout("r1")
    try:
        async with aiohttp.ClientSession() as client:
            request_task = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{server.port}/rollout/r1/v1/chat/completions",
                    headers={"Authorization": "Bearer secret-token"},
                    json={
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                )
            )
            _request_id = await asyncio.wait_for(queue.get(), timeout=1.0)
            server.unregister_rollout("r1")

            resp = await request_task
            assert resp.status == 499
            payload = await resp.json()
            assert payload["error"] == "rollout cancelled"
    finally:
        await server.stop()
