# SPDX-License-Identifier: BSD-3-Clause
"""Clients for privileged surfaces; no server imports or reward computation."""

import json
from typing import Any

import httpx
from websockets.asyncio.client import connect


class GraderClient:
    """Async MCP client bound to the grader endpoint and its credential.

    Args:
        endpoint (`str`):
            Full URL of the grader surface, ending in `/mcp/grader`.
        token (`str`):
            Grader credential supplied by the operator.
        timeout_s (`float`, *optional*, defaults to `300`):
            Request timeout, including oracle execution.
    """

    def __init__(self, endpoint: str, token: str, timeout_s: float = 300):
        if not token or not token.isascii() or not token.isprintable() or " " in token:
            raise ValueError("invalid grader bearer token")
        self.endpoint = endpoint
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"}, timeout=timeout_s
        )
        self._next_id = 0

    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._client.__aexit__(*args)

    async def _request(self, method: str, params: dict):
        self._next_id += 1
        response = await self._client.post(
            self.endpoint,
            json={
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": method,
                "params": params,
            },
        )
        response.raise_for_status()
        message = response.json()
        if message.get("error"):
            raise RuntimeError(message["error"]["message"])
        return message["result"]

    async def list_tools(self) -> list[dict]:
        return (await self._request("tools/list", {}))["tools"]

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict:
        return await self._request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )

    async def run_oracle(self) -> dict:
        """Reference oracle-replay consumer for environment-side contract graders."""
        result = await self.call_tool("grader.run_oracle")
        return json.loads(result["content"][0]["text"])


async def observer_stream(endpoint: str, token: str):
    """Yield typed JSON event payloads from an authenticated observer WebSocket."""
    async with connect(
        endpoint, additional_headers={"Authorization": f"Bearer {token}"}
    ) as websocket:
        async for message in websocket:
            yield json.loads(message)
