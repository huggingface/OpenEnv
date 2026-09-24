# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openenv.core.generic_client import GenericEnvClient
from openenv.core.openenvd.client import GraderClient


@pytest.mark.asyncio
async def test_orchestrator_credential_only_in_handshake():
    headers = {"Authorization": "Bearer private"}
    client = GenericEnvClient(base_url="http://localhost:8100", headers=headers)
    headers["Authorization"] = "mutated"
    socket = AsyncMock()
    with patch(
        "openenv.core.env_client.ws_connect", new=AsyncMock(return_value=socket)
    ) as connect:
        await client.connect()
        assert connect.call_args.kwargs["additional_headers"] == {
            "Authorization": "Bearer private"
        }
        socket.send.assert_not_called()
        session = await client.new_session()
        assert connect.call_args.kwargs["additional_headers"] == {
            "Authorization": "Bearer private"
        }
        assert session._headers is not client._headers
        await client.close()


@pytest.mark.asyncio
async def test_grader_reference_oracle_consumer():
    async def handler(request):
        assert request.url.path == "/mcp/grader"
        assert request.headers["authorization"] == "Bearer grade-secret"
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"returncode":0,"stdout":"ok","stderr":""}',
                        }
                    ]
                },
            },
        )

    client = GraderClient("http://test/mcp/grader", "grade-secret")
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer grade-secret"},
    )
    async with client:
        assert (await client.run_oracle())["returncode"] == 0


@pytest.mark.asyncio
async def test_handshake_error_does_not_echo_credentials():
    client = GenericEnvClient(
        base_url="http://localhost:8100", headers={"Authorization": "Bearer private"}
    )
    with patch(
        "openenv.core.env_client.ws_connect",
        new=AsyncMock(side_effect=ValueError("invalid header: Bearer private")),
    ):
        with pytest.raises(ConnectionError) as raised:
            await client.connect()
        assert "private" not in str(raised.value)
        assert raised.value.__cause__ is None
