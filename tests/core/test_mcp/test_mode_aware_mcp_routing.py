# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for mode-aware MCP tool discovery and execution via HTTPEnvServer.

Verifies that:
1. @self.tool(mode="production") tools are exposed via /mcp tools/list in production mode.
2. @self.tool(mode="simulation") tools are omitted via /mcp tools/list in production mode.
3. Production mode tools execute properly via /mcp tools/call and MCPToolClient.
4. Mode-agnostic (@mcp.tool) tools continue to function alongside mode-specific tools.
5. Simulation mode servers properly serve simulation-specific tools.
6. WebSocket MCP path (WSMCPMessage) handles mode-aware tools with parity.
"""

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import FastMCP
from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.mcp_client import MCPToolClient


class ModeAwareTestEnvironment(MCPEnvironment):
    """Test environment defining mode-agnostic and mode-specific tools."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("mode-aware-test")
        super().__init__(mcp)

        @mcp.tool
        def shared_tool(value: int) -> int:
            """Tool available in all modes."""
            return value * 2

        @self.tool(mode="production")
        def search_live(query: str) -> str:
            """Production-only tool searching live API."""
            return f"LIVE: {query}"

        @self.tool(mode="simulation")
        def search_mock(query: str) -> str:
            """Simulation-only tool querying local database."""
            return f"MOCK: {query}"

        self._state = State(episode_id="test-ep", step_count=0)

    def reset(self, **kwargs) -> Observation:
        return Observation(done=False, reward=None)

    def _step_impl(self, action: Action, **kwargs) -> Observation:
        return Observation(done=False, reward=None)

    @property
    def state(self) -> State:
        return self._state


@pytest.fixture
def prod_server_app() -> FastAPI:
    """Create a FastAPI app configured in production mode."""
    app = FastAPI()
    server = HTTPEnvServer(
        env=ModeAwareTestEnvironment,
        action_cls=CallToolAction,
        observation_cls=CallToolObservation,
    )
    server.register_routes(app, mode="production")
    return app


@pytest.fixture
def sim_server_app() -> FastAPI:
    """Create a FastAPI app configured in simulation mode."""
    app = FastAPI()
    server = HTTPEnvServer(
        env=ModeAwareTestEnvironment,
        action_cls=CallToolAction,
        observation_cls=CallToolObservation,
    )
    server.register_routes(app, mode="simulation")
    return app


class TestProductionModeAwareMCP:
    """Tests verifying mode-aware MCP tools in production mode."""

    def test_production_mode_lists_production_and_shared_tools(self, prod_server_app):
        """Production tools/list should include production and shared tools, excluding simulation tools."""
        client = TestClient(prod_server_app)

        # Create session
        create_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "id": 1,
            },
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["result"]["session_id"]

        # List tools
        list_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {"session_id": session_id},
                "id": 2,
            },
        )
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert "result" in data
        tools = data["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        assert "shared_tool" in tool_names
        assert "search_live" in tool_names
        assert "search_mock" not in tool_names

    def test_production_mode_calls_production_and_shared_tools(self, prod_server_app):
        """Production tools/call should successfully execute production and shared tools."""
        client = TestClient(prod_server_app)

        # Create session
        create_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "id": 1,
            },
        )
        session_id = create_resp.json()["result"]["session_id"]

        # Call shared tool
        call_shared = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "shared_tool",
                    "arguments": {"value": 21},
                    "session_id": session_id,
                },
                "id": 2,
            },
        )
        assert call_shared.status_code == 200
        res_shared = call_shared.json()
        assert "result" in res_shared
        assert res_shared["result"]["data"] == 42

        # Call production tool
        call_prod = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_live",
                    "arguments": {"query": "weather"},
                    "session_id": session_id,
                },
                "id": 3,
            },
        )
        assert call_prod.status_code == 200
        res_prod = call_prod.json()
        assert "result" in res_prod
        assert res_prod["result"]["data"] == "LIVE: weather"

        # Calling simulation-only tool in production should fail
        call_sim = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_mock",
                    "arguments": {"query": "weather"},
                    "session_id": session_id,
                },
                "id": 4,
            },
        )
        assert call_sim.status_code == 200
        res_sim = call_sim.json()
        assert "error" in res_sim
        assert "not available in production mode" in res_sim["error"]["message"]

    def test_websocket_mcp_message_mode_aware_parity(self, prod_server_app):
        """WebSocket /ws with type='mcp' should handle mode-aware tools identically to HTTP /mcp."""
        client = TestClient(prod_server_app)

        create_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "id": 1,
            },
        )
        session_id = create_resp.json()["result"]["session_id"]

        with client.websocket_connect(f"/ws?session_id={session_id}") as ws:
            # tools/list over WebSocket
            ws.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "tools/list",
                            "id": 10,
                        },
                    }
                )
            )
            resp_list = json.loads(ws.receive_text())
            assert resp_list["type"] == "mcp"
            tools = resp_list["data"]["result"]["tools"]
            tool_names = [t["name"] for t in tools]
            assert "shared_tool" in tool_names
            assert "search_live" in tool_names
            assert "search_mock" not in tool_names

            # tools/call over WebSocket
            ws.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": "search_live",
                                "arguments": {"query": "flights"},
                            },
                            "id": 11,
                        },
                    }
                )
            )
            resp_call = json.loads(ws.receive_text())
            assert resp_call["type"] == "mcp"
            assert resp_call["data"]["result"]["data"] == "LIVE: flights"

    @pytest.mark.asyncio
    async def test_mcp_tool_client_end_to_end(self, prod_server_app):
        """MCPToolClient should list and invoke production-mode tools end-to-end."""
        transport = httpx.ASGITransport(app=prod_server_app)
        async_http = httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        )

        client = MCPToolClient(base_url="http://testserver", mode="production")
        client._http_client = async_http

        try:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            assert "shared_tool" in tool_names
            assert "search_live" in tool_names
            assert "search_mock" not in tool_names

            result = await client.call_tool("search_live", query="hotels")
            assert result == "LIVE: hotels"
        finally:
            await client.close()


class TestSimulationModeAwareMCP:
    """Tests verifying mode-aware MCP tools in simulation mode."""

    def test_simulation_mode_lists_simulation_and_shared_tools(self, sim_server_app):
        """Simulation mode tools/list should include simulation and shared tools."""
        client = TestClient(sim_server_app)

        create_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "id": 1,
            },
        )
        session_id = create_resp.json()["result"]["session_id"]

        list_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {"session_id": session_id},
                "id": 2,
            },
        )
        tools = list_resp.json()["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        assert "shared_tool" in tool_names
        assert "search_mock" in tool_names
        assert "search_live" not in tool_names
