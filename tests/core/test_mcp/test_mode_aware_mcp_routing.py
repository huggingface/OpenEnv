# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for mode-aware MCP tool discovery and execution via HTTPEnvServer (Issue #1212).

Verifies:
1. Production tools/list exposes production tools + shared tools, and excludes simulation tools.
2. Simulation tools/list exposes simulation tools + shared tools, and excludes production tools.
3. Production mode tools can be called successfully.
4. Simulation mode tools can be called successfully.
5. Mode-specific tools shadow shared FastMCP tools with the same name.
6. HTTP /mcp transport works end-to-end.
7. Direct WebSocket /mcp transport works with parity.
8. /ws transport with MCP messages works with parity.
9. MCPToolClient works end-to-end for mode-aware tools.
10. Cross-mode session reuse and WebSocket attachment are rejected.
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

        @mcp.tool
        def override_me() -> str:
            """Base FastMCP tool."""
            return "FASTMCP_BASE"

        @self.tool(mode="production")
        def override_me() -> str:  # noqa: F811
            """Overridden in production mode."""
            return "PRODUCTION_OVERRIDE"

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


class TestModeAwareMCPRouting:
    """Tests verifying mode-aware MCP tool discovery, execution, and session isolation."""

    def test_production_mode_lists_production_and_shared_tools(self, prod_server_app):
        """Production tools/list includes production and shared tools, excluding simulation tools."""
        client = TestClient(prod_server_app)

        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        assert create_resp.status_code == 200
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
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert "result" in data
        tools = data["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        assert "shared_tool" in tool_names
        assert "search_live" in tool_names
        assert "search_mock" not in tool_names

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert "input_schema" not in tool

    def test_simulation_mode_lists_simulation_and_shared_tools(self, sim_server_app):
        """Simulation tools/list includes simulation and shared tools, excluding production tools."""
        client = TestClient(sim_server_app)

        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        assert create_resp.status_code == 200
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
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert "result" in data
        tools = data["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        assert "shared_tool" in tool_names
        assert "search_mock" in tool_names
        assert "search_live" not in tool_names

    def test_production_mode_calls_production_and_shared_tools(self, prod_server_app):
        """Production tools/call successfully executes production and shared tools, rejecting simulation tools."""
        client = TestClient(prod_server_app)

        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
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

        # Calling simulation tool in production returns error
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

    def test_simulation_mode_calls_simulation_tools(self, sim_server_app):
        """Simulation tools/call successfully executes simulation tools, rejecting production tools."""
        client = TestClient(sim_server_app)

        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        session_id = create_resp.json()["result"]["session_id"]

        # Call simulation tool
        call_sim = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_mock",
                    "arguments": {"query": "mock-weather"},
                    "session_id": session_id,
                },
                "id": 2,
            },
        )
        assert call_sim.status_code == 200
        res_sim = call_sim.json()
        assert "result" in res_sim
        assert res_sim["result"]["data"] == "MOCK: mock-weather"

        # Calling production tool in simulation returns error
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
        assert "error" in res_prod
        assert "not available in simulation mode" in res_prod["error"]["message"]

    def test_mode_aware_tool_shadows_fastmcp_shared_tool(self, prod_server_app):
        """Production mode tool overrides a shared FastMCP tool with the same name."""
        client = TestClient(prod_server_app)

        # tools/list should return exactly one override_me tool
        list_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1},
        )
        assert list_resp.status_code == 200
        tools = list_resp.json()["result"]["tools"]
        matches = [t for t in tools if t["name"] == "override_me"]
        assert len(matches) == 1
        assert matches[0]["description"] == "Overridden in production mode."

        # Calling it returns the production implementation
        call_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "override_me", "arguments": {}},
                "id": 2,
            },
        )
        assert call_resp.status_code == 200
        assert call_resp.json()["result"]["data"] == "PRODUCTION_OVERRIDE"

    def test_direct_websocket_mcp_parity(self, prod_server_app):
        """Direct /mcp WebSocket transport lists and calls mode-aware tools with parity."""
        client = TestClient(prod_server_app)

        with client.websocket_connect("/mcp") as ws:
            ws.send_text(
                json.dumps(
                    {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1}
                )
            )
            list_msg = json.loads(ws.receive_text())
            tool_names = [t["name"] for t in list_msg["result"]["tools"]]
            assert "search_live" in tool_names
            assert "search_mock" not in tool_names

            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "search_live",
                            "arguments": {"query": "ws-test"},
                        },
                        "id": 2,
                    }
                )
            )
            call_msg = json.loads(ws.receive_text())
            assert call_msg["result"]["data"] == "LIVE: ws-test"

    def test_ws_mcp_message_parity(self, prod_server_app):
        """Persistent /ws WebSocket with WSMCPMessage routes mode-aware tools."""
        client = TestClient(prod_server_app)

        with client.websocket_connect("/ws") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "tools/list",
                            "params": {},
                            "id": 1,
                        },
                    }
                )
            )
            resp_list = json.loads(ws.receive_text())
            assert resp_list["type"] == "mcp"
            tools = resp_list["data"]["result"]["tools"]
            tool_names = [t["name"] for t in tools]
            assert "search_live" in tool_names
            assert "search_mock" not in tool_names

            ws.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": "search_live",
                                "arguments": {"query": "ws-mcp"},
                            },
                            "id": 2,
                        },
                    }
                )
            )
            resp_call = json.loads(ws.receive_text())
            assert resp_call["type"] == "mcp"
            assert resp_call["data"]["result"]["data"] == "LIVE: ws-mcp"

    @pytest.mark.asyncio
    async def test_mcp_tool_client_end_to_end(self, prod_server_app):
        """MCPToolClient discovers and executes production mode tools."""
        transport = httpx.ASGITransport(app=prod_server_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as http_client:
            mcp_client = MCPToolClient(base_url="http://testserver")
            mcp_client._http_client = http_client

            tools = await mcp_client.list_tools()
            tool_names = [t.name for t in tools]
            assert "search_live" in tool_names
            assert "shared_tool" in tool_names
            assert "search_mock" not in tool_names

            res = await mcp_client.call_tool("search_live", query="client-test")
            assert res == "LIVE: client-test"

    def test_cross_app_session_reuse_rejected(self):
        """A session created in production mode cannot be accessed from a simulation app, and vice-versa."""
        server = HTTPEnvServer(
            env=ModeAwareTestEnvironment,
            action_cls=CallToolAction,
            observation_cls=CallToolObservation,
            max_concurrent_envs=4,
        )
        prod_app = FastAPI()
        sim_app = FastAPI()
        server.register_routes(prod_app, mode="production")
        server.register_routes(sim_app, mode="simulation")

        prod_client = TestClient(prod_app)
        sim_client = TestClient(sim_app)

        # Create a production session
        prod_create = prod_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        assert prod_create.status_code == 200
        prod_sid = prod_create.json()["result"]["session_id"]

        # Attempt to use production session from simulation app -> rejected with -32602
        sim_call = sim_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_live",
                    "arguments": {"query": "test"},
                    "session_id": prod_sid,
                },
                "id": 2,
            },
        )
        assert sim_call.status_code == 200
        err = sim_call.json().get("error")
        assert err is not None
        assert err["code"] == -32602
        assert "belongs to mode 'production', not 'simulation'" in err["message"]

    def test_cross_app_websocket_session_attach_rejected(self):
        """Attaching to a WebSocket session across different mode apps on the same server is rejected."""
        server = HTTPEnvServer(
            env=ModeAwareTestEnvironment,
            action_cls=CallToolAction,
            observation_cls=CallToolObservation,
            max_concurrent_envs=4,
        )
        prod_app = FastAPI()
        sim_app = FastAPI()
        server.register_routes(prod_app, mode="production")
        server.register_routes(sim_app, mode="simulation")

        prod_client = TestClient(prod_app)
        sim_client = TestClient(sim_app)

        # Create a production session
        prod_create = prod_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        prod_sid = prod_create.json()["result"]["session_id"]

        # Attempt WebSocket attach via simulation app -> error message received
        with sim_client.websocket_connect(f"/ws?session_id={prod_sid}") as ws:
            raw = ws.receive_text()
            data = json.loads(raw)
            assert data["type"] == "error"
            assert (
                "belongs to mode 'production', not 'simulation'"
                in data["data"]["message"]
            )
