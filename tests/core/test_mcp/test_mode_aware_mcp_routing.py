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
import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import FastMCP
from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
)
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

        @self.tool(mode="production")
        def slow_tool(delay: float = 0.15) -> str:
            """Slow synchronous production tool."""
            time.sleep(delay)
            return "done"

        @self.tool(mode="production")
        def calculate_tax(amount: int) -> int:
            """Calculate tax with strict integer typing."""
            return amount * 10

        @self.tool(mode="simulation")
        def search_mock(query: str) -> str:
            """Simulation-only tool querying local database."""
            return f"MOCK: {query}"

        @self.tool(mode="production")
        def structured_prod_tool(tag: str) -> dict:
            """Production tool returning structured data."""
            return {"status": "ok", "tag": tag, "items": [1, 2]}

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

        # Verify MCP standard wire format: inputSchema must be present and input_schema absent
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool, (
                f"Tool {tool['name']} missing standard inputSchema"
            )
            assert "input_schema" not in tool, (
                f"Tool {tool['name']} must not expose internal input_schema"
            )
            assert isinstance(tool["inputSchema"], dict)

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

    def test_production_mode_tool_missing_arguments_returns_invalid_params(
        self, prod_server_app
    ):
        """Calling a mode-aware tool with missing required arguments returns -32602 (INVALID_PARAMS)."""
        client = TestClient(prod_server_app)
        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        session_id = create_resp.json()["result"]["session_id"]

        call_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_live",
                    "arguments": {},
                    "session_id": session_id,
                },
                "id": 2,
            },
        )
        assert call_resp.status_code == 200
        assert call_resp.json()["error"]["code"] == -32602

    def test_production_mode_tool_invalid_argument_types_returns_invalid_params(
        self, prod_server_app
    ):
        """Mode-aware tools validate argument types: string or bool passed to int parameter returns -32602."""
        client = TestClient(prod_server_app)
        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        session_id = create_resp.json()["result"]["session_id"]

        # String passed to int parameter
        call_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "calculate_tax",
                    "arguments": {"amount": "not-an-integer"},
                    "session_id": session_id,
                },
                "id": 2,
            },
        )
        assert call_resp.status_code == 200
        assert call_resp.json()["error"]["code"] == -32602

        # Bool passed to int parameter
        call_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "calculate_tax",
                    "arguments": {"amount": True},
                    "session_id": session_id,
                },
                "id": 3,
            },
        )
        assert call_resp.status_code == 200
        assert call_resp.json()["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_mode_aware_tool_timeout_via_http_mcp(self, prod_server_app):
        """Async timeout enforcement: HTTP /mcp tools/call returns TIMEOUT within wall-clock bound.

        Verifies that the async _async_handle_call_tool path returns a TIMEOUT
        error in approximately timeout_s seconds, not the full tool duration.
        NOTE: Sync env.step() timeout is best-effort for sync tools because
        Python threads cannot be interrupted; the async transport path is the
        primary enforcement mechanism.
        """
        transport = httpx.ASGITransport(app=prod_server_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            create_resp = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
            )
            session_id = create_resp.json()["result"]["session_id"]

            start = time.monotonic()
            call_resp = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "slow_tool",
                        "arguments": {"delay": 2.0},
                        "session_id": session_id,
                        "timeout_s": 0.15,
                    },
                    "id": 2,
                },
            )
            elapsed = time.monotonic() - start

            assert call_resp.status_code == 200
            res = call_resp.json()
            assert "error" in res
            assert "timed out" in res["error"]["message"].lower()
            # Wall-clock must be well under the tool's 2s sleep
            assert elapsed < 1.0, f"Timeout took {elapsed:.2f}s, expected < 1.0s"

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

            # Verify MCP standard wire format over WebSocket
            for tool in tools:
                assert "name" in tool
                assert "description" in tool
                assert "inputSchema" in tool, (
                    f"Tool {tool['name']} missing inputSchema over WS"
                )
                assert "input_schema" not in tool, (
                    f"Tool {tool['name']} must not expose internal input_schema over WS"
                )
                assert isinstance(tool["inputSchema"], dict)

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

    def test_direct_websocket_mcp_mode_aware_parity(self):
        """Direct JSON-RPC over WebSocket endpoint /mcp establishes its session over WS and executes production tools."""
        # Use dedicated server app with explicit capacity (max_concurrent_envs=2) to isolate from HTTP sessions
        app = FastAPI()
        server = HTTPEnvServer(
            env=ModeAwareTestEnvironment,
            action_cls=CallToolAction,
            observation_cls=CallToolObservation,
            max_concurrent_envs=2,
        )
        server.register_routes(app, mode="production")
        client = TestClient(app)

        with client.websocket_connect("/mcp") as ws:
            # 1. tools/list over direct /mcp WebSocket
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "id": 101,
                    }
                )
            )
            resp_list = json.loads(ws.receive_text())
            assert resp_list.get("id") == 101
            assert "result" in resp_list
            tools = resp_list["result"]["tools"]
            tool_names = [t["name"] for t in tools]
            assert "shared_tool" in tool_names
            assert "search_live" in tool_names
            assert "search_mock" not in tool_names

            # 2. tools/call over direct /mcp WebSocket
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "search_live",
                            "arguments": {"query": "direct-ws"},
                        },
                        "id": 102,
                    }
                )
            )
            resp_call = json.loads(ws.receive_text())
            assert resp_call.get("id") == 102
            assert "result" in resp_call
            assert resp_call["result"]["data"] == "LIVE: direct-ws"

    def test_direct_websocket_mcp_capacity_freed(self, prod_server_app):
        """Direct /mcp WebSocket connects cleanly when capacity is explicitly freed from prior HTTP session."""
        client = TestClient(prod_server_app)

        # 1. Create an HTTP session occupying the default single capacity slot
        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["result"]["session_id"]

        # 2. Explicitly close the HTTP session to free capacity
        close_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": session_id},
                "id": 2,
            },
        )
        assert close_resp.status_code == 200
        assert close_resp.json()["result"]["closed"] is True

        # 3. Direct WebSocket connects successfully and executes mode-aware tools
        with client.websocket_connect("/mcp") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "id": 103,
                    }
                )
            )
            resp_list = json.loads(ws.receive_text())
            assert resp_list.get("id") == 103
            assert "result" in resp_list
            tool_names = [t["name"] for t in resp_list["result"]["tools"]]
            assert "search_live" in tool_names

    def test_mode_aware_tool_shadows_fastmcp_shared_tool(self, prod_server_app):
        """Mode-aware tool overrides an underlying FastMCP tool of the same name without duplication."""
        client = TestClient(prod_server_app)
        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        session_id = create_resp.json()["result"]["session_id"]

        # tools/list should list override_me exactly once
        resp_list = client.post(
            f"/mcp?session_id={session_id}",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 2},
        )
        tools = resp_list.json()["result"]["tools"]
        matches = [t for t in tools if t["name"] == "override_me"]
        assert len(matches) == 1

        # tools/call should execute the production override, not the FastMCP base
        resp_call = client.post(
            f"/mcp?session_id={session_id}",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "override_me", "arguments": {}},
                "id": 3,
            },
        )
        res = resp_call.json()
        assert res["result"]["data"] == "PRODUCTION_OVERRIDE"

    def test_production_mode_tool_structured_return(self, prod_server_app):
        """Mode-aware tool returning structured dict is properly serialized over JSON-RPC."""
        client = TestClient(prod_server_app)
        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        session_id = create_resp.json()["result"]["session_id"]

        resp_call = client.post(
            f"/mcp?session_id={session_id}",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "structured_prod_tool",
                    "arguments": {"tag": "prod-v1"},
                },
                "id": 4,
            },
        )
        res = resp_call.json()["result"]
        assert res["structured_content"]["result"] == {
            "status": "ok",
            "tag": "prod-v1",
            "items": [1, 2],
        }
        assert res["data"] == {
            "status": "ok",
            "tag": "prod-v1",
            "items": [1, 2],
        }


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

        # Verify MCP standard wire format in simulation mode
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool, (
                f"Tool {tool['name']} missing inputSchema in sim mode"
            )
            assert "input_schema" not in tool, (
                f"Tool {tool['name']} must not expose internal input_schema in sim mode"
            )
            assert isinstance(tool["inputSchema"], dict)


class TestServerModeNonMCPEnvironment:
    """Tests verifying server mode does not mutate unrelated private attributes on non-MCP environments."""

    def test_non_mcp_environment_private_mode_attribute_is_not_overwritten(self):
        """HTTPEnvServer configured with a mode must not overwrite private _mode on non-MCP envs."""

        class NonMCPDomainEnvironment(Environment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def __init__(self):
                self._mode = "custom_domain_mode"
                self._state = State(episode_id="ep1", step_count=0)

            def reset(self, **kwargs) -> Observation:
                return Observation(done=False, reward=None)

            def step(self, action: Action, **kwargs) -> Observation:
                return Observation(done=False, reward=None)

            @property
            def state(self) -> State:
                return self._state

        app = FastAPI()
        server = HTTPEnvServer(
            env=NonMCPDomainEnvironment,
            action_cls=Action,
            observation_cls=Observation,
        )
        server.register_routes(app, mode="production")
        client = TestClient(app)

        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        assert resp.status_code == 200
        session_id = resp.json()["result"]["session_id"]
        assert server._sessions[session_id]._mode == "custom_domain_mode"


class TestMultiAppModeIsolationAndRobustness:
    """Tests verifying app-scoped mode isolation, REST endpoints mode awareness, and error resilience."""

    def test_mode_isolation_across_multiple_apps_on_same_server(self):
        """Registering multiple apps on one server keeps each app's mode isolated."""
        server = HTTPEnvServer(
            env=ModeAwareTestEnvironment,
            action_cls=CallToolAction,
            observation_cls=CallToolObservation,
            max_concurrent_envs=4,
        )

        app_prod = FastAPI()
        server.register_routes(app_prod, mode="production")

        app_sim = FastAPI()
        server.register_routes(app_sim, mode="simulation")

        client_prod = TestClient(app_prod)
        client_sim = TestClient(app_sim)

        # 1. Create session on production app -> only production & shared tools
        prod_create = client_prod.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        assert prod_create.status_code == 200
        assert "result" in prod_create.json()
        prod_sid = prod_create.json()["result"]["session_id"]

        prod_list = client_prod.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {"session_id": prod_sid},
                "id": 2,
            },
        )
        prod_tools = [t["name"] for t in prod_list.json()["result"]["tools"]]
        assert "search_live" in prod_tools
        assert "shared_tool" in prod_tools
        assert "search_mock" not in prod_tools

        # 2. Create session on simulation app -> only simulation & shared tools
        sim_create = client_sim.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 3},
        )
        assert sim_create.status_code == 200
        assert "result" in sim_create.json()
        sim_sid = sim_create.json()["result"]["session_id"]

        sim_list = client_sim.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {"session_id": sim_sid},
                "id": 4,
            },
        )
        sim_tools = [t["name"] for t in sim_list.json()["result"]["tools"]]
        assert "search_mock" in sim_tools
        assert "shared_tool" in sim_tools
        assert "search_live" not in sim_tools

    def test_simulation_mode_rest_step_receives_mode(self):
        """Simulation REST /step endpoint properly configures simulation mode on the created env."""
        app = FastAPI()
        server = HTTPEnvServer(
            env=ModeAwareTestEnvironment,
            action_cls=CallToolAction,
            observation_cls=CallToolObservation,
        )
        server.register_routes(app, mode="simulation")
        client = TestClient(app)

        resp = client.post(
            "/step",
            json={
                "action": {
                    "tool_name": "search_mock",
                    "arguments": {"query": "sim-query"},
                }
            },
        )
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs["error"] is None
        assert obs["result"]["data"] == "MOCK: sim-query"

    def test_set_mode_failure_does_not_leak_environment(self):
        """Failure inside set_mode during session creation cleans up executor and does not leak session slots."""

        class RaisingSetModeEnvironment(ModeAwareTestEnvironment):
            def set_mode(self, mode: str | None = None) -> None:
                raise RuntimeError("Simulated set_mode crash")

        app = FastAPI()
        server = HTTPEnvServer(
            env=RaisingSetModeEnvironment,
            action_cls=CallToolAction,
            observation_cls=CallToolObservation,
            max_concurrent_envs=1,
        )
        server.register_routes(app, mode="production")
        client = TestClient(app)

        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

        # Session map must be completely empty (slot reclaimed)
        assert len(server._sessions) == 0
        assert len(server._session_executors) == 0
