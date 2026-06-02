# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for mode selection in OpenEnv clients and environments.

This file combines two aspects of mode selection:

1. Client mode selection (from main): Tests for selecting between WebSocket (Gym-style)
   and MCP modes through constructor parameters and environment variables. The mode
   selection determines which protocol the client uses to communicate with the server.

2. Environment code mode (from issue #347): Tests for mode selection between tool-calling
   and code mode. Per RFC 003, MCP environments should support two modes:
   - Tool-calling mode: one tool call per step (traditional MCP)
   - Code mode: code blocks with direct Python function calls (CodeAct pattern)

Test coverage:
- Client: Mode selection via constructor parameter and environment variable
- Client: GenericEnvClient and MCPToolClient mode behavior
- Environment: Code mode with get_callables() and execute_code()
- Environment: Code mode with mode-aware tool registration
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import ListToolsAction, ListToolsObservation
from openenv.core.env_server.types import Observation, State
from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient


# ============================================================================
# Client Mode Selection Tests (from main)
# ============================================================================


# ============================================================================
# Test Fixtures - Client Mode Tests
# ============================================================================


@pytest.fixture
def clean_env():
    """Ensure OPENENV_CLIENT_MODE is not set."""
    old_mode = os.environ.pop("OPENENV_CLIENT_MODE", None)
    yield
    if old_mode is not None:
        os.environ["OPENENV_CLIENT_MODE"] = old_mode


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.recv.return_value = '{"type": "response", "data": {}}'
    return ws


# ============================================================================
# Constructor Parameter Mode Selection Tests
# ============================================================================


class TestConstructorModeSelection:
    """Test mode selection via constructor parameter."""

    def test_default_mode_is_simulation(self, clean_env):
        """Test that default mode is 'simulation' when no mode specified."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        # Should have simulation mode set
        assert hasattr(client, "_mode")
        assert client._mode == "simulation"

    def test_explicit_simulation_mode(self, clean_env):
        """Test explicit simulation mode via constructor."""
        client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation")

        assert client._mode == "simulation"

    def test_explicit_production_mode(self, clean_env):
        """Test explicit production mode via constructor."""
        client = GenericEnvClient(base_url="http://localhost:8000", mode="production")

        assert client._mode == "production"

    def test_invalid_mode_raises_error(self, clean_env):
        """Test that invalid mode value raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            GenericEnvClient(base_url="http://localhost:8000", mode="invalid_mode")

        assert "mode" in str(exc_info.value).lower()
        assert "simulation" in str(exc_info.value).lower()
        assert "production" in str(exc_info.value).lower()

    def test_case_insensitive_mode(self, clean_env):
        """Test that mode parameter is case-insensitive."""
        client1 = GenericEnvClient(base_url="http://localhost:8000", mode="SIMULATION")
        client2 = GenericEnvClient(base_url="http://localhost:8000", mode="PRODUCTION")

        assert client1._mode == "simulation"
        assert client2._mode == "production"


# ============================================================================
# Environment Variable Mode Selection Tests
# ============================================================================


class TestEnvironmentVariableModeSelection:
    """Test mode selection via OPENENV_CLIENT_MODE environment variable."""

    def test_env_var_simulation_mode(self):
        """Test mode selection via OPENENV_CLIENT_MODE=simulation."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "simulation"}):
            client = GenericEnvClient(base_url="http://localhost:8000")
            assert client._mode == "simulation"

    def test_env_var_production_mode(self):
        """Test mode selection via OPENENV_CLIENT_MODE=production."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "production"}):
            client = GenericEnvClient(base_url="http://localhost:8000")
            assert client._mode == "production"

    def test_env_var_case_insensitive(self):
        """Test that OPENENV_CLIENT_MODE is case-insensitive."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "PRODUCTION"}):
            client = GenericEnvClient(base_url="http://localhost:8000")
            assert client._mode == "production"

    def test_env_var_overrides_default(self):
        """Test that environment variable overrides default mode."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "production"}):
            # No explicit mode in constructor
            client = GenericEnvClient(base_url="http://localhost:8000")
            assert client._mode == "production"

    def test_constructor_overrides_env_var(self):
        """Test that explicit constructor parameter overrides environment variable."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "production"}):
            # Explicit mode in constructor should take precedence
            client = GenericEnvClient(
                base_url="http://localhost:8000", mode="simulation"
            )
            assert client._mode == "simulation"

    def test_invalid_env_var_raises_error(self):
        """Test that invalid OPENENV_CLIENT_MODE raises ValueError."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "invalid"}):
            with pytest.raises(ValueError) as exc_info:
                GenericEnvClient(base_url="http://localhost:8000")

            assert "OPENENV_CLIENT_MODE" in str(exc_info.value)
            assert "invalid" in str(exc_info.value).lower()


# ============================================================================
# Mode Behavior Tests
# ============================================================================


class TestModeBehavior:
    """Test that different modes result in different client behavior."""

    @pytest.mark.asyncio
    async def test_simulation_mode_uses_gym_protocol(self, clean_env, mock_websocket):
        """Test that simulation mode uses Gym-style WebSocket messages."""
        client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation")

        with patch.object(client, "_send") as mock_send:
            with patch.object(
                client,
                "_receive",
                return_value={
                    "type": "response",
                    "data": {"observation": {}, "reward": None, "done": False},
                },
            ):
                with patch.object(client, "_ws", mock_websocket):
                    await client.reset()

                    # Should send WSResetMessage format
                    call_args = mock_send.call_args_list
                    reset_call = [
                        call for call in call_args if call[0][0].get("type") == "reset"
                    ]
                    assert len(reset_call) > 0, (
                        "Should send reset message with type='reset'"
                    )

    @pytest.mark.asyncio
    async def test_production_mode_uses_jsonrpc_protocol(
        self, clean_env, mock_websocket
    ):
        """Test that production mode uses JSON-RPC format for tool calls."""
        client = MCPToolClient(base_url="http://localhost:8000", mode="production")

        with patch.object(client, "_send") as mock_send:
            with patch.object(
                client,
                "_receive",
                return_value={
                    "type": "response",
                    "data": {
                        "observation": {"tools": []},
                        "reward": None,
                        "done": False,
                    },
                },
            ):
                with patch.object(client, "_ws", mock_websocket):
                    await client.list_tools()

                    # Should send step message with list_tools action
                    call_args = mock_send.call_args_list
                    step_call = [
                        call for call in call_args if call[0][0].get("type") == "step"
                    ]
                    assert len(step_call) > 0, "Should send message with type='step'"

                    # Check that the action payload is list_tools
                    step_message = step_call[0][0][0]
                    assert "data" in step_message
                    assert step_message["data"].get("type") == "list_tools"


# ============================================================================
# Mode Immutability Tests
# ============================================================================


class TestModeImmutability:
    """Test that mode cannot be changed after client creation."""

    def test_mode_cannot_be_changed_after_creation(self, clean_env):
        """Test that mode attribute is read-only after initialization."""
        client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation")

        # Attempting to change mode should raise AttributeError or have no effect
        with pytest.raises((AttributeError, ValueError)):
            client._mode = "mcp"

    def test_mode_cannot_be_changed_after_connection(self, clean_env):
        """Test that mode cannot be changed after connection is established."""
        client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation")

        with patch.object(client, "_ws", MagicMock()):
            # Mark as connected
            client._ws = MagicMock()

            # Should not allow mode change
            with pytest.raises((AttributeError, ValueError)):
                client._mode = "mcp"


# ============================================================================
# Cross-Client Mode Consistency Tests
# ============================================================================


class TestCrossClientModeConsistency:
    """Test that mode selection works consistently across different client types."""

    def test_generic_client_supports_both_modes(self, clean_env):
        """Test that GenericEnvClient supports both simulation and production modes."""
        ws_client = GenericEnvClient(
            base_url="http://localhost:8000", mode="simulation"
        )
        mcp_client = GenericEnvClient(
            base_url="http://localhost:8000", mode="production"
        )

        assert ws_client._mode == "simulation"
        assert mcp_client._mode == "production"

    def test_mcp_client_defaults_to_production_mode(self, clean_env):
        """Test that MCPToolClient defaults to 'production' mode."""
        client = MCPToolClient(base_url="http://localhost:8000")

        # MCPToolClient should default to production mode
        assert client._mode == "production"

    def test_mcp_client_cannot_use_simulation_mode(self, clean_env):
        """Test that MCPToolClient raises error if simulation mode is requested."""
        with pytest.raises(ValueError) as exc_info:
            MCPToolClient(base_url="http://localhost:8000", mode="simulation")

        assert "MCPToolClient" in str(exc_info.value)
        assert "production" in str(exc_info.value).lower()


# ============================================================================
# Mode Documentation Tests
# ============================================================================


class TestModeDocumentation:
    """Test that mode parameter is properly documented."""

    def test_mode_parameter_in_docstring(self, clean_env):
        """Test that mode parameter is documented in __init__ docstring."""
        # GenericEnvClient should document mode parameter
        docstring = GenericEnvClient.__init__.__doc__

        # Should mention mode in Args section
        assert docstring is not None
        assert "mode" in docstring.lower()

    def test_mode_values_documented(self, clean_env):
        """Test that valid mode values are documented."""
        docstring = GenericEnvClient.__init__.__doc__

        # Should document both simulation and production modes
        assert "simulation" in docstring.lower()
        assert "production" in docstring.lower()


# ============================================================================
# Environment Code Mode Tests (from issue #347)
# ============================================================================


class _TestMCPEnv(MCPEnvironment):
    """Concrete MCPEnvironment for testing with real FastMCP server."""

    def __init__(self, mcp_server):
        super().__init__(mcp_server)
        self._state = State(episode_id="test", step_count=0)

    def reset(self, **kwargs):
        self._state = State(episode_id=kwargs.get("episode_id", "test"), step_count=0)
        return Observation(done=False, reward=0.0)

    def _step_impl(self, action, **kwargs):
        self._state.step_count += 1
        return Observation(done=False, reward=0.0)

    @property
    def state(self):
        return self._state


# =============================================================================
# Test Fixtures - Environment Code Mode
# =============================================================================


@pytest.fixture
def mcp_server_with_tools():
    """Create a real FastMCP server with tools for testing."""
    mcp = FastMCP("test-code-mode")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @mcp.tool()
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    return mcp


# =============================================================================
# Code Mode Capability Tests
# =============================================================================


class TestCodeModeCapability:
    """Tests for code mode capability detection."""

    def test_environment_has_code_mode_capability(self, mcp_server_with_tools):
        """Test environment can report code mode support."""
        env = _TestMCPEnv(mcp_server_with_tools)

        assert hasattr(env, "supports_code_mode")
        assert env.supports_code_mode is True


# =============================================================================
# Code Mode Tests (with FastMCP Server)
# =============================================================================


class TestCodeModeWithFastMCP:
    """Tests for code mode with real FastMCP servers."""

    def test_get_callables_returns_tool_functions(self, mcp_server_with_tools):
        """Test get_callables() extracts functions from FastMCP server."""
        env = _TestMCPEnv(mcp_server_with_tools)

        callables = env.get_callables()

        assert "add" in callables
        assert callable(callables["add"])
        assert "multiply" in callables
        assert callable(callables["multiply"])

    def test_callables_work_directly(self, mcp_server_with_tools):
        """Test callables from get_callables() can be called directly."""
        env = _TestMCPEnv(mcp_server_with_tools)

        callables = env.get_callables()
        result = callables["add"](a=5, b=3)

        assert result == 8

    def test_code_mode_executes_python_directly(self, mcp_server_with_tools):
        """Test code mode executes Python code with tools as direct callables."""
        env = _TestMCPEnv(mcp_server_with_tools)
        env.reset()

        code = """
result = add(a=5, b=3)
"""

        obs = env.execute_code(code)

        assert isinstance(obs, Observation)
        assert obs.metadata.get("result") == 8

    def test_code_mode_multiple_tool_calls_in_one_step(self, mcp_server_with_tools):
        """Test code mode allows multiple tool calls in a single step."""
        env = _TestMCPEnv(mcp_server_with_tools)
        env.reset()

        code = """
x = add(a=2, b=3)
y = multiply(x=x, y=4)
result = y
"""

        obs = env.execute_code(code)

        # (2 + 3) * 4 = 20
        assert obs.metadata.get("result") == 20

    def test_code_mode_with_complex_python_logic(self, mcp_server_with_tools):
        """Test code mode supports arbitrary Python logic around tool calls."""
        env = _TestMCPEnv(mcp_server_with_tools)
        env.reset()

        code = """
numbers = [1, 2, 3, 4, 5]
total = 0
for n in numbers:
    total = add(a=total, b=n)
result = total
"""

        obs = env.execute_code(code)

        assert obs.metadata.get("result") == 15  # Sum of 1+2+3+4+5


# =============================================================================
# Code Mode with Mode-Aware Tools
# =============================================================================


class TestCodeModeWithModeAwareTools:
    """Tests for code mode integration with mode-aware tool registration."""

    def test_get_callables_includes_mode_specific_tools(self):
        """Test get_callables() returns mode-specific tools for current mode."""
        mcp = FastMCP("mode-test")

        class ModeEnv(_TestMCPEnv):
            def __init__(self):
                super().__init__(mcp)
                self._mode = "simulation"

                @self.tool(mode="simulation")
                def sim_tool(x: int) -> int:
                    return x * 10

                @self.tool(mode="production")
                def prod_tool(x: int) -> int:
                    return x * 100

        env = ModeEnv()
        callables = env.get_callables()

        # In simulation mode, should have sim_tool but not prod_tool
        assert "sim_tool" in callables
        assert "prod_tool" not in callables
        assert callables["sim_tool"](x=5) == 50

    def test_get_callables_switches_with_mode(self):
        """Test get_callables() returns different tools when mode changes."""
        mcp = FastMCP("mode-switch-test")

        class ModeEnv(_TestMCPEnv):
            def __init__(self):
                super().__init__(mcp)
                self._mode = "simulation"

                @self.tool(mode="simulation")
                def lookup(query: str) -> str:
                    return f"sim:{query}"

                @self.tool(mode="production")
                def lookup(query: str) -> str:  # noqa: F811
                    return f"prod:{query}"

        env = ModeEnv()

        # In simulation mode
        callables_sim = env.get_callables()
        assert callables_sim["lookup"](query="test") == "sim:test"

        # Switch to production mode
        env._mode = "production"
        callables_prod = env.get_callables()
        assert callables_prod["lookup"](query="test") == "prod:test"

    def test_execute_code_uses_mode_specific_tools(self):
        """Test execute_code() uses the correct mode-specific tools."""
        mcp = FastMCP("code-mode-test")

        class ModeEnv(_TestMCPEnv):
            def __init__(self):
                super().__init__(mcp)
                self._mode = "simulation"

                @self.tool(mode="simulation")
                def compute(x: int) -> int:
                    return x + 1

                @self.tool(mode="production")
                def compute(x: int) -> int:  # noqa: F811
                    return x + 1000

        env = ModeEnv()

        # In simulation mode
        obs = env.execute_code("result = compute(x=5)")
        assert obs.metadata.get("result") == 6

        # Switch to production mode
        env._mode = "production"
        obs = env.execute_code("result = compute(x=5)")
        assert obs.metadata.get("result") == 1005


# =============================================================================
# Tool-Calling Mode Tests (Backwards Compatibility)
# =============================================================================


class TestToolCallingMode:
    """Tests that tool-calling mode still works (backwards compatibility)."""

    def test_list_tools_still_works(self, mcp_server_with_tools):
        """Test ListToolsAction still works in tool-calling mode."""
        env = _TestMCPEnv(mcp_server_with_tools)

        action = ListToolsAction()
        obs = env.step(action)

        assert isinstance(obs, ListToolsObservation)
        assert len(obs.tools) > 0

    def test_code_mode_preserves_tool_schemas_for_discovery(
        self, mcp_server_with_tools
    ):
        """Test code mode doesn't break tool discovery (list_tools still works)."""
        env = _TestMCPEnv(mcp_server_with_tools)

        # Tool discovery should still work via step()
        obs = env.step(ListToolsAction())

        assert isinstance(obs, ListToolsObservation)
        assert len(obs.tools) > 0

        # And also via get_callables() for code mode
        callables = env.get_callables()
        assert len(callables) == len(obs.tools)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCodeModeErrorHandling:
    """Tests for error handling in code mode."""

    def test_code_mode_handles_syntax_errors(self, mcp_server_with_tools):
        """Test code mode returns proper error for Python syntax errors."""
        env = _TestMCPEnv(mcp_server_with_tools)
        env.reset()

        code = """
result = add(a=5, b=  # Syntax error
"""

        obs = env.execute_code(code)

        assert obs.metadata.get("error") is not None
        assert "syntax" in obs.metadata["error"].lower()

    def test_code_mode_handles_runtime_errors(self, mcp_server_with_tools):
        """Test code mode returns proper error for runtime errors."""
        env = _TestMCPEnv(mcp_server_with_tools)
        env.reset()

        code = """
result = add(a=5, b="not a number")  # Type error
"""

        obs = env.execute_code(code)

        assert obs.metadata.get("error") is not None

    def test_code_mode_handles_missing_tool(self, mcp_server_with_tools):
        """Test code mode returns proper error when calling non-existent tool."""
        env = _TestMCPEnv(mcp_server_with_tools)
        env.reset()

        code = """
result = nonexistent_tool(x=1)
"""

        obs = env.execute_code(code)

        assert obs.metadata.get("error") is not None
        assert "nonexistent_tool" in obs.metadata["error"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestCodeModeIntegration:
    """Integration tests for code mode with real MCP servers."""

    def test_echo_env_in_code_mode(self):
        """Test EchoEnvironment supports code mode."""
        from echo_env.server.echo_environment import EchoEnvironment

        env = EchoEnvironment()
        env.reset()

        code = """
msg = echo_message(message="Hello from code mode!")
result = msg
"""

        obs = env.execute_code(code)

        assert "Hello from code mode!" in str(obs.metadata.get("result"))
