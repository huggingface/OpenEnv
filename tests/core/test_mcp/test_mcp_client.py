# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for MCPToolClient.

These tests verify the MCPToolClient class functionality including:
1. Tool discovery via list_tools()
2. Tool invocation via call_tool()
3. Helper methods get_tool() and has_tool()
4. Error handling for tool failures
"""

from unittest.mock import AsyncMock

import pytest
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
    ToolError,
    ToolErrorType,
)
from openenv.core.mcp_client import MCPClientBase, MCPToolClient


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    return [
        Tool(
            name="add",
            description="Add two numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="greet",
            description="Greet a person",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        ),
    ]


# =============================================================================
# MCPClientBase Tests
# =============================================================================


class TestMCPClientBase:
    """Tests for MCPClientBase class."""

    def test_step_payload_list_tools(self):
        """Test _step_payload for ListToolsAction."""
        client = MCPClientBase.__new__(MCPClientBase)
        client._ws = None
        client._tools_cache = None

        action = ListToolsAction()
        payload = client._step_payload(action)

        assert payload == {"type": "list_tools"}

    def test_step_payload_call_tool(self):
        """Test _step_payload for CallToolAction."""
        client = MCPClientBase.__new__(MCPClientBase)
        client._ws = None
        client._tools_cache = None

        action = CallToolAction(
            tool_name="add",
            arguments={"a": 5, "b": 3},
        )
        payload = client._step_payload(action)

        assert payload == {
            "type": "call_tool",
            "tool_name": "add",
            "arguments": {"a": 5, "b": 3},
        }

    def test_parse_result_list_tools_observation(self, mock_tools):
        """Test _parse_result for ListToolsObservation."""
        client = MCPClientBase.__new__(MCPClientBase)
        client._ws = None
        client._tools_cache = None

        payload = {
            "observation": {
                "tools": [
                    {
                        "name": "add",
                        "description": "Add two numbers",
                        "input_schema": {"type": "object"},
                    },
                ],
            },
            "done": False,
            "reward": None,
        }

        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert isinstance(result.observation, ListToolsObservation)
        assert len(result.observation.tools) == 1
        assert result.observation.tools[0].name == "add"
        assert result.done is False

    def test_parse_result_list_tools_accepts_input_schema_alias(self):
        """Test _parse_result accepts MCP-style inputSchema tool payloads."""
        client = MCPClientBase.__new__(MCPClientBase)
        client._ws = None
        client._tools_cache = None

        payload = {
            "observation": {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search documents",
                        "inputSchema": {"type": "object"},
                    },
                ],
            },
        }

        result = client._parse_result(payload)

        assert isinstance(result.observation, ListToolsObservation)
        assert result.observation.tools[0].input_schema == {"type": "object"}

    def test_parse_result_call_tool_observation(self):
        """Test _parse_result for CallToolObservation."""
        client = MCPClientBase.__new__(MCPClientBase)
        client._ws = None
        client._tools_cache = None

        payload = {
            "observation": {
                "tool_name": "add",
                "result": 8,
            },
            "done": False,
            "reward": None,
        }

        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert isinstance(result.observation, CallToolObservation)
        assert result.observation.tool_name == "add"
        assert result.observation.result == 8
        assert result.observation.error is None

    def test_parse_result_call_tool_with_error(self):
        """Test _parse_result for CallToolObservation with error."""
        client = MCPClientBase.__new__(MCPClientBase)
        client._ws = None
        client._tools_cache = None

        payload = {
            "observation": {
                "tool_name": "invalid_tool",
                "result": None,
                "error": {
                    "error_type": "tool_not_found",
                    "message": "Tool 'invalid_tool' not found",
                },
            },
            "done": False,
            "reward": None,
        }

        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert isinstance(result.observation, CallToolObservation)
        assert result.observation.tool_name == "invalid_tool"
        assert result.observation.error is not None
        assert result.observation.error.error_type == ToolErrorType.TOOL_NOT_FOUND

    def test_parse_result_generic_observation_preserves_custom_fields(self):
        """Test _parse_result keeps env-specific observation fields on reset/step."""
        client = MCPClientBase.__new__(MCPClientBase)
        client._ws = None
        client._tools_cache = None

        payload = {
            "observation": {
                "problem": "Prove that the sum of two even integers is even.",
                "problem_id": "bootstrap_000001",
                "problem_type": "proof",
                "metadata": {"dataset_source": "bootstrap"},
            },
            "done": False,
            "reward": None,
        }

        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert getattr(result.observation, "problem_id") == "bootstrap_000001"
        assert getattr(result.observation, "problem_type") == "proof"
        assert result.observation.metadata == {"dataset_source": "bootstrap"}


# =============================================================================
# MCPToolClient Tests
# =============================================================================


class TestMCPToolClient:
    """Tests for MCPToolClient class."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_tools):
        """Test call_tool returns result on success."""
        client = MCPToolClient.__new__(MCPToolClient)
        client._ws = None
        client._tools_cache = None

        # Mock the step method (now async)
        mock_obs = CallToolObservation(
            tool_name="add",
            result=8,
            error=None,
            done=False,
        )
        client.step = AsyncMock(return_value=StepResult(observation=mock_obs))

        result = await client.call_tool("add", a=5, b=3)

        assert result == 8
        client.step.assert_called_once()
        call_args = client.step.call_args[0][0]
        assert isinstance(call_args, CallToolAction)
        assert call_args.tool_name == "add"
        assert call_args.arguments == {"a": 5, "b": 3}

    @pytest.mark.asyncio
    async def test_call_tool_raises_on_error(self):
        """Test call_tool raises RuntimeError on tool error."""
        client = MCPToolClient.__new__(MCPToolClient)
        client._ws = None
        client._tools_cache = None

        # Mock the step method to return an error (now async)
        mock_obs = CallToolObservation(
            tool_name="invalid_tool",
            result=None,
            error=ToolError(
                error_type=ToolErrorType.TOOL_NOT_FOUND,
                message="Tool 'invalid_tool' not found",
            ),
            done=False,
        )
        client.step = AsyncMock(return_value=StepResult(observation=mock_obs))

        with pytest.raises(RuntimeError) as exc_info:
            await client.call_tool("invalid_tool")

        assert "invalid_tool" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_list_tools_caching(self, mock_tools):
        """Test list_tools caches results."""
        client = MCPToolClient.__new__(MCPToolClient)
        client._ws = None
        client._tools_cache = None
        client.use_production_mode = False

        # Mock the step method (now async)
        mock_obs = ListToolsObservation(tools=mock_tools, done=False)
        client.step = AsyncMock(return_value=StepResult(observation=mock_obs))

        # First call should invoke step
        tools1 = await client.list_tools()
        assert len(tools1) == 2
        assert client.step.call_count == 1

        # Second call should use cache
        tools2 = await client.list_tools()
        assert len(tools2) == 2
        assert client.step.call_count == 1  # Not called again

        # Force refresh should invoke step again
        tools3 = await client.list_tools(use_cache=False)
        assert len(tools3) == 2
        assert client.step.call_count == 2

    @pytest.mark.asyncio
    async def test_get_tool_found(self, mock_tools):
        """Test get_tool returns tool when found."""
        client = MCPToolClient.__new__(MCPToolClient)
        client._ws = None
        client._tools_cache = mock_tools

        tool = await client.get_tool("add")

        assert tool is not None
        assert tool.name == "add"
        assert tool.description == "Add two numbers"

    @pytest.mark.asyncio
    async def test_get_tool_not_found(self, mock_tools):
        """Test get_tool returns None when not found."""
        client = MCPToolClient.__new__(MCPToolClient)
        client._ws = None
        client._tools_cache = mock_tools

        tool = await client.get_tool("nonexistent")

        assert tool is None

    @pytest.mark.asyncio
    async def test_has_tool_true(self, mock_tools):
        """Test has_tool returns True when tool exists."""
        client = MCPToolClient.__new__(MCPToolClient)
        client._ws = None
        client._tools_cache = mock_tools

        assert await client.has_tool("add") is True
        assert await client.has_tool("greet") is True

    @pytest.mark.asyncio
    async def test_has_tool_false(self, mock_tools):
        """Test has_tool returns False when tool doesn't exist."""
        client = MCPToolClient.__new__(MCPToolClient)
        client._ws = None
        client._tools_cache = mock_tools

        assert await client.has_tool("nonexistent") is False


# =============================================================================
# Integration with EchoEnv Tests
# =============================================================================


class TestEchoEnvAsMCPToolClient:
    """Tests verifying EchoEnv works as an MCPToolClient."""

    def test_echo_env_is_mcp_tool_client(self):
        """Test EchoEnv is a subclass of MCPToolClient."""
        from echo_env import EchoEnv

        assert issubclass(EchoEnv, MCPToolClient)

    def test_echo_env_inherits_mcp_methods(self):
        """Test EchoEnv has all MCPToolClient methods."""
        from echo_env import EchoEnv

        # Check that methods are inherited
        assert hasattr(EchoEnv, "list_tools")
        assert hasattr(EchoEnv, "call_tool")
        assert hasattr(EchoEnv, "get_tool")
        assert hasattr(EchoEnv, "has_tool")
        assert hasattr(EchoEnv, "reset")
        assert hasattr(EchoEnv, "step")


# =============================================================================
# Production Session Consistency Tests
# =============================================================================


class TestMCPProductionSessionConsistency:
    """Tests verifying production MCP session lifecycle and session identity."""

    @pytest.mark.asyncio
    async def test_production_session_persistence_and_reset(self):
        """Test production mode uses a single session_id and resets it on reset()."""
        client = MCPToolClient(base_url="http://localhost:8000", mode="production")
        assert client.use_production_mode is True

        requests_made = []

        async def mock_mcp_request(method, params=None):
            requests_made.append((method, params or {}))
            if method == "openenv/session/create":
                # Return a new session ID each time create is called
                session_num = len([r for r in requests_made if r[0] == "openenv/session/create"])
                return {"result": {"session_id": f"session_{session_num}"}}
            elif method == "tools/list":
                return {"result": {"tools": [{"name": "echo", "description": "Echo", "input_schema": {}}]}}
            elif method == "tools/call":
                return {"result": "echo_result"}
            elif method == "openenv/session/close":
                return {"result": {"closed": True}}
            return {}

        client._production_mcp_request = AsyncMock(side_effect=mock_mcp_request)

        # First session creation on connect or initial request
        session1 = await client._ensure_production_session()
        assert session1 == "session_1"

        # Calling list_tools and call_tool reuses session1
        tools = await client.list_tools()
        assert len(tools) == 1
        assert client._production_session_id == "session_1"

        result = await client.call_tool("echo", message="hello")
        assert result == "echo_result"
        assert client._production_session_id == "session_1"

        # Resetting client closes session_1 and creates session_2
        await client.reset()
        assert client._production_session_id == "session_2"

        # Subsequent step or call_tool uses session_2 consistently
        res = await client.step(CallToolAction(tool_name="echo", arguments={"message": "hello"}))
        assert res.observation.result == "echo_result"
        assert client._production_session_id == "session_2"

        # Closing client closes session_2
        await client.close()
        assert client._production_session_id is None
        close_requests = [r for r in requests_made if r[0] == "openenv/session/close"]
        assert len(close_requests) == 2  # Once for reset(), once for close()
