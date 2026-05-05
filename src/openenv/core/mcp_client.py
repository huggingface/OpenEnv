# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP Client classes for tool-calling environments.

This module provides async client classes for interacting with MCP-enabled environments:
- MCPClientBase: Base class with shared tool discovery
- MCPToolClient: Client for tool-calling style (one tool per step)

These clients abstract away the MCP protocol details, providing a clean interface
for listing and calling tools on remote environments. All clients are async by default.

Architecture Overview::

    ┌─────────────────────────────────────────────────────────┐
    │                    HTTPEnvServer                        │
    ├─────────────────────────────────────────────────────────┤
    │  Simulation Mode (default):                             │
    │    /ws    → OpenEnv protocol (reset/step/state)         │
    │    /mcp   → MCP JSON-RPC (tools/list, tools/call)       │
    │    /reset, /step, /state → HTTP endpoints               │
    ├─────────────────────────────────────────────────────────┤
    │  Production Mode (use_production_mode=True):                     │
    │    /mcp   → MCP JSON-RPC (tools/list, tools/call)       │
    │    Bypasses step() for direct tool access               │
    └─────────────────────────────────────────────────────────┘

    Client Usage:
      MCPToolClient (default)     → /ws (step-based, with rewards)
      MCPToolClient (production)    → /mcp (direct tool access, no rewards)

Example (async):
    >>> from openenv.core.mcp_client import MCPToolClient
    >>>
    >>> async with MCPToolClient(base_url="http://localhost:8000") as env:
    ...     # Discover available tools
    ...     tools = await env.list_tools()
    ...     print([t.name for t in tools])
    ...
    ...     # Call a tool
    ...     result = await env.call_tool("echo_message", message="Hello!")
    ...     print(result)

Example (sync wrapper):
    >>> env = MCPToolClient(base_url="http://localhost:8000").sync()
    >>> with env:
    ...     tools = env.list_tools()
    ...     result = env.call_tool("echo_message", message="Hello!")
"""

import asyncio
from typing import Any, Dict, List, Optional

from .client_types import StepResult
from .env_client import EnvClient
from .env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
    ToolError,
)
from .env_server.types import Observation, State


class MCPClientBase(EnvClient[Any, Observation, State]):
    """
    Base class for MCP clients with tool discovery.

    This class provides the common `list_tools()` method for discovering
    available tools from an MCP-enabled environment. Subclasses implement
    specific interaction patterns (tool-calling or CodeAct).

    Attributes:
        _tools_cache: Cached list of tools (populated on first `list_tools()` call)
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        provider: Optional[Any] = None,
        mode: Optional[str] = None,
    ):
        """
        Initialize MCP client.

        Args:
            base_url: Base URL of the environment server (http:// or ws://).
            connect_timeout_s: Timeout for establishing WebSocket connection.
            message_timeout_s: Timeout for receiving responses to messages.
            provider: Optional container/runtime provider for lifecycle management.
            mode: Communication mode. Must be 'production' for MCP clients. Defaults to 'production'.
        """
        # MCPClientBase defaults to production mode, but allow override for validation
        if mode is None:
            mode = "production"

        # Validate that mode is production
        mode_lower = mode.lower()
        if mode_lower != "production":
            raise ValueError(
                f"MCPToolClient only supports 'production' mode, got '{mode}'. "
                f"Use GenericEnvClient for simulation mode."
            )

        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            provider=provider,
            mode=mode,
        )
        self._tools_cache: Optional[List[Tool]] = None
        self.use_production_mode = False
        self._production_session_id: Optional[str] = None
        self._production_session_lock = asyncio.Lock()
        self._jsonrpc_request_id = 0
        self._http_client: Optional[Any] = None  # lazily-created httpx.AsyncClient

    def _next_request_id(self) -> int:
        """Generate a monotonically increasing JSON-RPC request id."""
        self._jsonrpc_request_id += 1
        return self._jsonrpc_request_id

    def _production_mcp_url(self) -> str:
        """Build HTTP MCP endpoint URL from the client's websocket URL."""
        url = self._ws_url.replace("ws://", "http://").replace("wss://", "https://")
        if url.endswith("/ws"):
            url = url[: -len("/ws")]
        return url.rstrip("/") + "/mcp"

    async def _get_http_client(self) -> Any:
        """Return a shared httpx.AsyncClient, creating one lazily."""
        if self._http_client is None:
            import httpx

            self._http_client = httpx.AsyncClient()
        return self._http_client

    async def _production_mcp_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request to HTTP /mcp and return parsed JSON response."""
        client = await self._get_http_client()
        response = await client.post(
            self._production_mcp_url(),
            json={
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
                "id": self._next_request_id(),
            },
            timeout=self._message_timeout,
        )
        response.raise_for_status()
        return response.json()

    async def _ensure_production_session(self) -> str:
        """Create and cache a persistent HTTP MCP session id if needed."""
        async with self._production_session_lock:
            if self._production_session_id is not None:
                return self._production_session_id

            data = await self._production_mcp_request("openenv/session/create")
            if "error" in data:
                message = data.get("error", {}).get("message", "unknown error")
                raise RuntimeError(f"Failed to create MCP session: {message}")

            session_id = data.get("result", {}).get("session_id")
            if not session_id:
                raise RuntimeError("Failed to create MCP session: missing session_id")

            self._production_session_id = session_id
            return session_id

    async def list_tools(self, use_cache: bool = True) -> List[Tool]:
        """
        Discover available tools from the environment.

        Args:
            use_cache: If True, return cached tools if available.
                      Set to False to force a fresh request.

        Returns:
            List of Tool objects with name, description, and input_schema.

        Example:
            >>> tools = await env.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        # Use production mode HTTP endpoint if enabled.
        # Some tests instantiate with __new__ and skip __init__, so default missing flag to False.
        if getattr(self, "use_production_mode", False):
            try:
                session_id = await self._ensure_production_session()
                data = await self._production_mcp_request(
                    "tools/list",
                    {"session_id": session_id},
                )
                if "error" in data:
                    message = data.get("error", {}).get("message", "unknown error")
                    raise RuntimeError(f"list_tools failed: {message}")
                if "result" in data and "tools" in data["result"]:
                    tools = [
                        Tool(
                            name=t.get("name", ""),
                            description=t.get("description", ""),
                            input_schema=t.get(
                                "input_schema", t.get("inputSchema", {})
                            ),
                        )
                        for t in data["result"]["tools"]
                    ]
                    self._tools_cache = tools
                    return tools
            except Exception:
                # If HTTP request fails, return empty list
                pass
            return []

        result = await self.step(ListToolsAction())
        if isinstance(result.observation, ListToolsObservation):
            self._tools_cache = result.observation.tools
            return self._tools_cache

        # Unexpected observation type; keep API stable with an empty tool list.
        self._tools_cache = []
        return self._tools_cache

    def _step_payload(self, action: Any) -> Dict[str, Any]:
        """Convert an Action object to the JSON data expected by the env server."""
        if isinstance(action, ListToolsAction):
            return {"type": "list_tools"}
        elif isinstance(action, CallToolAction):
            return {
                "type": "call_tool",
                "tool_name": action.tool_name,
                "arguments": action.arguments,
            }
        else:
            # For unknown actions, try to serialize as dict
            if hasattr(action, "model_dump"):
                return action.model_dump()
            return {"action": str(action)}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Observation]:
        """Convert a JSON response from the env server to StepResult[Observation]."""
        obs_data = payload.get("observation", {})

        # Check if this is a ListToolsObservation
        if "tools" in obs_data:
            tools = [
                Tool(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("input_schema", t.get("inputSchema", {})),
                )
                for t in obs_data.get("tools", [])
            ]
            observation = ListToolsObservation(
                tools=tools,
                done=payload.get("done", False),
                reward=payload.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )
        # Check if this is a CallToolObservation
        elif "tool_name" in obs_data:
            error = None
            if obs_data.get("error"):
                error = ToolError(**obs_data["error"])

            observation = CallToolObservation(
                tool_name=obs_data.get("tool_name", ""),
                result=obs_data.get("result"),
                error=error,
                done=payload.get("done", False),
                reward=payload.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )
        else:
            # Generic observation
            observation = Observation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Convert a JSON response from the state endpoint to a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    async def close(self) -> None:
        """
        Close client resources.

        In production MCP mode, this also closes the server-side persistent
        MCP session (best effort) before closing websocket/provider resources.
        """
        if self._production_session_id is not None:
            try:
                await self._production_mcp_request(
                    "openenv/session/close",
                    {"session_id": self._production_session_id},
                )
            except Exception:
                # Best effort cleanup - do not mask normal close behavior
                pass
            finally:
                self._production_session_id = None

        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            finally:
                self._http_client = None

        await super().close()


class MCPToolClient(MCPClientBase):
    """
    Async client for tool-calling style MCP interactions.

    Each step invokes a single tool. Use this for traditional function-calling
    agent patterns where the agent decides which tool to call next.

    This client provides convenience methods for tool discovery and invocation:
    - `list_tools()`: Get all available tools with their schemas
    - `call_tool(name, **kwargs)`: Invoke a tool by name with arguments

    Example (async):
        >>> async with MCPToolClient(base_url="http://localhost:8000") as env:
        ...     # Reset the environment
        ...     await env.reset()
        ...
        ...     # Discover available tools
        ...     tools = await env.list_tools()
        ...     print([t.name for t in tools])  # ['echo_message', 'echo_with_length']
        ...
        ...     # Call a tool directly
        ...     result = await env.call_tool("echo_message", message="Hello!")
        ...     print(result)  # "Hello!"
        ...
        ...     # Or use the full action interface
        ...     from openenv.core.env_server.mcp_types import CallToolAction
        ...     step_result = await env.step(CallToolAction(
        ...         tool_name="echo_with_length",
        ...         arguments={"message": "Test"}
        ...     ))
        ...     print(step_result.observation.result)

    Example (sync wrapper):
        >>> env = MCPToolClient(base_url="http://localhost:8000").sync()
        >>> with env:
        ...     tools = env.list_tools()
        ...     result = env.call_tool("echo_message", message="Hello!")
    """

    async def call_tool(self, name: str, **kwargs: Any) -> Any:
        """
        Call a tool by name.

        This is a convenience method that creates a CallToolAction, executes it,
        and returns the result directly. For more control, use `step()` with
        a CallToolAction directly.

        Args:
            name: Name of the tool to invoke (must match a tool from `list_tools()`).
            **kwargs: Arguments to pass to the tool. Must match the tool's input_schema.

        Returns:
            The tool's result. The type depends on the tool being called.

        Raises:
            RuntimeError: If the server returns an error response.

        Example:
            >>> result = await env.call_tool("add", a=5, b=3)
            >>> print(result)  # 8
            >>>
            >>> result = await env.call_tool("greet", name="Claude")
            >>> print(result)  # "Hello, Claude!"
        """
        if getattr(self, "use_production_mode", False):
            session_id = await self._ensure_production_session()
            data = await self._production_mcp_request(
                "tools/call",
                {
                    "name": name,
                    "arguments": kwargs,
                    "session_id": session_id,
                },
            )

            if "error" in data:
                message = data.get("error", {}).get("message", "unknown error")
                raise RuntimeError(f"Tool '{name}' failed: {message}")

            result = data.get("result")
            if isinstance(result, dict) and "data" in result:
                return result["data"]
            return result

        action = CallToolAction(tool_name=name, arguments=kwargs)
        result = await self.step(action)
        obs = result.observation

        # Check for transport/framework errors
        if isinstance(obs, CallToolObservation) and obs.error is not None:
            raise RuntimeError(
                f"Tool '{name}' failed: {obs.error.message} "
                f"(type: {obs.error.error_type.value})"
            )

        # Return the result
        if isinstance(obs, CallToolObservation):
            result = obs.result
            # Handle FastMCP CallToolResult objects
            # - As object: has .data attribute
            # - As dict (from JSON): has "data" key
            if hasattr(result, "data"):
                return result.data
            if isinstance(result, dict) and "data" in result:
                return result["data"]
            return result

        # Fallback for unexpected observation types
        return obs

    async def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a specific tool by name.

        Args:
            name: Name of the tool to find.

        Returns:
            The Tool object if found, None otherwise.

        Example:
            >>> tool = await env.get_tool("echo_message")
            >>> if tool:
            ...     print(tool.description)
            ...     print(tool.input_schema)
        """
        tools = await self.list_tools()
        for tool in tools:
            if tool.name == name:
                return tool
        return None

    async def has_tool(self, name: str) -> bool:
        """
        Check if a tool exists.

        Args:
            name: Name of the tool to check.

        Returns:
            True if the tool exists, False otherwise.
        """
        return await self.get_tool(name) is not None
