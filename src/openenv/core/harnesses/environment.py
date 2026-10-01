# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HarnessEnvironment: wraps an external agentic harness (RFC 005).

This module provides HarnessEnvironment, which bridges an external harness
(OpenClaw, Claude Code, etc.) with OpenEnv's Gym-style API.

Each step() is one conversational turn. The harness maintains conversation
context across turns. reset() starts a fresh conversation.
"""

from typing import Any, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.harnesses.adapter import HarnessAdapter
from openenv.core.harnesses.types import HarnessAction, HarnessEvent
from openenv.core.utils import run_async_safely


class HarnessEnvironment(Environment):
    """Environment that wraps an external agentic harness.

    In simulation mode:
    - reset() starts a fresh harness process and conversation
    - step(message) sends one user message, harness does its ReAct loop
      for that turn, and returns the response as an observation
    - Multiple step() calls form a multi-turn conversation
    - The training loop controls episode boundaries
    - done is set when the harness signals task completion

    In production mode:
    - Clients connect directly to the harness
    - OpenEnv handles session management and tool injection
    - No step/reset API (standard production mode behavior)

    Args:
        adapter: HarnessAdapter for the target harness.
        mcp: Optional FastMCP server with additional environment tools.
        rubric: Optional rubric for reward computation.
    """

    def __init__(
        self,
        adapter: HarnessAdapter,
        mcp: Any = None,
        rubric: Any = None,
    ) -> None:
        super().__init__(rubric=rubric)
        self.adapter = adapter
        self._mcp_server = mcp
        self._state = State(episode_id=None, step_count=0)
        self._trajectory: List[HarnessEvent] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset: stop any running harness, start a fresh conversation."""
        # Stop existing session if any
        if run_async_safely(self.adapter.is_alive()):
            run_async_safely(self.adapter.stop())

        # Inject environment MCP tools into harness if we have an MCP server
        if self._mcp_server is not None:
            tools = self._get_mcp_tool_definitions()
            run_async_safely(self.adapter.inject_tools(tools))

        # Start fresh harness process
        run_async_safely(
            self.adapter.start(working_directory=self.adapter.config.working_directory)
        )

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._trajectory = []

        if self.rubric:
            self._reset_rubric()

        return Observation(done=False, reward=0.0, metadata={})

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Send one message to the harness, get one turn's response.

        Each step() is one conversational turn. The harness does its
        internal ReAct loop (LLM calls, tool invocations, etc.) and
        returns when it has a response for the user.
        """
        message = self._extract_message(action)

        # Run one conversational turn (harness does its ReAct loop)
        harness_response = run_async_safely(self.adapter.send_message(message))

        # Accumulate trajectory across turns
        self._trajectory.extend(harness_response.events)
        self._state.step_count += 1

        # Build observation
        obs = Observation(
            done=harness_response.done,
            reward=0.0,
            metadata={
                "response": harness_response.response,
                "turn_events": harness_response.events,
                "turn_number": self._state.step_count,
            },
        )

        # Apply rubric if configured
        if self.rubric:
            obs.reward = self._apply_rubric(action, obs)

        return obs

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    @property
    def trajectory(self) -> List[HarnessEvent]:
        """Full trajectory across all turns in this episode."""
        return self._trajectory

    def close(self) -> None:
        """Clean up harness process."""
        if run_async_safely(self.adapter.is_alive()):
            run_async_safely(self.adapter.stop())

    def _extract_message(self, action: Action) -> str:
        """Extract the message string from an action."""
        if isinstance(action, HarnessAction):
            return action.message
        # Fallback: try to get a message field or convert to string
        if hasattr(action, "message"):
            return action.message
        return str(action)

    def _get_mcp_tool_definitions(self) -> List:
        """Extract tool definitions from the MCP server."""
        if self._mcp_server is None:
            return []
        try:
            from fastmcp import Client

            async def _list_tools():
                async with Client(self._mcp_server) as client:
                    return await client.list_tools()

            return run_async_safely(_list_tools())
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to extract MCP tool definitions: %s", e
            )
            return []
