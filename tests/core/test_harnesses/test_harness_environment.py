# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for HarnessEnvironment (RFC 005)."""

import time
from typing import AsyncIterator, List

import pytest
from openenv.core.env_server.types import Observation
from openenv.core.harnesses import (
    HarnessAction,
    HarnessAdapter,
    HarnessConfig,
    HarnessEvent,
    HarnessEventType,
    HarnessResponse,
)


# =============================================================================
# Test Fixtures: Mock Adapter
# =============================================================================


class MockHarnessAdapter(HarnessAdapter):
    """Mock adapter for testing HarnessEnvironment."""

    def __init__(self, config: HarnessConfig):
        super().__init__(config)
        self._alive = False
        self._started_count = 0
        self._stopped_count = 0
        self._injected_tools: List = []
        self._messages: List[str] = []
        self._responses: List[HarnessResponse] = []
        self._response_index = 0

    def set_responses(self, responses: List[HarnessResponse]) -> None:
        """Pre-configure responses for send_message calls."""
        self._responses = responses
        self._response_index = 0

    async def start(self, working_directory: str) -> None:
        self._alive = True
        self._started_count += 1
        self._working_directory = working_directory

    async def stop(self) -> None:
        self._alive = False
        self._stopped_count += 1

    async def inject_tools(self, tools: List) -> None:
        self._injected_tools = list(tools)

    async def send_message(self, message: str) -> HarnessResponse:
        self._messages.append(message)
        if self._response_index < len(self._responses):
            resp = self._responses[self._response_index]
            self._response_index += 1
            return resp
        return HarnessResponse(response=f"Echo: {message}", done=False)

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        resp = await self.send_message(message)
        for event in resp.events:
            yield event
        yield HarnessEvent(
            type=HarnessEventType.TURN_COMPLETE,
            timestamp=time.time(),
            data={"response": resp.response},
        )

    async def is_alive(self) -> bool:
        return self._alive


@pytest.fixture
def mock_config():
    return HarnessConfig(name="mock", command=["mock", "run"])


@pytest.fixture
def mock_adapter(mock_config):
    return MockHarnessAdapter(config=mock_config)


@pytest.fixture
def harness_env(mock_adapter):
    from openenv.core.harnesses.environment import HarnessEnvironment

    return HarnessEnvironment(adapter=mock_adapter)


# =============================================================================
# Tests
# =============================================================================


class TestHarnessEnvironmentImport:
    """Test that HarnessEnvironment is importable."""

    def test_import_from_module(self):
        from openenv.core.harnesses.environment import HarnessEnvironment

        assert HarnessEnvironment is not None

    def test_import_from_package(self):
        from openenv.core.harnesses import HarnessEnvironment

        assert HarnessEnvironment is not None

    def test_inherits_from_environment(self):
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.harnesses import HarnessEnvironment

        assert issubclass(HarnessEnvironment, Environment)


class TestHarnessEnvironmentReset:
    """Test reset behavior."""

    def test_reset_returns_observation(self, harness_env):
        obs = harness_env.reset()
        assert isinstance(obs, Observation)
        assert obs.done is False
        assert obs.reward == 0.0

    def test_reset_starts_adapter(self, harness_env, mock_adapter):
        harness_env.reset()
        assert mock_adapter._alive is True
        assert mock_adapter._started_count == 1

    def test_reset_stops_existing_session(self, harness_env, mock_adapter):
        harness_env.reset()
        assert mock_adapter._started_count == 1
        harness_env.reset()
        assert mock_adapter._stopped_count == 1
        assert mock_adapter._started_count == 2

    def test_reset_assigns_episode_id(self, harness_env):
        harness_env.reset(episode_id="ep-42")
        assert harness_env.state.episode_id == "ep-42"

    def test_reset_generates_episode_id_if_none(self, harness_env):
        harness_env.reset()
        assert harness_env.state.episode_id is not None
        assert len(harness_env.state.episode_id) > 0

    def test_reset_clears_step_count(self, harness_env, mock_adapter):
        mock_adapter.set_responses([HarnessResponse(response="ok", done=False)])
        harness_env.reset()
        harness_env.step(HarnessAction(message="test"))
        assert harness_env.state.step_count == 1
        harness_env.reset()
        assert harness_env.state.step_count == 0

    def test_reset_clears_trajectory(self, harness_env, mock_adapter):
        events = [
            HarnessEvent(
                type=HarnessEventType.TOOL_CALL,
                timestamp=1.0,
                data={"tool_name": "bash"},
            )
        ]
        mock_adapter.set_responses(
            [HarnessResponse(response="ok", events=events, done=False)]
        )
        harness_env.reset()
        harness_env.step(HarnessAction(message="test"))
        assert len(harness_env.trajectory) > 0
        harness_env.reset()
        assert len(harness_env.trajectory) == 0


class TestHarnessEnvironmentStep:
    """Test step behavior (multi-turn conversations)."""

    def test_step_sends_message_to_adapter(self, harness_env, mock_adapter):
        harness_env.reset()
        harness_env.step(HarnessAction(message="Fix the bug"))
        assert "Fix the bug" in mock_adapter._messages

    def test_step_returns_observation(self, harness_env, mock_adapter):
        mock_adapter.set_responses([HarnessResponse(response="I fixed it.", done=True)])
        harness_env.reset()
        obs = harness_env.step(HarnessAction(message="Fix the bug"))
        assert isinstance(obs, Observation)
        assert obs.metadata["response"] == "I fixed it."
        assert obs.done is True

    def test_step_increments_step_count(self, harness_env, mock_adapter):
        mock_adapter.set_responses(
            [
                HarnessResponse(response="r1", done=False),
                HarnessResponse(response="r2", done=False),
            ]
        )
        harness_env.reset()
        harness_env.step(HarnessAction(message="turn 1"))
        assert harness_env.state.step_count == 1
        harness_env.step(HarnessAction(message="turn 2"))
        assert harness_env.state.step_count == 2

    def test_step_propagates_done_from_harness(self, harness_env, mock_adapter):
        mock_adapter.set_responses(
            [
                HarnessResponse(response="working...", done=False),
                HarnessResponse(response="done!", done=True),
            ]
        )
        harness_env.reset()
        obs1 = harness_env.step(HarnessAction(message="start"))
        assert obs1.done is False
        obs2 = harness_env.step(HarnessAction(message="continue"))
        assert obs2.done is True

    def test_step_includes_turn_events(self, harness_env, mock_adapter):
        events = [
            HarnessEvent(
                type=HarnessEventType.TOOL_CALL,
                timestamp=1.0,
                data={"tool_name": "read_file"},
            ),
            HarnessEvent(
                type=HarnessEventType.TOOL_RESULT,
                timestamp=2.0,
                data={"tool_name": "read_file", "result": "contents"},
            ),
        ]
        mock_adapter.set_responses(
            [HarnessResponse(response="done", events=events, done=True)]
        )
        harness_env.reset()
        obs = harness_env.step(HarnessAction(message="read the file"))
        assert "turn_events" in obs.metadata
        assert len(obs.metadata["turn_events"]) == 2

    def test_step_includes_turn_number(self, harness_env, mock_adapter):
        mock_adapter.set_responses(
            [
                HarnessResponse(response="r1", done=False),
                HarnessResponse(response="r2", done=False),
            ]
        )
        harness_env.reset()
        obs1 = harness_env.step(HarnessAction(message="t1"))
        assert obs1.metadata["turn_number"] == 1
        obs2 = harness_env.step(HarnessAction(message="t2"))
        assert obs2.metadata["turn_number"] == 2

    def test_multi_turn_conversation(self, harness_env, mock_adapter):
        """Test a realistic multi-turn conversation."""
        mock_adapter.set_responses(
            [
                HarnessResponse(
                    response="I see the bug. Let me fix it.",
                    events=[
                        HarnessEvent(
                            type=HarnessEventType.TOOL_CALL,
                            timestamp=1.0,
                            data={"tool_name": "read_file"},
                        ),
                    ],
                    done=False,
                ),
                HarnessResponse(
                    response="Fixed. Tests pass now.",
                    events=[
                        HarnessEvent(
                            type=HarnessEventType.TOOL_CALL,
                            timestamp=2.0,
                            data={"tool_name": "write_file"},
                        ),
                        HarnessEvent(
                            type=HarnessEventType.TOOL_CALL,
                            timestamp=3.0,
                            data={"tool_name": "bash"},
                        ),
                    ],
                    done=True,
                ),
            ]
        )
        harness_env.reset()

        obs1 = harness_env.step(HarnessAction(message="Fix the bug in auth.py"))
        assert obs1.done is False
        assert obs1.metadata["response"] == "I see the bug. Let me fix it."

        obs2 = harness_env.step(HarnessAction(message="Tests still failing on line 42"))
        assert obs2.done is True
        assert obs2.metadata["response"] == "Fixed. Tests pass now."

        # Full trajectory should have events from both turns
        assert len(harness_env.trajectory) == 3


class TestHarnessEnvironmentTrajectory:
    """Test trajectory accumulation across turns."""

    def test_trajectory_empty_after_reset(self, harness_env):
        harness_env.reset()
        assert harness_env.trajectory == []

    def test_trajectory_accumulates_across_turns(self, harness_env, mock_adapter):
        mock_adapter.set_responses(
            [
                HarnessResponse(
                    response="r1",
                    events=[
                        HarnessEvent(
                            type=HarnessEventType.TOOL_CALL,
                            timestamp=1.0,
                            data={"tool_name": "tool_a"},
                        )
                    ],
                    done=False,
                ),
                HarnessResponse(
                    response="r2",
                    events=[
                        HarnessEvent(
                            type=HarnessEventType.TOOL_CALL,
                            timestamp=2.0,
                            data={"tool_name": "tool_b"},
                        )
                    ],
                    done=True,
                ),
            ]
        )
        harness_env.reset()
        harness_env.step(HarnessAction(message="t1"))
        harness_env.step(HarnessAction(message="t2"))

        traj = harness_env.trajectory
        assert len(traj) == 2
        assert traj[0].data["tool_name"] == "tool_a"
        assert traj[1].data["tool_name"] == "tool_b"


class TestHarnessEnvironmentState:
    """Test state property."""

    def test_state_has_episode_id(self, harness_env):
        harness_env.reset(episode_id="ep-1")
        assert harness_env.state.episode_id == "ep-1"

    def test_state_has_step_count(self, harness_env, mock_adapter):
        mock_adapter.set_responses([HarnessResponse(response="ok", done=False)])
        harness_env.reset()
        assert harness_env.state.step_count == 0
        harness_env.step(HarnessAction(message="test"))
        assert harness_env.state.step_count == 1


class TestHarnessEnvironmentClose:
    """Test cleanup behavior."""

    def test_close_stops_adapter(self, harness_env, mock_adapter):
        harness_env.reset()
        assert mock_adapter._alive is True
        harness_env.close()
        assert mock_adapter._alive is False

    def test_close_when_not_started(self, harness_env, mock_adapter):
        # Should not raise
        harness_env.close()
        assert mock_adapter._stopped_count == 0


class TestHarnessEnvironmentWithMCPTools:
    """Test HarnessEnvironment with additional MCP tools."""

    def test_creation_with_mcp_server(self, mock_adapter):
        from fastmcp import FastMCP
        from openenv.core.harnesses import HarnessEnvironment

        mcp = FastMCP("test-tools")

        @mcp.tool
        def my_tool(arg: str) -> str:
            return arg

        env = HarnessEnvironment(adapter=mock_adapter, mcp=mcp)
        assert env is not None

    def test_mcp_tools_injected_on_reset(self, mock_adapter):
        from fastmcp import FastMCP
        from openenv.core.harnesses import HarnessEnvironment

        mcp = FastMCP("test-tools")

        @mcp.tool
        def my_tool(arg: str) -> str:
            return arg

        env = HarnessEnvironment(adapter=mock_adapter, mcp=mcp)
        env.reset()
        # Adapter should have had tools injected
        assert mock_adapter._injected_tools is not None

    def test_creation_without_mcp_server(self, mock_adapter):
        from openenv.core.harnesses import HarnessEnvironment

        env = HarnessEnvironment(adapter=mock_adapter)
        env.reset()
        # No tools should be injected
        assert mock_adapter._injected_tools == []
