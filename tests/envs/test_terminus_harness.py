# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Terminus harness-oriented session adapter."""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation
from openenv.core.harness import (
    HarnessRunLimits,
    MCPHarnessAdapter,
    ModelStepResult,
    ResourceSessionFactory,
    build_harness_rollout_func,
)
from openenv.core.llm_client import LLMResponse, ToolCall
from terminus_env.harness import TerminusSessionFactory, build_terminal_tool_call
from terminus_env.models import CommandResult, TerminusState


class FakeTerminusClient:
    """Small Terminus-like client used for harness tests."""

    def __init__(self):
        self.closed = False
        self.reset_kwargs: dict[str, Any] = {}
        self.step_actions: list[CallToolAction] = []
        self._state = TerminusState(
            episode_id="terminus-episode",
            sandbox_id="fake-sandbox",
        )

    def reset(self, **kwargs: Any) -> StepResult[Observation]:
        self.reset_kwargs = dict(kwargs)
        self._state.verify_commands = list(kwargs.get("verify", []) or [])
        self._state.setup_results = [
            CommandResult(command=command, output="setup", success=True)
            for command in kwargs.get("setup", []) or []
        ]
        return StepResult(
            observation=Observation(
                done=False,
                reward=None,
                metadata={
                    "message": "Terminus environment ready.",
                    "verify_commands": list(self._state.verify_commands),
                },
            ),
            reward=None,
            done=False,
        )

    def step(
        self,
        action: CallToolAction,
    ) -> StepResult[CallToolObservation]:
        self.step_actions.append(action)
        self._state.step_count += 1
        arguments = action.arguments
        if arguments.get("final_answer"):
            self._state.submitted_answer = str(arguments["final_answer"])
            self._state.last_reward = 1.0
            output = "Verification: 1/1 passed; reward=1.0"
            return StepResult(
                observation=CallToolObservation(
                    tool_name="terminal",
                    result={"content": [{"type": "text", "text": output}]},
                    done=True,
                    reward=1.0,
                ),
                reward=1.0,
                done=True,
            )

        command = str(arguments.get("command", ""))
        self._state.commands.append(
            CommandResult(command=command, output=f"shell: {command}", success=True)
        )
        return StepResult(
            observation=CallToolObservation(
                tool_name="terminal",
                result={"content": [{"type": "text", "text": f"shell: {command}"}]},
                done=False,
                reward=0.0,
            ),
            reward=0.0,
            done=False,
        )

    def state(self) -> TerminusState:
        return self._state

    def close(self) -> None:
        self.closed = True


def test_terminus_session_factory_exposes_terminal_tool():
    client = FakeTerminusClient()
    factory = TerminusSessionFactory(
        client_factory=lambda: client,
        default_setup=["echo setup"],
        default_verify=["test -f answer.txt"],
    )
    assert isinstance(factory, ResourceSessionFactory)

    session = factory.create(task="Write answer.txt")

    assert [tool.name for tool in session.list_tools()] == ["terminal"]
    assert client.reset_kwargs["setup"] == ["echo setup"]
    assert client.reset_kwargs["verify"] == ["test -f answer.txt"]
    messages = session.initial_messages()
    assert "Write answer.txt" in messages[0]["content"]
    assert "Verification commands will run after final_answer" in messages[0]["content"]

    session.close()
    assert client.closed is True


def test_terminus_tool_calls_forward_environment_rewards():
    client = FakeTerminusClient()
    factory = TerminusSessionFactory(client_factory=lambda: client)
    session = factory.create(
        task={
            "instruction": "Create answer.txt",
            "verify": ["test -f answer.txt"],
        }
    )

    command_result = session.call_tool("terminal", {"command": "pwd"})
    final_result = session.call_tool("terminal", {"final_answer": "done"})
    verify_result = session.verify(transcript=[{"role": "assistant", "content": ""}])

    assert command_result.done is False
    assert command_result.metadata["reward"] == 0.0
    assert final_result.done is True
    assert final_result.metadata["reward"] == 1.0
    assert verify_result.env_reward == 1.0
    assert verify_result.done is True
    assert client.step_actions[0].tool_name == "terminal"
    assert client.step_actions[0].arguments == {"command": "pwd"}
    session.close()


def test_terminus_terminal_json_parser():
    command = build_terminal_tool_call('{"command": "pytest -q"}')
    final_answer = build_terminal_tool_call('```json\n{"final_answer": "done"}\n```')
    pi_command = build_terminal_tool_call('terminal(command="echo terminus > answer.txt")')
    pi_final_answer = build_terminal_tool_call("terminal(final_answer='done')")
    tool_call = build_terminal_tool_call(
        '<tool_call>{"name": "terminal", "arguments": "{\\"final_answer\\": '
        '\\"done\\"}"}</tool_call>'
    )
    mixed = build_terminal_tool_call(
        '{"command": "printf terminus > answer.txt", "final_answer": "done"}'
    )

    assert command.name == "terminal"
    assert command.args == {"command": "pytest -q"}
    assert final_answer.name == "terminal"
    assert final_answer.args == {"final_answer": "done"}
    assert pi_command.name == "terminal"
    assert pi_command.args == {"command": "echo terminus > answer.txt"}
    assert pi_final_answer.name == "terminal"
    assert pi_final_answer.args == {"final_answer": "done"}
    assert tool_call.name == "terminal"
    assert tool_call.args == {"final_answer": "done"}
    assert mixed.name == "terminal"
    assert mixed.args == {"command": "printf terminus > answer.txt"}


def test_terminus_session_factory_works_with_generic_rollout_helper():
    factory = TerminusSessionFactory(
        client_factory=FakeTerminusClient,
        default_verify=["test -f answer.txt"],
    )
    adapter = MCPHarnessAdapter()

    def model_step_builder(trainer, session):
        tool_call = ToolCall(
            id="terminal-1",
            name="terminal",
            args={"final_answer": "done"},
        )
        return lambda messages, tools, sampling: ModelStepResult(
            response=LLMResponse(content="done", tool_calls=[tool_call]),
            prompt_ids=[3, 4],
            completion_ids=[5, 6],
            logprobs=[-0.3, -0.4],
        )

    rollout_func = build_harness_rollout_func(
        session_factory=factory,
        harness_adapter=adapter,
        model_step_builder=model_step_builder,
        limits=HarnessRunLimits(max_turns=3),
    )

    result = rollout_func(["Create answer.txt"], trainer=object())

    assert result["prompt_ids"] == [[3, 4]]
    assert result["completion_ids"] == [[5, 6]]
    assert result["logprobs"] == [[-0.3, -0.4]]
    assert result["env_reward"] == [1.0]
