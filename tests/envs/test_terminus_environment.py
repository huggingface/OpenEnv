# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from terminus_env.models import TerminusState
from terminus_env.server.e2b_sandbox import ShellResult
from terminus_env.server.terminus_env_environment import TerminusEnvironment


class FakeSandbox:
    sandbox_id = "fake-sandbox"

    def __init__(self) -> None:
        self.killed = False
        self.shell_commands: list[str] = []

    def run_shell(self, command: str, timeout_s: int = 120) -> ShellResult:
        self.shell_commands.append(command)
        if "exit 1" in command:
            return ShellResult(
                stdout="",
                stderr="failed",
                error="SystemExit: 1",
                success=False,
            )
        if command.startswith("cat /home/user/logs/verifier/reward.txt"):
            return ShellResult(stdout="", stderr="", error=None, success=True)
        return ShellResult(
            stdout=f"shell: {command}",
            stderr="",
            error=None,
            success=True,
        )

    def kill(self) -> None:
        self.killed = True


def _extract_text(result) -> str:
    if hasattr(result, "content") and result.content:
        return result.content[0].text
    if hasattr(result, "data"):
        return str(result.data)
    return str(result)


def test_lists_single_terminal_tool_without_reset():
    env = TerminusEnvironment()

    obs = env.step(ListToolsAction())

    assert [tool.name for tool in obs.tools] == ["terminal"]


def test_reset_without_e2b_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    env = TerminusEnvironment()

    obs = env.reset()

    assert obs.done is True
    assert obs.metadata["status"] == "error"
    assert "E2B_API_KEY" in obs.metadata["error"]


def test_reset_runs_setup_and_stores_verify_commands(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "fake-key")
    fake_sandbox = FakeSandbox()
    monkeypatch.setattr(
        "terminus_env.server.terminus_env_environment.E2BSandbox",
        lambda api_key: fake_sandbox,
    )
    env = TerminusEnvironment()

    obs = env.reset(setup=["echo setup"], verify=["test -f answer.txt"])

    assert obs.done is False
    assert fake_sandbox.shell_commands == [
        "mkdir -p /home/user/logs/verifier",
        "echo setup",
    ]
    assert env.state.sandbox_id == "fake-sandbox"
    assert env.state.setup_results[0].success is True
    assert env.state.verify_commands == ["test -f answer.txt"]
    assert obs.metadata["verify_commands"] == ["test -f answer.txt"]


def test_reset_fails_when_setup_command_fails(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "fake-key")
    monkeypatch.setattr(
        "terminus_env.server.terminus_env_environment.E2BSandbox",
        lambda api_key: FakeSandbox(),
    )
    env = TerminusEnvironment()

    obs = env.reset(setup=["exit 1"], verify=["test -f answer.txt"])

    assert obs.done is True
    assert obs.metadata["status"] == "error"
    assert obs.metadata["setup_results"][0]["success"] is False


def test_terminal_command_runs_inside_existing_sandbox():
    env = TerminusEnvironment()
    fake_sandbox = FakeSandbox()
    env._sandbox = fake_sandbox
    env._state = TerminusState(episode_id="episode-1", sandbox_id="fake-sandbox")

    obs = env.step(
        CallToolAction(
            tool_name="terminal",
            arguments={"command": "pwd"},
        )
    )

    assert obs.error is None
    assert "shell: pwd" in _extract_text(obs.result)
    assert env.state.step_count == 1
    assert env.state.commands[0].command == "pwd"


def test_terminal_final_answer_runs_verify_commands():
    env = TerminusEnvironment()
    fake_sandbox = FakeSandbox()
    env._sandbox = fake_sandbox
    env._state = TerminusState(
        episode_id="episode-1",
        sandbox_id="fake-sandbox",
        verify_commands=["test -f answer.txt", "exit 1"],
    )

    obs = env.step(
        CallToolAction(
            tool_name="terminal",
            arguments={"final_answer": "done"},
        )
    )

    assert obs.error is None
    assert "Verification: 1/2 passed; reward=0.5" in _extract_text(obs.result)
    assert env.state.submitted_answer == "done"
    assert env.state.last_reward == 0.5
    assert [result.command for result in env.state.verify_results] == [
        "test -f answer.txt",
        "exit 1",
    ]
