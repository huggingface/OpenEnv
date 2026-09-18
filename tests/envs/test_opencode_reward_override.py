# SPDX-License-Identifier: BSD-3-Clause

"""OpenCode rollouts must only take ``reward.txt`` from verification, not from the agent."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opencode_env.models import RolloutResult  # noqa: E402
from opencode_env.server.opencode_environment import (  # noqa: E402
    OpenCodeEnvironment,
    REWARD_FILE,
)


class _Sandbox:
    """Sandbox whose reward file any command can write, as the agent's shell can."""

    sandbox_id = "sbx-test"

    def __init__(self, *, removable: bool = True) -> None:
        self.reward_file: str | None = None
        self.removable = removable

    def exec(self, cmd: str, timeout: int | None = None) -> SimpleNamespace:
        exit_code = 0
        if cmd.startswith("echo ") and cmd.endswith(f" > {REWARD_FILE}"):
            self.reward_file = cmd[len("echo ") : -len(f" > {REWARD_FILE}")]
        elif cmd.startswith(f"rm -f {REWARD_FILE}"):
            if self.removable:
                self.reward_file = None
            exit_code = 0 if self.reward_file is None else 1
        elif cmd == "exit 1":
            exit_code = 1
        return SimpleNamespace(exit_code=exit_code, stdout="", stderr="")

    def read_text(self, path: str) -> str:
        if path == REWARD_FILE and self.reward_file is not None:
            return self.reward_file + "\n"
        raise FileNotFoundError(path)


class _Session:
    def __init__(self, sandbox: _Sandbox, agent_commands: list[str]) -> None:
        self.sandbox = sandbox
        self._agent_commands = agent_commands

    def start_agent(self) -> None:
        pass

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        for cmd in self._agent_commands:
            self.sandbox.exec(cmd)
        return 0

    def close(self) -> None:
        pass


def _rollout(monkeypatch, sandbox, verify, agent_commands=()) -> RolloutResult:
    monkeypatch.setenv("E2B_API_KEY", "test")
    env = OpenCodeEnvironment()
    session = _Session(sandbox, list(agent_commands))
    monkeypatch.setattr(
        env,
        "_OpenCodeSessionFactory",
        lambda **kwargs: SimpleNamespace(create=lambda **kw: session),
    )
    monkeypatch.setattr(env, "_E2BSandboxBackend", lambda **kwargs: None)
    raw = env._run_rollout_impl(
        base_url="http://localhost:8000/v1",
        api_key="test",
        model="test-model",
        instruction="solve the task",
        setup=[],
        verify=verify,
        task_id="t1",
        mode="black_box",
        disable_thinking=False,
        max_tokens_cap=0,
        top_logprobs=0,
        agent_timeout_s=5,
        template="",
    )
    return RolloutResult.model_validate_json(raw)


def test_reward_file_written_by_the_agent_is_ignored(monkeypatch):
    result = _rollout(
        monkeypatch,
        _Sandbox(),
        ["true", "exit 1"],
        agent_commands=[f"echo 1.0 > {REWARD_FILE}"],
    )

    assert result.error is None
    assert result.reward == 0.5
    assert result.reward_override_ignored is None


def test_reward_file_written_by_a_verify_command_overrides_the_pass_rate(monkeypatch):
    result = _rollout(monkeypatch, _Sandbox(), [f"echo 0.8 > {REWARD_FILE}", "exit 1"])

    assert result.reward == 0.8
    assert result.reward_override_ignored is None


@pytest.mark.parametrize("value", ["nan", "inf", "-0.5", "1.5", "passed"])
def test_invalid_reward_file_falls_back_to_the_pass_rate(monkeypatch, value):
    result = _rollout(
        monkeypatch, _Sandbox(), [f"echo {value} > {REWARD_FILE}", "exit 1"]
    )

    assert result.reward == 0.5
    assert repr(value) in result.reward_override_ignored


def test_reward_file_that_cannot_be_removed_is_ignored(monkeypatch):
    result = _rollout(
        monkeypatch,
        _Sandbox(removable=False),
        ["true", "exit 1"],
        agent_commands=[f"echo 1.0 > {REWARD_FILE}"],
    )

    assert result.reward == 0.5
    assert "could not be removed" in result.reward_override_ignored
