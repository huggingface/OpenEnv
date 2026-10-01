# SPDX-License-Identifier: BSD-3-Clause

"""Offline tests for the Claude Code harness evaluation example (RFC 005).

`fake_claude_code.py` plays Claude Code: it speaks the same stream-json events
and calls the environment's tools over the real MCP bridge. The simulated
customer is scripted, so the adapter, `HarnessEnvironment` and a real τ²-bench
airline task run end to end without the CLI, a model or an API key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("tau2", reason="τ²-bench is not installed")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))
sys.path.insert(0, str(_REPO_ROOT / "examples" / "claude_code_harness_eval"))

import litellm
import tau2.utils.llm_utils as llm_utils
from openenv.core.harness import HarnessAction, HarnessConfig
from tau2_env.server.tau2_environment import Tau2Environment
from tau2_harness import AGENT_PROMPT, converse, harness_for

FAKE_CLAUDE = Path(__file__).parent / "fake_claude_code.py"


@pytest.fixture
def customer(monkeypatch):
    """Make the simulated customer say each of `replies` in turn."""
    replies: list[str] = []

    def completion(**kwargs):
        return litellm.completion(
            model="openai/scripted",
            messages=kwargs["messages"],
            mock_response=replies.pop(0),
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    return replies


def start(tmp_path: Path, session_timeout_s: float = 30.0):
    tau2 = Tau2Environment(domain="airline", split="test")
    observation = tau2.reset(task_id="2")
    config = HarnessConfig(
        name="claude-code",
        command=[sys.executable, "-u", str(FAKE_CLAUDE)],
        working_directory=str(tmp_path),
        env_vars={"FAKE_CLAUDE_ARGV": str(tmp_path / "argv.json")},
        model="haiku",
        session_timeout_s=session_timeout_s,
    )
    harness = harness_for(tau2, observation.metadata["policy"], config)
    return tau2, observation, harness


def test_conversation_runs_through_the_adapter(tmp_path, customer):
    customer += [
        "Hi, my user id is noah_muller_9847.",
        "That's all, thanks. ###STOP###",
    ]
    tau2, observation, harness = start(tmp_path)
    try:
        injected = harness.reset().metadata["injected_tools"]
        turns = list(converse(tau2, harness, observation.metadata["user_message"]))
    finally:
        harness.close()

    # Only the domain's tools reach Claude Code; it talks to the customer through its replies.
    assert "get_user_details" in injected
    assert "respond_to_user" not in injected and "done" not in injected

    (message, turn), (last_message, last_turn) = turns
    assert message == "Hi, my user id is noah_muller_9847."
    call, result = turn.metadata["turn_events"][:2]
    assert call["data"] == {
        "tool_name": "get_user_details",
        "arguments": {"user_id": "noah_muller_9847"},
    }
    assert '"user_id": "noah_muller_9847"' in result["data"]["result"]

    # Claude Code's reply went to the customer, who ended the conversation.
    assert last_message == "That's all, thanks." and last_turn is None
    assert tau2.state.done
    assert tau2.state.reward_info["reward_basis"] == ["DB", "COMMUNICATE"]

    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--allowedTools") + 1] == "mcp__env"
    assert argv[argv.index("--append-system-prompt") + 1].startswith(AGENT_PROMPT)
    assert "--strict-mcp-config" in argv


def test_claude_code_exiting_mid_turn_ends_the_conversation(tmp_path, customer):
    customer += ["crash"]
    tau2, observation, harness = start(tmp_path)
    try:
        harness.reset()
        ((_, turn),) = converse(tau2, harness, observation.metadata["user_message"])
    finally:
        harness.close()

    assert turn.done
    assert turn.metadata["error_type"] == "harness_crashed"
    assert "exited mid-turn" in turn.metadata["error"]


def test_claude_code_going_quiet_is_a_timeout(tmp_path, customer):
    customer += ["Hi, my user id is noah_muller_9847."]
    tau2, _, harness = start(tmp_path, session_timeout_s=1.0)
    try:
        harness.reset()
        # A step allowed longer than the adapter's silence limit hits the adapter's limit.
        turn = harness.step(HarnessAction(message="stall"), timeout_s=30.0)
    finally:
        harness.close()

    assert turn.done
    assert turn.metadata["error_type"] == "turn_timeout"
