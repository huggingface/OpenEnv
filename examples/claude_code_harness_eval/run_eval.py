# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate Claude Code on τ²-bench tasks through `HarnessEnvironment`.

Each task gets a fresh `tau2_env` (the simulated customer, the database and the
domain's tools) and a fresh Claude Code process with those tools injected. Every
customer message is one `step()`, and τ²-bench scores the conversation when the
customer is done.

    PYTHONPATH=src:envs:examples/claude_code_harness_eval \\
        python examples/claude_code_harness_eval/run_eval.py --domain airline --tasks 3

Requires the `claude` CLI (logged in, or `ANTHROPIC_API_KEY`), `HF_TOKEN` for the
simulated customer on Inference Providers, and `TAU2_DATA_DIR` outside the
`tau2_env` Docker image (see `envs/tau2_env/README.md`).
"""

from __future__ import annotations

import argparse
import tempfile

from openenv.core.harness import HarnessConfig
from tau2_env.server.tau2_environment import Tau2Environment
from tau2_harness import converse, harness_for


def run_task(task_id: str, args: argparse.Namespace) -> float:
    tau2 = Tau2Environment(
        domain=args.domain, split=args.split, user_model=args.user_model
    )
    observation = tau2.reset(task_id=task_id)
    config = HarnessConfig(
        name="claude-code",
        command=[args.claude],
        # Claude Code has no tools of its own here, so a scratch directory is enough.
        working_directory=tempfile.mkdtemp(prefix="claude-code-tau2-"),
        model=args.model,
        session_timeout_s=args.turn_timeout,
    )
    harness = harness_for(tau2, observation.metadata["policy"], config)
    print(f"\n=== {args.domain} task {task_id}")
    try:
        harness.reset()
        for message, turn in converse(
            tau2, harness, observation.metadata["user_message"]
        ):
            print(f"customer: {message}")
            if turn is None:
                continue
            for event in turn.metadata.get("turn_events", []):
                if event["type"] == "tool_call":
                    call = event["data"]
                    print(f"  -> {call['tool_name']}({call['arguments']})")
            print(
                f"agent: {turn.metadata.get('response') or turn.metadata.get('error', '')}"
            )
    finally:
        harness.close()
    breakdown = tau2.state.reward_info.get("reward_breakdown", {})
    print(f"reward: {tau2.state.reward:.2f} {breakdown}")
    return tau2.state.reward if tau2.state.done else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="airline", help="τ²-bench domain")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--tasks", type=int, default=3, help="how many tasks of the split to run"
    )
    parser.add_argument("--task-ids", nargs="*", help="run these tasks instead")
    parser.add_argument("--model", default="haiku", help="passed to `claude --model`")
    parser.add_argument(
        "--user-model",
        default=None,
        help="simulated customer model (Inference Providers)",
    )
    parser.add_argument(
        "--claude", default="claude", help="path to the Claude Code CLI"
    )
    parser.add_argument("--turn-timeout", type=float, default=180.0)
    args = parser.parse_args()

    task_ids = (
        args.task_ids
        or Tau2Environment(domain=args.domain, split=args.split).task_ids[: args.tasks]
    )
    rewards = {task_id: run_task(task_id, args) for task_id in task_ids}
    print(
        "\n"
        + "\n".join(
            f"{'PASS' if r >= 1.0 else 'FAIL'}  task {t}" for t, r in rewards.items()
        )
    )
    print(f"pass^1: {sum(r >= 1.0 for r in rewards.values())}/{len(rewards)}")


if __name__ == "__main__":
    main()
