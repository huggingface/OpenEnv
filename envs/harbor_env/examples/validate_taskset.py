# SPDX-License-Identifier: BSD-3-Clause

"""Validate a Harbor task set by running each task's own reference solution.

This is the check to run against a freshly generated task set — the equivalent
of Harbor's oracle agent (`harbor run -a oracle`). A task whose own solution
does not score full reward is broken, and training against it teaches nothing.

```bash
# the bundled example task
PYTHONPATH=src:envs uv run python envs/harbor_env/examples/validate_taskset.py

# a task set you generated with Repo2RLEnv (image-backed, so Docker mode)
PYTHONPATH=src:envs uv run --with docker python envs/harbor_env/examples/validate_taskset.py \\
    --tasks ./tasks --mode docker

# straight from the Hub
PYTHONPATH=src:envs uv run --with docker python envs/harbor_env/examples/validate_taskset.py \\
    --tasks hf://datasets/my-org/click-pr-tasks --mode docker
```

Runs in-process against the environment, so no server is needed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from harbor_env import HarborAction
from harbor_env.server import HarborEnvironment
from harbor_env.server.harbor_env_environment import DEFAULT_MODE


#: A task whose oracle scores below this is treated as broken.
PASS_THRESHOLD = 1.0


@dataclass
class Outcome:
    task_id: str
    reward: float | None
    detail: str

    @property
    def ok(self) -> bool:
        return self.reward is not None and self.reward >= PASS_THRESHOLD

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        score = "n/a" if self.reward is None else f"{self.reward:.3f}"
        return f"[{mark}] {self.task_id:<40} reward={score:<7} {self.detail}"


def validate(env: HarborEnvironment, task_id: str) -> Outcome:
    """Reset, apply the task's own solution, and grade it."""
    try:
        env.reset(task_id=task_id)
    except Exception as exc:
        return Outcome(task_id, None, f"reset failed: {exc}")

    solved = env.step(HarborAction(action_type="solve"))
    if not solved.success:
        return Outcome(
            task_id, None, f"solution failed: {solved.error or solved.output[-200:]}"
        )

    graded = env.step(HarborAction(action_type="evaluate"))
    detail = "" if graded.success else (graded.error or "")
    return Outcome(task_id, graded.reward, detail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", default=None, help="task directory or hf://datasets/<org>/<name>"
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=["docker", "local"],
        help=(
            f"sandbox backend (default: {DEFAULT_MODE}). `local` needs no Docker but "
            "only runs self-contained tasks that declare no network, resource or user "
            "policy."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N tasks")
    args = parser.parse_args()

    env = HarborEnvironment(
        tasks=args.tasks,
        mode=args.mode,
        # This script *is* the orchestration: it must be able to solve and
        # grade even where the serving deployment disables those controls.
        allow_control_actions=True,
    )
    task_ids = env.catalog.task_ids()[: args.limit]
    if not task_ids:
        print(f"no Harbor tasks found under {env.catalog.root}", file=sys.stderr)
        return 1

    print(
        f"validating {len(task_ids)} task(s) from {env.catalog.root} in {args.mode} mode\n"
    )
    outcomes = []
    try:
        for task_id in task_ids:
            outcome = validate(env, task_id)
            outcomes.append(outcome)
            print(outcome)
    finally:
        env.close()

    failed = [o for o in outcomes if not o.ok]
    print(
        f"\n{len(outcomes) - len(failed)}/{len(outcomes)} tasks solvable by their own oracle"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
