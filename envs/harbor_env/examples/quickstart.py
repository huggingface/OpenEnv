# SPDX-License-Identifier: BSD-3-Clause

"""Quickstart: solve a Harbor task through a running Harbor environment server.

Start the server first (it serves the bundled example task by default):

```bash
uv run --project envs/harbor_env server
```

then run this script:

```bash
PYTHONPATH=src:envs uv run python envs/harbor_env/examples/quickstart.py
```

It walks the full episode: read the instruction, inspect the working directory,
write a fix, and grade the result with the task's own verifier.
"""

from __future__ import annotations

import argparse
import asyncio

from harbor_env import HarborEnv


FIXED_STATS = '''"""Small statistics helpers."""


def total(values):
    """Return the sum of ``values``, or ``0`` when there are none."""
    return sum(values)


def mean(values):
    """Return the arithmetic mean of ``values``, or ``0.0`` when there are none."""
    if not values:
        return 0.0
    return total(values) / len(values)
'''


async def main(base_url: str, task_id: str) -> None:
    env = HarborEnv(base_url=base_url)
    try:
        start = await env.reset(task_id=task_id)
        print(
            f"task     : {start.observation.task_name} ({start.observation.mode} mode)"
        )
        print(f"workdir  : {start.observation.workdir}")
        print(f"\n{start.observation.instruction}")

        listing = await env.run("ls -1")
        print(f"files    : {listing.observation.output.split()}")

        # Grade the starting state, to see what an agent has to beat.
        before = await env.evaluate()
        print(f"\nbaseline reward: {before.reward}")
        print(f"  metrics      : {before.observation.info.get('reward_metrics')}")

        # Episodes are one-shot: `evaluate` ends them, so start a fresh one to
        # actually solve the task.
        await env.reset(task_id=task_id)
        await env.write_file("stats.py", FIXED_STATS)
        after = await env.evaluate()

        print(f"\nreward after fix: {after.reward} (done={after.done})")
        print(f"  source        : {after.observation.info.get('reward_source')}")
    finally:
        await env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--task-id", default="fix-sum-bug")
    args = parser.parse_args()
    asyncio.run(main(args.base_url, args.task_id))
