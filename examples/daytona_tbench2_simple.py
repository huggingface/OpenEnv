#!/usr/bin/env python3
"""Minimal example for the TB2 OpenEnv runner (Daytona mode).

Usage:
    PYTHONPATH=src:envs uv run python examples/daytona_tbench2_simple.py

Requires:
    DAYTONA_API_KEY environment variable.
"""

import asyncio
import os

from openenv.core.containers.runtime.daytona_provider import DaytonaProvider
from tbench2_env import Tbench2Action, Tbench2Env


async def main() -> int:
    tasks_dir = os.environ.get("TB2_TASKS_DIR")
    if not tasks_dir:
        print("TB2_TASKS_DIR not set. TB2 repo will be downloaded.")

    task_id = os.environ.get("TB2_TASK_ID", "headless-terminal")

    image = DaytonaProvider.image_from_dockerfile(
        "envs/tbench2_env/server/Dockerfile",
    )
    provider = DaytonaProvider()
    base_url = provider.start_container(image=image)
    provider.wait_for_ready(base_url, timeout_s=180)

    try:
        async with Tbench2Env(base_url=base_url, provider=provider) as env:
            result = await env.reset(task_id=task_id)
            print("Instruction head:")
            print(result.observation.instruction[:200])

            result = await env.step(Tbench2Action(action_type="exec", command="ls -la"))
            print("Command output:")
            print(result.observation.output[:400])
    finally:
        provider.stop_container()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
