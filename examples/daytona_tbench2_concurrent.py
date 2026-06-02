#!/usr/bin/env python3
"""Spawn N environments concurrently on Daytona.

Shows how Daytona scales horizontally — each sandbox is an isolated
container, so we can spin up many in parallel.

Usage:
    PYTHONPATH=src:envs uv run python examples/daytona_tbench2_concurrent.py
    PYTHONPATH=src:envs uv run python examples/daytona_tbench2_concurrent.py --n 20

Requires:
    DAYTONA_API_KEY environment variable.
"""

import argparse
import asyncio
import time

from openenv.core.containers.runtime.daytona_provider import DaytonaProvider
from tbench2_env import Tbench2Action, Tbench2Env


async def run_one(env_id: int, image, task_id: str) -> dict:
    """Spin up one sandbox, run a reset + step, tear it down."""
    t0 = time.time()
    provider = DaytonaProvider()
    # start_container and wait_for_ready are blocking (sync Daytona SDK),
    # so run them in threads to get actual concurrency.
    base_url = await asyncio.to_thread(provider.start_container, image)
    t_started = time.time()

    try:
        await asyncio.to_thread(provider.wait_for_ready, base_url, 300)
        t_ready = time.time()

        async with Tbench2Env(base_url=base_url, provider=provider) as env:
            result = await env.reset(task_id=task_id)
            instruction = result.observation.instruction[:80]

            cmd = f"printf 'sandbox-{env_id} on %s, 2^{env_id}=%s' $(hostname) $(python3 -c 'print(2**{env_id})')"
            result = await env.step(Tbench2Action(action_type="exec", command=cmd))
            output = result.observation.output.strip()

        t_done = time.time()
        return {
            "id": env_id,
            "ok": True,
            "start_s": round(t_started - t0, 1),
            "ready_s": round(t_ready - t_started, 1),
            "work_s": round(t_done - t_ready, 1),
            "total_s": round(t_done - t0, 1),
            "instruction": instruction,
            "whoami": output,
        }
    except Exception as e:
        return {"id": env_id, "ok": False, "error": str(e), "total_s": round(time.time() - t0, 1)}
    finally:
        await asyncio.to_thread(provider.stop_container)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of concurrent envs")
    parser.add_argument("--task-id", default="headless-terminal")
    args = parser.parse_args()

    image = DaytonaProvider.image_from_dockerfile("envs/tbench2_env/server/Dockerfile")
    print(f"Launching {args.n} concurrent Daytona sandboxes...")

    t0 = time.time()
    results = await asyncio.gather(
        *(run_one(i, image, args.task_id) for i in range(args.n))
    )
    wall_time = time.time() - t0

    print(f"\n{'id':>3}  {'status':>7}  {'start':>6}  {'ready':>6}  {'work':>6}  {'total':>6}  details")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["id"]):
        if r["ok"]:
            print(f"{r['id']:>3}  {'OK':>7}  {r['start_s']:>5.1f}s  {r['ready_s']:>5.1f}s  {r['work_s']:>5.1f}s  {r['total_s']:>5.1f}s  {r['whoami']}")
        else:
            print(f"{r['id']:>3}  {'FAIL':>7}  {'':>6}  {'':>6}  {'':>6}  {r['total_s']:>5.1f}s  {r['error'][:50]}")

    ok = sum(1 for r in results if r["ok"])
    print(f"\n{ok}/{args.n} succeeded in {wall_time:.1f}s wall time")
    return 0 if ok == args.n else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
