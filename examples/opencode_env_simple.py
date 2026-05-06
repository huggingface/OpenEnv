#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end opencode_env example: write binary_search.py and verify it.

Hits the deployed HF Space ``AdithyaSK/opencode-env`` (override via
``OPENCODE_ENV_SPACE`` env var to point at your own Space or a local
container). The single MCP tool ``run_rollout`` does:

  1. Spawns a fresh E2B sandbox (using the prebaked ``opencode-rl``
     template — falls back to a cold install if the template isn't
     present in your E2B account).
  2. Bootstraps an in-sandbox FastAPI proxy that captures per-token
     logprobs (``mode="transparent_proxy"``).
  3. Runs ``opencode run`` with the instruction.
  4. Executes the verify bash commands; reward = passed / total.
  5. Returns a ``RolloutResult`` with reward + per-turn logprobs +
     the file contents the agent produced.

Prerequisites
-------------
- ``OPENAI_API_KEY`` in the environment (passed to the Space per-call;
  doesn't need to be a Space secret). Swap to ``endpoint="vllm"`` or
  ``endpoint="hf_router"`` for those backends.

Usage::

    PYTHONPATH=src:envs uv run python examples/opencode_env_simple.py

Expected output (~20s with the prebaked template)::

    reward: 1.0
    turns:  3
    files:  ['/home/user/workdir/binary_search.py', ...]
    wall:   19.8 s
"""

from __future__ import annotations

import asyncio
import os
import sys

# Make ``envs/`` importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))

from opencode_env import OpenCodeEnv  # noqa: E402
from opencode_env.client import _extract_text  # noqa: E402
from opencode_env.models import RolloutResult  # noqa: E402


SPACE = os.environ.get("OPENCODE_ENV_SPACE", "https://adithyask-opencode-env.hf.space")

INSTRUCTION = (
    "Create a single Python file named `binary_search.py` in the current "
    "working directory. Use the relative path `binary_search.py`. Expose "
    "exactly one function:\n"
    "    def binary_search(arr: list[int], target: int) -> int\n"
    "Return the index of `target` in the sorted list `arr`, or -1 if absent. "
    "Use the binary-search algorithm; do not call list.index."
)

VERIFY = [
    "test -f /home/user/workdir/binary_search.py",
    "python -c \"import sys; sys.path.insert(0, '/home/user/workdir'); "
    "import binary_search; "
    "assert binary_search.binary_search([1,2,3,4,5], 3) == 2; "
    "assert binary_search.binary_search([1,2,3], 99) == -1; "
    "assert binary_search.binary_search([], 1) == -1; "
    "print('OK')\"",
]


async def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is required (or set as a Space secret and "
            "drop the api_key= kwarg below).",
            file=sys.stderr,
        )
        return 2

    print(f"Hitting Space:   {SPACE}")
    print("Endpoint:        openai (gpt-4o-mini)")
    print(f"Instruction:     {INSTRUCTION.splitlines()[0]} ...")
    print()

    async with OpenCodeEnv(base_url=SPACE) as env:
        await env.reset()
        raw = await env.call_tool(
            "run_rollout",
            endpoint="openai",  # vllm | openai | hf_router
            api_key=api_key,  # or set as Space secret
            model="gpt-4o-mini",
            instruction=INSTRUCTION,
            setup=[],  # no setup commands
            verify=VERIFY,
            template="opencode-rl",  # prebaked E2B template
            task_id="binary_search_simple",
            agent_timeout_s=600,
        )
        result = RolloutResult.model_validate_json(_extract_text(raw))

    print("--- result ---")
    print(f"reward:    {result.reward}")
    print(f"turns:     {len(result.proxy_turns)}")
    print(f"tokens:    {sum(len(t.completion_tokens) for t in result.proxy_turns)}")
    print(f"sandbox:   {result.sandbox_id}")
    print(f"wall_s:    {result.wall_s}")
    print(f"files:     {sorted(result.files)}")
    print(f"verify:    {[(v.cmd[:40], v.exit_code) for v in result.verify_results]}")
    if result.error:
        print(f"error:     {result.error}")

    if result.proxy_turns:
        first = next((t for t in result.proxy_turns if t.completion_tokens), None)
        if first:
            print()
            print("--- first productive turn (first 8 tokens with logprobs) ---")
            toks = first.completion_tokens[:8]
            lps = first.per_token_logps[:8]
            for tok, lp in zip(toks, lps):
                print(f"  {tok!r:<14}  {lp:+.3f}")

    return 0 if result.reward == 1.0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
