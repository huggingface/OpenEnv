#!/usr/bin/env python3
"""Train SWE with TRL AsyncGRPOTrainer + environment_factory.

Uses TRL's built-in AsyncRolloutWorker which handles:
- Tokenization via apply_chat_template
- Generation via /v1/completions with exact token_ids + logprobs
- Multi-turn tool calling (bash, answer)
- NCCL weight sync to vLLM
- Sample assembly (prompt_ids + completion_ids)

We only provide:
- SWEToolEnv: environment with bash() and answer() tools
- swe_reward: parses completion messages for grading result
- Dataset: SWE-Gym tasks as prompts

Prerequisites:
  - vLLM server running on a separate GPU
  - HF Sandbox or Docker backend available

Example:
  # GPU 0: vLLM
  CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-1.7B \\
    --tensor-parallel-size 1 --max-model-len 4096

  # GPU 1: Trainer
  CUDA_VISIBLE_DEVICES=1 \\
  SWE_MODEL=Qwen/Qwen3-1.7B \\
  PYTHONPATH=src:envs python examples/mini_swe_env/train_swe_async_grpo.py \\
    --task-variant lite --max-tasks 5 --max-steps 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
for _p in (_root / "src", _root / "envs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from datasets import Dataset  # noqa: E402
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer  # noqa: E402

from mini_swe_env.async_grpo.swe_tool_env import SWEToolEnv, swe_reward  # noqa: E402
from mini_swe_env.task_loader_swegym import load_swegym_tasks  # noqa: E402
from openenv.core.harness.sandbox import create_sandbox_backend  # noqa: E402


_log = logging.getLogger("swe-async-grpo")


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SWE with TRL AsyncGRPO")
    p.add_argument("--task-variant", default="lite", choices=["lite", "full"])
    p.add_argument("--max-tasks", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--sandbox-backend", default="hf", choices=["docker", "e2b", "hf"])
    p.add_argument("--vllm-url", default="http://localhost:8000")
    return p


def _must_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _arg_parser().parse_args()
    model = _must_env("SWE_MODEL")

    # ── Load SWE-Gym tasks ────────────────────────────────────────
    gym_tasks = load_swegym_tasks(args.task_variant)
    gym_tasks = gym_tasks[: args.max_tasks]
    if not gym_tasks:
        raise RuntimeError("No tasks loaded")

    _log.info("loaded %d SWE-Gym tasks", len(gym_tasks))

    # ── Build dataset ─────────────────────────────────────────────
    # Each row has "prompt" (chat messages) and "task_json" (full task
    # for SWEToolEnv.reset).  TRL passes all non-prompt columns to
    # reset(**row) and to reward_func(**kwargs).
    rows = []
    for gt in gym_tasks:
        swe_task = gt.to_swe_task()
        rows.append({
            "prompt": [{"role": "user", "content": swe_task.instruction}],
            "task_json": json.dumps(gt.to_dict()),
            "instance_id": gt.instance_id,
        })
    dataset = Dataset.from_list(rows)

    # ── Sandbox backend ───────────────────────────────────────────
    backend = create_sandbox_backend(args.sandbox_backend)

    # ── Train ─────────────────────────────────────────────────────
    trainer = AsyncGRPOTrainer(
        model=model,
        reward_funcs=swe_reward,
        train_dataset=dataset,
        environment_factory=SWEToolEnv.factory(backend),
        args=AsyncGRPOConfig(
            output_dir="outputs/swe_async_grpo",
            vllm_server_base_url=args.vllm_url,
            max_completion_length=2048,
            max_tool_calling_iterations=args.max_turns,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=1,
            learning_rate=1e-6,
            temperature=1.0,
            max_staleness=2,
            weight_sync_steps=1,
            max_inflight_tasks=2,
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=1,
            report_to=[],
        ),
    )

    _log.info(
        "starting AsyncGRPO training: model=%s tasks=%d max_steps=%d backend=%s",
        model, len(gym_tasks), args.max_steps, args.sandbox_backend,
    )

    trainer.train()

    _log.info("training complete: global_step=%s", getattr(trainer.state, "global_step", "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
