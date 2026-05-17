#!/usr/bin/env python3
"""Train SWE with AsyncGRPOTrainer + Pi agent + InterceptionServer.

Architecture:
    Pi (HF Sandbox) → InterceptionServer → SWERolloutWorker → vLLM /v1/completions
                                                             ← chat response back to Pi

vLLM runs on a separate GPU.  The trainer, interception server, and rollout
worker share a process on the training GPU.

Prerequisites:
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-1.7B \\
        --tensor-parallel-size 1 --max-model-len 4096

    CUDA_VISIBLE_DEVICES=1 \\
    SWE_MODEL=Qwen/Qwen3-1.7B \\
    INTERCEPTION_AUTH_TOKEN=secret123 \\
    INTERCEPTION_BASE_URL=http://localhost:8765 \\
    PYTHONPATH=src:envs python examples/mini_swe_env/train_swe_async_grpo.py
"""

from __future__ import annotations

import argparse
import asyncio
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
from transformers import AutoTokenizer  # noqa: E402
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer  # noqa: E402

from mini_swe_env.async_grpo.control_plane import (  # noqa: E402
    SWEAsyncControlPlane,
    SWEAsyncControlPlaneConfig,
)
from mini_swe_env.async_grpo.rollout_worker import (  # noqa: E402
    SWERolloutWorker,
    WorkerConfig,
)
from mini_swe_env.harness import SWEAgentConfig, SWESessionFactory  # noqa: E402
from mini_swe_env.task_loader_swegym import load_swegym_tasks  # noqa: E402
from openenv.core.harness.sandbox import create_sandbox_backend  # noqa: E402


_log = logging.getLogger("swe-async-grpo")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-variant", default="lite", choices=["lite", "full"])
    p.add_argument("--max-tasks", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--sandbox-backend", default="hf", choices=["docker", "e2b", "hf"])
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--agent", default="pi", choices=["pi", "opencode"])
    return p.parse_args()


def _env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _args()
    model = _env("SWE_MODEL")
    vllm_url = args.vllm_url
    vllm_key = os.environ.get("VLLM_API_KEY", "token").strip()

    # ── Load tasks ────────────────────────────────────────────────
    gym_tasks = load_swegym_tasks(args.task_variant)[: args.max_tasks]
    swe_tasks = [t.to_swe_task() for t in gym_tasks]
    _log.info("loaded %d tasks", len(swe_tasks))

    # ── Dataset (prompt per task) ─────────────────────────────────
    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": t.instruction}],
            "instance_id": t.instance_id,
        }
        for t in swe_tasks
    ])

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Interception control plane ────────────────────────────────
    control_cfg = SWEAsyncControlPlaneConfig.from_env()
    control_plane = SWEAsyncControlPlane(config=control_cfg)
    asyncio.run(control_plane.start())

    try:
        # ── Session factory (Pi in sandbox) ───────────────────────
        backend = create_sandbox_backend(args.sandbox_backend)
        session_factory = SWESessionFactory(
            agent=args.agent,
            config=SWEAgentConfig(
                base_url=control_plane.interception_base_url,
                api_key=control_plane.auth_token,
                model=model,
                agent_timeout_s=1800.0,
            ),
            sandbox_backend=backend,
            mode="interception_gate",
            interception_server=control_plane.server,
            interception_base_url=control_plane.interception_base_url,
        )

        # ── Rollout worker ────────────────────────────────────────
        worker = SWERolloutWorker(
            session_factory=session_factory,
            tasks=swe_tasks,
            tokenizer=tokenizer,
            vllm_base_url=vllm_url,
            vllm_api_key=vllm_key,
            vllm_model=model,
            config=WorkerConfig(
                max_inflight=2,
                max_turns=args.max_turns,
            ),
        )

        # ── Trainer ───────────────────────────────────────────────
        def _noop_reward(**kwargs: Any) -> list[float]:  # noqa: ANN401
            """Unused — rewards come from rollout_worker.advantage."""
            prompts = kwargs.get("prompts", [])
            return [0.0] * len(prompts)

        trainer = AsyncGRPOTrainer(
            model=model,
            reward_funcs=_noop_reward,
            train_dataset=dataset,
            processing_class=tokenizer,
            rollout_worker=worker,
            args=AsyncGRPOConfig(
                output_dir="outputs/swe_async_grpo",
                vllm_server_base_url=vllm_url,
                vllm_server_timeout=2400.0,
                max_completion_length=2048,
                max_steps=args.max_steps,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                num_generations=1,
                learning_rate=1e-6,
                temperature=1.0,
                max_staleness=4,
                weight_sync_steps=1,
                max_inflight_tasks=2,
                logging_steps=1,
                report_to="trackio",
                run_name=f"swe-grpo-{model.split('/')[-1]}",
                trackio_space_id=os.environ.get("TRACKIO_SPACE_ID", "").strip() or None,
            ),
        )

        _log.info("starting training: model=%s tasks=%d", model, len(swe_tasks))
        trainer.train()
        _log.info("done: step=%s", getattr(trainer.state, "global_step", "?"))
        return 0
    finally:
        asyncio.run(control_plane.stop())


if __name__ == "__main__":
    from typing import Any  # noqa: E402
    raise SystemExit(main())
