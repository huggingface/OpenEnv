#!/usr/bin/env python3
"""Train SWE with AsyncGRPOTrainer + Pi agent + InterceptionServer.

Architecture:
    Pi (HF Sandbox) → InterceptionServer → SWERolloutWorker → vLLM /v1/completions
                                                             ← chat response back to Pi

vLLM runs on a separate GPU.  The trainer, interception server, and rollout
worker share a process on the training GPU.

The InterceptionServer runs in a background thread with its own asyncio
event loop so it stays alive while the synchronous trainer.train() runs
on the main thread.

Prerequisites:
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-1.7B \\
        --tensor-parallel-size 1 --max-model-len 40960

    CUDA_VISIBLE_DEVICES=1 \\
    SWE_MODEL=Qwen/Qwen3-1.7B \\
    INTERCEPTION_AUTH_TOKEN=secret123 \\
    INTERCEPTION_BASE_URL=http://localhost:8765 \\
    PYTHONPATH=src:envs python examples/mini_swe_env/train_swe_async_grpo.py
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
for _p in (_root, _root / "src", _root / "envs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from datasets import Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer  # noqa: E402

try:
    from peft import LoraConfig, TaskType  # noqa: E402
except ImportError:
    LoraConfig = None  # type: ignore[assignment,misc]
    TaskType = None  # type: ignore[assignment,misc]

from examples.mini_swe_env.async_grpo.control_plane import (  # noqa: E402
    SWEAsyncControlPlane,
    SWEAsyncControlPlaneConfig,
)
from examples.mini_swe_env.async_grpo.rollout_worker import (  # noqa: E402
    SWERolloutWorker,
    WorkerConfig,
)
from mini_swe_env.harness import SWEAgentConfig, SWESessionFactory  # noqa: E402
from mini_swe_env.task_loader_swegym import load_swegym_tasks  # noqa: E402
from openenv.core.harness.sandbox import create_sandbox_backend  # noqa: E402


_log = logging.getLogger("swe-async-grpo")


# ── InterceptionServer background thread ──────────────────────────


def _run_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Run an asyncio event loop forever in the current thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def start_interception_server(
    control_plane: SWEAsyncControlPlane,
) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start the InterceptionServer in a daemon thread.

    Returns the event loop and the thread so the caller can shut it down.
    The aiohttp server runs on this loop and stays alive as long as the
    thread is running.
    """
    loop = asyncio.new_event_loop()
    # Start the server on the new loop.
    future = asyncio.run_coroutine_threadsafe(control_plane.start(), loop)
    # The loop must be running for the coroutine to execute.
    thread = threading.Thread(
        target=_run_event_loop, args=(loop,), daemon=True, name="interception-server"
    )
    thread.start()
    # Wait for start() to complete.
    future.result(timeout=30)
    return loop, thread


def stop_interception_server(
    control_plane: SWEAsyncControlPlane,
    loop: asyncio.AbstractEventLoop,
    thread: threading.Thread,
) -> None:
    """Shut down the InterceptionServer and its event loop."""
    future = asyncio.run_coroutine_threadsafe(control_plane.stop(), loop)
    try:
        future.result(timeout=10)
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


# ── CLI ────────────────────────────────────────────────────────────


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-variant", default="lite", choices=["lite", "full"])
    p.add_argument("--max-tasks", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--sandbox-backend", default="hf", choices=["docker", "e2b", "hf"])
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--agent", default="pi", choices=["pi", "opencode"])
    p.add_argument(
        "--num-generations", type=int, default=16,
        help="Rollouts per prompt for group-relative advantage (GRPO). Polar used 16.",
    )
    return p.parse_args()


def _env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _derive_checkpoint_repo_id() -> str | None:
    explicit = os.environ.get("SWE_HUB_MODEL_ID", "").strip()
    if explicit:
        return explicit
    space_id = (
        os.environ.get("HF_SPACE_ID", "").strip()
        or os.environ.get("SPACE_ID", "").strip()
    )
    if "/" not in space_id:
        return None
    return f"{space_id}-checkpoints"


def _build_checkpoint_args() -> tuple[dict[str, Any], str | None, bool]:
    in_space = bool(
        os.environ.get("HF_SPACE_ID")
        or os.environ.get("SPACE_ID")
        or os.environ.get("SPACE_HOST")
    )
    enabled = _bool_env("SWE_CHECKPOINT_TO_HUB", default=in_space)
    if not enabled:
        return {}, None, False

    repo_id = _derive_checkpoint_repo_id()
    if not repo_id:
        _log.warning(
            "checkpointing requested, but SWE_HUB_MODEL_ID/HF_SPACE_ID missing; "
            "disabling hub checkpointing"
        )
        return {}, None, False

    save_steps = _int_env("SWE_CHECKPOINT_SAVE_STEPS", default=2)
    save_total_limit = _int_env("SWE_CHECKPOINT_SAVE_TOTAL_LIMIT", default=2)

    checkpoint_args = {
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "push_to_hub": True,
        "hub_model_id": repo_id,
        "hub_strategy": "checkpoint",
        "hub_private_repo": _bool_env("SWE_HUB_PRIVATE_REPO", default=True),
        "hub_token": os.environ.get("HF_TOKEN", "").strip() or None,
    }

    resume_pref = os.environ.get("SWE_RESUME_FROM_CHECKPOINT", "auto").strip().lower()
    if resume_pref in {"", "0", "false", "off", "none"}:
        resume_from_checkpoint: str | None = None
    elif resume_pref == "auto":
        resume_from_checkpoint = "last-checkpoint"
    else:
        resume_from_checkpoint = os.environ.get(
            "SWE_RESUME_FROM_CHECKPOINT", ""
        ).strip()

    return checkpoint_args, resume_from_checkpoint, True


def _filter_async_grpo_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(AsyncGRPOConfig)
    return {k: v for k, v in values.items() if k in sig.parameters}


def _is_missing_checkpoint_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "last-checkpoint" not in msg:
        return False
    hints = (
        "no valid checkpoint",
        "can't find",
        "cannot find",
        "does not exist",
        "not found",
        "404",
    )
    return any(hint in msg for hint in hints)


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
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": t.instruction}],
                "instance_id": t.instance_id,
            }
            for t in swe_tasks
        ]
    )

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Interception control plane (background thread) ────────────
    # Only rank 0 owns the interception server, session factory, and rollout
    # worker.  Other ranks only participate in gradient computation.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    control_plane: SWEAsyncControlPlane | None = None
    server_loop = None
    server_thread = None
    worker: SWERolloutWorker | None = None

    if is_main:
        control_cfg = SWEAsyncControlPlaneConfig.from_env()
        control_plane = SWEAsyncControlPlane(config=control_cfg)
        server_loop, server_thread = start_interception_server(control_plane)
        _log.info("InterceptionServer running in background thread")

    try:
        # ── Session factory + rollout worker (rank 0 only) ────────
        if is_main:
            assert control_plane is not None
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
                    num_generations=args.num_generations,
                ),
            )

        # ── Trainer ───────────────────────────────────────────────
        def _noop_reward(**kwargs: Any) -> list[float]:
            """Unused — rewards come from rollout_worker.advantage."""
            prompts = kwargs.get("prompts", [])
            return [0.0] * len(prompts)

        checkpoint_args, resume_from_checkpoint, checkpoint_requested = (
            _build_checkpoint_args()
        )
        async_grpo_args: dict[str, Any] = {
            "output_dir": os.path.join(
                os.environ.get("HOME", "/tmp"), "outputs/swe_async_grpo"
            ),
            "vllm_server_base_url": vllm_url,
            "vllm_server_timeout": 2400.0,
            "model_init_kwargs": {"dtype": "bfloat16"},
            "max_completion_length": 2048,
            "max_steps": args.max_steps,
            # Polar: rollout_batch_size=4. With single GPU trainer,
            # batch_size=4 via gradient accumulation.
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_generations": args.num_generations,
            # Polar: lr=1e-6, weight_decay=0.1
            "learning_rate": 1e-6,
            "weight_decay": 0.1,
            "temperature": 1.0,
            "optim": "adamw_bnb_8bit",
            "bf16": True,
            "gradient_checkpointing": True,
            "max_staleness": 4,
            "weight_sync_steps": 1,
            "max_inflight_tasks": 2,
            "logging_steps": 1,
            "report_to": "trackio",
            "run_name": f"swe-grpo-{model.split('/')[-1]}",
            "trackio_space_id": os.environ.get("TRACKIO_SPACE_ID", "").strip() or None,
        }
        filtered_checkpoint_args = _filter_async_grpo_kwargs(checkpoint_args)
        async_grpo_args.update(filtered_checkpoint_args)

        checkpoint_enabled = bool(filtered_checkpoint_args.get("push_to_hub"))
        if checkpoint_requested and not checkpoint_enabled:
            _log.warning(
                "checkpointing requested, but AsyncGRPOConfig does not expose hub args; "
                "continuing without hub checkpointing"
            )
            resume_from_checkpoint = None

        # ── LoRA config (Unsloth-recommended hyperparams for Qwen3.5) ──
        peft_config = None
        if LoraConfig is not None:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=16,  # alpha == r recommended for Qwen3.5
                lora_dropout=0,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
            )
            _log.info(
                "LoRA config: r=%d alpha=%d targets=%s",
                peft_config.r,
                peft_config.lora_alpha,
                peft_config.target_modules,
            )

        trainer = AsyncGRPOTrainer(
            model=model,
            reward_funcs=_noop_reward,
            train_dataset=dataset,
            processing_class=tokenizer,
            rollout_worker=worker,
            peft_config=peft_config,
            args=AsyncGRPOConfig(**async_grpo_args),
        )

        _log.info(
            "starting training: model=%s tasks=%d checkpointing=%s resume=%s",
            model,
            len(swe_tasks),
            checkpoint_enabled,
            resume_from_checkpoint or "none",
        )
        if resume_from_checkpoint is None:
            trainer.train()
        else:
            try:
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            except Exception as exc:
                if (
                    resume_from_checkpoint == "last-checkpoint"
                    and _is_missing_checkpoint_error(exc)
                ):
                    _log.warning(
                        "No hub checkpoint found at 'last-checkpoint'; starting from scratch"
                    )
                    trainer.train()
                else:
                    raise

        if checkpoint_enabled and hasattr(trainer, "push_to_hub"):
            trainer.push_to_hub(
                commit_message=f"Final checkpoint at step {getattr(trainer.state, 'global_step', '?')}"
            )

        _log.info("done: step=%s", getattr(trainer.state, "global_step", "?"))
        return 0
    finally:
        if control_plane is not None and server_loop is not None and server_thread is not None:
            stop_interception_server(control_plane, server_loop, server_thread)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as exc:
        if exc.code == 0:
            # Normal exit — sleep forever to prevent Space restart.
            import time as _t

            _log.info("training completed successfully; sleeping to hold Space alive")
            while True:
                _t.sleep(3600)
        else:
            # Crash — log and sleep forever so logs are preserved.
            _log.exception("training crashed (exit code %s); sleeping to preserve logs", exc.code)
            import time as _t

            while True:
                _t.sleep(3600)
    except Exception:
        _log.exception("unhandled exception in training; sleeping to preserve logs")
        import time as _t

        while True:
            _t.sleep(3600)
