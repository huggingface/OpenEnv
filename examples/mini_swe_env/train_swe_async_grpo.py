#!/usr/bin/env python3
"""Train SWE with AsyncGRPOTrainer + Pi agent + InterceptionServer.

Architecture:
    Pi (HF Sandbox) → InterceptionServer → SWERolloutWorker → vLLM /v1/chat/completions
                                                             ← chat response back to Pi

vLLM runs on a separate GPU.  The trainer, interception server, and rollout
worker share a process on the training GPU.

The InterceptionServer runs in a background thread with its own asyncio
event loop so it stays alive while the synchronous trainer.train() runs
on the main thread.

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
import contextlib
import inspect
import logging
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any

import torch
_root = Path(__file__).resolve().parent.parent.parent
for _p in (_root, _root / "src", _root / "envs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from datasets import Dataset  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer  # noqa: E402
from trl.trainer.utils import _ChunkedLogProbFunction  # noqa: E402

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
    p.add_argument(
        "--task-offset",
        type=int,
        default=int(os.environ.get("SWE_TASK_OFFSET", "0")),
    )
    p.add_argument(
        "--task-stride",
        type=int,
        default=int(os.environ.get("SWE_TASK_STRIDE", "1")),
    )
    p.add_argument("--task-indices", default=os.environ.get("SWE_TASK_INDICES", ""))
    p.add_argument(
        "--repeat-tasks",
        type=int,
        default=int(os.environ.get("SWE_REPEAT_TASKS", "1")),
    )
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument(
        "--num-generations",
        type=int,
        default=int(os.environ.get("SWE_NUM_GENERATIONS", "4")),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("SWE_TEMPERATURE", "1.0")),
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.environ.get("SWE_LEARNING_RATE", "1e-6")),
    )
    p.add_argument(
        "--max-completion-tokens",
        type=int,
        default=int(os.environ.get("SWE_MAX_COMPLETION_TOKENS", "2048")),
    )
    p.add_argument(
        "--sandbox-backend",
        default=os.environ.get("SWE_SANDBOX_BACKEND", "hf"),
        choices=["docker", "e2b", "hf", "local"],
    )
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument(
        "--agent",
        default=os.environ.get("SWE_AGENT", "pi"),
        choices=["pi", "opencode"],
    )
    p.add_argument(
        "--agent-thinking",
        default=os.environ.get("SWE_AGENT_THINKING", "off"),
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


def _float_env(name: str, default: float, *, min_value: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = float(raw)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _parse_task_indices(raw: str) -> list[int]:
    text = raw.strip()
    if not text:
        return []
    indices: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        indices.append(int(piece))
    if not indices:
        raise ValueError("task-indices must contain at least one integer")
    return indices


def _select_task_indices(
    *,
    total_tasks: int,
    max_tasks: int,
    task_offset: int,
    task_stride: int,
    task_indices_raw: str,
    repeat_tasks: int,
) -> list[int]:
    if total_tasks <= 0:
        raise ValueError("SWE-Gym task source is empty")
    if max_tasks < 1:
        raise ValueError(f"max_tasks must be >= 1, got {max_tasks}")
    if task_offset < 0:
        raise ValueError(f"task_offset must be >= 0, got {task_offset}")
    if task_stride < 1:
        raise ValueError(f"task_stride must be >= 1, got {task_stride}")
    if repeat_tasks < 1:
        raise ValueError(f"repeat_tasks must be >= 1, got {repeat_tasks}")

    base_indices = _parse_task_indices(task_indices_raw)
    if not base_indices:
        base_indices = list(range(task_offset, total_tasks, task_stride))[:max_tasks]

    if not base_indices:
        raise ValueError(
            "task selection produced no tasks; adjust max-tasks/task-offset/task-stride"
        )

    for idx in base_indices:
        if idx < 0 or idx >= total_tasks:
            raise IndexError(
                f"task index {idx} out of range [0, {total_tasks - 1}]"
            )

    selected_indices: list[int] = []
    for _ in range(repeat_tasks):
        selected_indices.extend(base_indices)
    return selected_indices


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


def _is_main_process() -> bool:
    """Return true for the single rank that owns rollout infrastructure."""
    rank = os.environ.get("RANK")
    if rank is not None and rank.strip():
        return rank.strip() == "0"
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and local_rank.strip():
        return local_rank.strip() == "0"
    return True


def _train_dtype_from_env() -> torch.dtype:
    raw = os.environ.get("SWE_TRAIN_DTYPE", "bf16").strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    dtype = mapping.get(raw)
    if dtype is None:
        raise ValueError(
            "SWE_TRAIN_DTYPE must be one of "
            f"{sorted(mapping)}, got {raw!r}"
        )
    return dtype


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _csv_env(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    values = tuple(piece.strip() for piece in raw.split(",") if piece.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one non-empty value")
    return values


def _count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _single_gpu_trainable_param_limit() -> int:
    raw = os.environ.get("SWE_SINGLE_GPU_TRAINABLE_PARAM_LIMIT", "").strip()
    if not raw:
        return 9_000_000_000
    value = int(raw)
    if value < 1:
        raise ValueError(
            f"SWE_SINGLE_GPU_TRAINABLE_PARAM_LIMIT must be >= 1, got {value}"
        )
    return value


def _model_context_limit(model_name: str) -> int:
    cfg = AutoConfig.from_pretrained(model_name)
    candidates = [
        getattr(cfg, "max_position_embeddings", None),
        getattr(cfg, "model_max_length", None),
        getattr(cfg, "max_seq_len", None),
        getattr(cfg, "seq_length", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    raise ValueError(
        f"Could not infer a positive context limit for model {model_name!r}"
    )


def _default_lora_target_modules(model_name: str) -> tuple[str, ...]:
    model_id = model_name.lower().replace("_", "")
    if any(
        token in model_id
        for token in ("qwen", "llama", "mistral", "mixtral", "deepseek")
    ):
        return (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    return ("q_proj", "k_proj", "v_proj", "o_proj")


def _sanitize_lora_merged_weight_name(name: str) -> str | None:
    cleaned = name.removeprefix("module.").removeprefix("base_model.model.")
    if (
        ".lora_" in cleaned
        or ".modules_to_save." in cleaned
        or ".original_module." in cleaned
    ):
        return None
    return cleaned.replace(".base_layer.", ".")


def _lora_config_from_env(model_name: str) -> tuple[Any | None, str]:
    if not _bool_env("SWE_LORA", False):
        return None, "full"

    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:  # pragma: no cover - exercised in real runtime
        raise RuntimeError(
            "SWE_LORA=1 requires `peft` to be installed in the active environment."
        ) from exc

    rank = _int_env("SWE_LORA_R", 16)
    alpha = _int_env("SWE_LORA_ALPHA", rank * 2)
    dropout = _float_env("SWE_LORA_DROPOUT", 0.0)
    target_modules = list(
        _csv_env("SWE_LORA_TARGET_MODULES")
        or _default_lora_target_modules(model_name)
    )
    modules_to_save = list(_csv_env("SWE_LORA_MODULES_TO_SAVE")) or None
    bias = os.environ.get("SWE_LORA_BIAS", "none").strip() or "none"

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias=bias,
        use_rslora=_bool_env("SWE_LORA_USE_RSLORA", False),
    )
    summary = (
        "lora("
        f"r={rank},alpha={alpha},dropout={dropout:g},"
        f"targets={','.join(target_modules)}"
        ")"
    )
    return config, summary


@contextlib.contextmanager
def _patched_async_grpo_model_loader(
    *,
    dtype: torch.dtype,
    attn_implementation: str | None,
    lora_config: Any | None = None,
):
    """Force AsyncGRPOTrainer to load the policy in the requested dtype.

    TRL's experimental AsyncGRPOTrainer currently hardcodes
    ``AutoModelForCausalLM.from_pretrained(..., dtype=torch.float32)``,
    which makes 8B full-finetuning overflow 80GB H100s on the first
    optimizer step. Patch the loader just around trainer construction so
    the example can run with bf16/fp16 policy weights.
    """

    original = AutoModelForCausalLM.__dict__["from_pretrained"]

    def _patched(
        cls,
        pretrained_model_name_or_path: str,
        *args: Any,
        **kwargs: Any,
    ):
        requested_dtype = kwargs.get("dtype")
        if requested_dtype in {None, torch.float32, "float32", "fp32"}:
            kwargs["dtype"] = dtype
        if attn_implementation and not kwargs.get("attn_implementation"):
            kwargs["attn_implementation"] = attn_implementation
        kwargs.setdefault("low_cpu_mem_usage", True)
        model = original.__get__(None, cls)(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
        )
        if lora_config is not None:
            from peft import get_peft_model

            model = get_peft_model(model, lora_config)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model, "config"):
                model.config.use_cache = False
        return model

    AutoModelForCausalLM.from_pretrained = classmethod(_patched)
    try:
        yield
    finally:
        AutoModelForCausalLM.from_pretrained = original


def _patch_lora_weight_streaming(trainer: AsyncGRPOTrainer) -> None:
    def _streaming_iter_with_merged_lora(self: AsyncGRPOTrainer):
        device = self.accelerator.device
        model = self.model
        merge_adapter = getattr(model, "merge_adapter", None)
        unmerge_adapter = getattr(model, "unmerge_adapter", None)
        get_base_model = getattr(model, "get_base_model", None)
        if not callable(merge_adapter) or not callable(unmerge_adapter):
            raise RuntimeError(
                "SWE_LORA requires a PEFT model with merge_adapter()/unmerge_adapter() "
                "support so vLLM can receive merged weights."
            )
        if not callable(get_base_model):
            raise RuntimeError(
                "SWE_LORA weight sync requires get_base_model() on the PEFT wrapper."
            )

        merge_adapter()
        try:
            base_model = get_base_model()
            for name, param in base_model.named_parameters():
                mapped_name = _sanitize_lora_merged_weight_name(name)
                if mapped_name is None:
                    continue
                full = param.full_tensor() if hasattr(param, "full_tensor") else param.detach()
                if full.device != device:
                    full = full.to(device)
                yield mapped_name, full
        finally:
            unmerge_adapter()

    trainer._streaming_iter = types.MethodType(  # type: ignore[method-assign]
        _streaming_iter_with_merged_lora,
        trainer,
    )


def _chunked_logprob_backbone(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "model", model)
    if hasattr(backbone, "lm_head") and hasattr(backbone, "model"):
        return backbone.model
    return backbone


def _patch_chunked_lm_head_for_wrapped_causal_lm(
    model: torch.nn.Module,
    *,
    temperature: float,
    chunk_size: int = 8192,
) -> None:
    def _chunked_forward(
        self: torch.nn.Module,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        completion_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        assert labels is not None, "requires labels to not be None for logprob computation"

        outputs = _chunked_logprob_backbone(self)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            all_hidden_states = getattr(outputs, "hidden_states", None)
            if not all_hidden_states:
                raise AttributeError(
                    "Chunked GRPO forward requires `last_hidden_state` or "
                    "`hidden_states` on the model outputs."
                )
            hidden_states = all_hidden_states[-1]

        hidden_states = hidden_states[:, :-1, :]
        labels = labels[:, 1:]

        b, s, h = hidden_states.shape
        hidden_flat = hidden_states.reshape(b * s, h).contiguous()
        targets_flat = labels.reshape(b * s).contiguous()

        valid_mask = None
        if completion_mask is not None:
            completion_mask = completion_mask[:, 1:]
            valid_mask = completion_mask.bool().reshape(b * s)
            hidden_flat = hidden_flat[valid_mask]
            targets_flat = targets_flat[valid_mask]

        logprobs_valid, entropy_valid = _ChunkedLogProbFunction.apply(
            hidden_flat,
            self.lm_head.weight,
            targets_flat,
            temperature,
            chunk_size,
            logit_scale,
        )

        if valid_mask is not None:
            logprobs = torch.zeros(
                b * s,
                device=logprobs_valid.device,
                dtype=logprobs_valid.dtype,
            )
            entropy = torch.zeros(
                b * s,
                device=entropy_valid.device,
                dtype=entropy_valid.dtype,
            )
            logprobs[valid_mask] = logprobs_valid
            entropy[valid_mask] = entropy_valid
        else:
            logprobs = logprobs_valid
            entropy = entropy_valid

        return {
            "log_probs": logprobs.reshape(b, s),
            "entropy": entropy.reshape(b, s),
        }

    model.forward = types.MethodType(_chunked_forward, model)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _args()
    if args.num_generations < 2:
        raise ValueError(
            f"--num-generations must be >= 2 for GRPO, got {args.num_generations}"
        )
    model = _env("SWE_MODEL")
    vllm_url = args.vllm_url
    vllm_key = os.environ.get("VLLM_API_KEY", "token").strip()
    train_dtype = _train_dtype_from_env()
    train_attn_implementation = _optional_env("SWE_TRAIN_ATTN_IMPLEMENTATION")
    lora_config, adapter_summary = _lora_config_from_env(model)
    if lora_config is not None and _bool_env("SWE_DISABLE_WEIGHT_TRANSFER", False):
        raise RuntimeError(
            "SWE_LORA requires SWE_DISABLE_WEIGHT_TRANSFER=0 so rollout generation "
            "tracks the trained policy."
        )

    # ── Load tasks ────────────────────────────────────────────────
    all_gym_tasks = load_swegym_tasks(args.task_variant)
    selected_indices = _select_task_indices(
        total_tasks=len(all_gym_tasks),
        max_tasks=args.max_tasks,
        task_offset=args.task_offset,
        task_stride=args.task_stride,
        task_indices_raw=args.task_indices,
        repeat_tasks=args.repeat_tasks,
    )
    gym_tasks = [all_gym_tasks[idx] for idx in selected_indices]
    swe_tasks = [t.to_swe_task() for t in gym_tasks]
    unique_indices = sorted(set(selected_indices))
    _log.info(
        "loaded %d task slots (%d unique tasks) indices=%s",
        len(swe_tasks),
        len(unique_indices),
        selected_indices,
    )
    _log.info(
        "task instances=%s",
        [all_gym_tasks[idx].instance_id for idx in unique_indices],
    )

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
    model_context_limit = _model_context_limit(model)

    # ── Interception control plane (background thread) ────────────
    is_main_process = _is_main_process()
    control_plane: SWEAsyncControlPlane | None = None
    server_loop: asyncio.AbstractEventLoop | None = None
    server_thread: threading.Thread | None = None
    if is_main_process:
        control_cfg = SWEAsyncControlPlaneConfig.from_env()
        control_plane = SWEAsyncControlPlane(config=control_cfg)
        server_loop, server_thread = start_interception_server(control_plane)
        _log.info("InterceptionServer running in background thread")

    try:
        # ── Session factory (Pi in sandbox) ───────────────────────
        worker: SWERolloutWorker | None = None
        if is_main_process:
            assert control_plane is not None
            backend_kwargs: dict[str, Any] = {}
            if args.sandbox_backend == "hf":
                backend_kwargs = {
                    "create_retries": _int_env("SWE_HF_SANDBOX_CREATE_RETRIES", 6),
                    "create_backoff_s": _float_env(
                        "SWE_HF_SANDBOX_CREATE_BACKOFF_S", 20.0
                    ),
                }
            elif args.sandbox_backend == "local":
                backend_kwargs = {
                    "root_dir": os.environ.get("OPENENV_LOCAL_SANDBOX_ROOT", "").strip()
                    or None,
                    "preserve_root": _bool_env(
                        "OPENENV_LOCAL_SANDBOX_PRESERVE",
                        False,
                    ),
                }
            default_rollout_inflight = min(
                args.num_generations,
                1 if args.sandbox_backend == "hf" else max(2, args.num_generations),
            )
            backend = create_sandbox_backend(args.sandbox_backend, **backend_kwargs)
            session_factory = SWESessionFactory(
                agent=args.agent,
                config=SWEAgentConfig(
                    base_url=control_plane.interception_base_url,
                    api_key=control_plane.auth_token,
                    model=model,
                    agent_timeout_s=_float_env("SWE_AGENT_TIMEOUT_S", 1800.0),
                    thinking=args.agent_thinking,
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
                    max_inflight=_int_env(
                        "SWE_ROLLOUT_MAX_INFLIGHT",
                        default_rollout_inflight,
                    ),
                    max_rollout_attempts=_int_env(
                        "SWE_ROLLOUT_MAX_ATTEMPTS",
                        4,
                    ),
                    num_generations=args.num_generations,
                    request_timeout_s=_float_env("SWE_ROLLOUT_REQUEST_TIMEOUT_S", 600.0),
                    max_turns=args.max_turns,
                    max_model_len=_int_env(
                        "SWE_VLLM_MAX_MODEL_LEN",
                        model_context_limit,
                    ),
                    max_completion_tokens=args.max_completion_tokens,
                    temperature=args.temperature,
                    failure_backoff_s=_float_env(
                        "SWE_ROLLOUT_FAILURE_BACKOFF_S",
                        30.0,
                    ),
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
        train_optim = os.environ.get("SWE_OPTIM", "").strip() or "paged_adamw_8bit"
        async_grpo_args: dict[str, Any] = {
            "output_dir": os.path.join(
                os.environ.get("HOME", "/tmp"), "outputs/swe_async_grpo"
            ),
            "vllm_server_base_url": vllm_url,
            "vllm_server_timeout": _float_env("SWE_ROLLOUT_QUEUE_TIMEOUT_S", 900.0),
            "max_completion_length": args.max_completion_tokens,
            "max_steps": args.max_steps,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_generations": args.num_generations,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "optim": train_optim,
            "bf16": train_dtype == torch.bfloat16,
            "fp16": train_dtype == torch.float16,
            "gradient_checkpointing": True,
            "torch_empty_cache_steps": _int_env(
                "SWE_TORCH_EMPTY_CACHE_STEPS",
                1,
            ),
            "max_staleness": _int_env("SWE_ASYNC_MAX_STALENESS", 4),
            "weight_sync_steps": _int_env("SWE_ASYNC_WEIGHT_SYNC_STEPS", 1),
            "max_inflight_tasks": _int_env(
                "SWE_ASYNC_MAX_INFLIGHT_TASKS",
                args.num_generations,
            ),
            "queue_maxsize": _int_env("SWE_ASYNC_QUEUE_MAXSIZE", 64),
            "logging_steps": 1,
            "report_to": "trackio",
            "run_name": f"swe-grpo-{model.split('/')[-1]}",
            "project": (
                os.environ.get("SWE_TRACKIO_PROJECT", "").strip()
                or os.environ.get("TRACKIO_PROJECT", "").strip()
                or "huggingface"
            ),
            "trackio_space_id": os.environ.get("TRACKIO_SPACE_ID", "").strip() or None,
            "log_completions": _bool_env("SWE_LOG_COMPLETIONS", False),
            "num_completions_to_print": _int_env(
                "SWE_LOG_COMPLETIONS_LIMIT",
                3,
            ),
            "accelerator_config": {
                "split_batches": True,
                "dispatch_batches": True,
            },
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

        trainer_args = AsyncGRPOConfig(**async_grpo_args)
        with _patched_async_grpo_model_loader(
            dtype=train_dtype,
            attn_implementation=train_attn_implementation,
            lora_config=lora_config,
        ):
            trainer = AsyncGRPOTrainer(
                model=model,
                reward_funcs=_noop_reward,
                train_dataset=dataset,
                processing_class=tokenizer,
                rollout_worker=worker,
                args=trainer_args,
            )

        trainable_params = _count_trainable_parameters(trainer.model)
        trainer_world_size = trainer.accelerator.num_processes
        if lora_config is not None:
            if trainer_world_size != 1:
                raise RuntimeError(
                    "SWE_LORA currently supports a single trainer process in this "
                    "example so merged adapter weights can be streamed to vLLM from "
                    "rank 0."
                )
            _patch_lora_weight_streaming(trainer)
            _patch_chunked_lm_head_for_wrapped_causal_lm(
                trainer.model,
                temperature=args.temperature,
            )
        single_gpu_param_limit = _single_gpu_trainable_param_limit()
        if trainer_world_size == 1 and trainable_params > single_gpu_param_limit:
            raise RuntimeError(
                "Unsupported full-parameter Async GRPO config for a single training GPU: "
                f"model={model} trainable_params={trainable_params:,} "
                f"limit={single_gpu_param_limit:,}. "
                "Use a smaller base model, add sharded training, or reduce trainable "
                "state before launching this example."
            )

        _log.info(
            "starting training: model=%s tasks=%d generations=%d temp=%.2f lr=%g optim=%s adapter=%s dtype=%s attn_impl=%s checkpointing=%s resume=%s trainable_params=%s world_size=%d",
            model,
            len(swe_tasks),
            args.num_generations,
            args.temperature,
            args.learning_rate,
            train_optim,
            adapter_summary,
            str(train_dtype).split(".")[-1],
            train_attn_implementation or "auto",
            checkpoint_enabled,
            resume_from_checkpoint or "none",
            f"{trainable_params:,}",
            trainer_world_size,
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

        if (
            checkpoint_enabled
            and hasattr(trainer, "push_to_hub")
            and trainer.is_world_process_zero()
        ):
            trainer.push_to_hub(
                commit_message=f"Final checkpoint at step {getattr(trainer.state, 'global_step', '?')}"
            )

        _log.info("done: step=%s", getattr(trainer.state, "global_step", "?"))
        return 0
    finally:
        if (
            control_plane is not None
            and server_loop is not None
            and server_thread is not None
        ):
            stop_interception_server(control_plane, server_loop, server_thread)


if __name__ == "__main__":
    raise SystemExit(main())
