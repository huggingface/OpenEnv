"""Custom rollout worker for Pi-in-sandbox SWE training.

Implements ``RolloutWorkerProtocol`` from TRL's ``AsyncGRPOTrainer``.

Architecture:
    Pi (sandbox) → InterceptionServer → this worker → vLLM /v1/chat/completions
                                                    ← chat response back to Pi

Pi drives the generation loop inside the sandbox.  This worker:

1. Dequeues each intercepted LLM request from Pi.
2. Forwards the intercepted request to vLLM ``/v1/chat/completions``.
3. Requests exact ``prompt_token_ids`` / ``completion_ids`` and per-token
   logprobs from vLLM.
4. Delivers the OpenAI-compatible chat response back to Pi.
5. Tracks multi-turn token sequences matching TRL's pattern:
   ``input_ids = initial_prompt_ids + [turn_ids + suffix_ids]*N``
   ``completion_mask = [0]*prompt + [1]*turn + [0]*suffix + ...``
6. On ``answer()``, bridges to host-side grading.
7. Assembles the final ``RolloutSample`` and pushes to ``rollout_buffer``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence, cast

import requests

try:
    from trl.chat_template_utils import add_response_schema, parse_response
except Exception:  # pragma: no cover - defensive for older TRL versions
    add_response_schema = None
    parse_response = None

try:
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )
    from vllm.utils.network_utils import get_ip, get_open_port
except Exception:  # pragma: no cover - optional at import time
    NCCLTrainerSendWeightsArgs = None
    NCCLWeightTransferEngine = None
    get_ip = None
    get_open_port = None

from mini_swe_env.models import SWETask

_log = logging.getLogger(__name__)


# ── Sample dataclass ───────────────────────────────────────────────────


@dataclass
class RolloutSample:
    """Matches the fields TRL's ``RolloutQueueDataset`` reads."""

    input_ids: list[int]
    completion_mask: list[int]
    old_log_probs: list[float]
    advantage: float
    model_version: int
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingRollout:
    """One rollout before group-relative advantage normalization."""

    input_ids: list[int]
    completion_mask: list[int]
    old_log_probs: list[float]
    reward: float
    model_version: int
    metrics: dict[str, Any] = field(default_factory=dict)


# ── Config ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkerConfig:
    max_inflight: int = 2
    max_rollout_attempts: int = 4
    num_generations: int = 4
    queue_maxsize: int = 64
    request_timeout_s: float = 600.0
    max_turns: int = 50
    max_model_len: int = 4096
    max_completion_tokens: int = 2048
    temperature: float = 1.0
    max_tool_message_chars: int = 6000
    min_tool_message_chars: int = 256
    max_assistant_message_chars: int = 4000
    min_assistant_message_chars: int = 256
    # After returning a terminal plain-text response (finish_reason=stop,
    # no tool_calls), wait briefly for a follow-up request before treating
    # the rollout as complete. This avoids 600s stalls when agent exit
    # detection is delayed on remote backends.
    post_response_grace_s: float = 10.0
    stop_on_idle_terminal_response: bool = True
    idle_backoff_s: float = 0.5
    failure_backoff_s: float = 30.0


# ── Worker ─────────────────────────────────────────────────────────────


class SWERolloutWorker:
    """Background rollout producer for Pi + InterceptionServer + vLLM.

    Implements ``RolloutWorkerProtocol`` so it plugs into
    ``AsyncGRPOTrainer(rollout_worker=...)``.
    """

    def __init__(
        self,
        *,
        session_factory: Any,
        tasks: Sequence[SWETask],
        tokenizer: Any,
        vllm_base_url: str,
        vllm_api_key: str,
        vllm_model: str,
        config: WorkerConfig | None = None,
    ) -> None:
        self._factory = session_factory
        self._tasks = list(tasks)
        self._tokenizer = tokenizer
        if add_response_schema is not None:
            try:
                self._tokenizer = add_response_schema(self._tokenizer)
            except Exception as exc:
                _log.debug("add_response_schema unavailable for tokenizer: %s", exc)
        self._vllm_base_url = vllm_base_url.rstrip("/")
        self._vllm_api_key = vllm_api_key
        self._vllm_model = vllm_model
        self._cfg = config or WorkerConfig()
        if not self._tasks:
            raise ValueError("SWERolloutWorker requires at least one SWE task")
        if self._cfg.num_generations < 2:
            raise ValueError(
                "WorkerConfig.num_generations must be >= 2 for valid GRPO grouping"
            )

        self.rollout_buffer: queue.Queue[RolloutSample] = queue.Queue(
            maxsize=self._cfg.queue_maxsize,
        )

        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.Lock()
        self._weight_sync_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._task_idx = 0
        self._model_version = 0
        self._started = False
        self._model_update_group: Any | None = None
        self._next_group_id = 0
        self._current_group_task: SWETask | None = None
        self._current_group_id: int | None = None
        self._current_group_model_version: int | None = None
        self._group_replica_idx = 0
        self._pending_groups: dict[int, list[PendingRollout]] = {}

        self._init_weight_transfer()

    # ── RolloutWorkerProtocol ──────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._stop.clear()
        for i in range(max(1, self._cfg.max_inflight)):
            t = threading.Thread(
                target=self._loop,
                args=(i,),
                daemon=True,
                name=f"swe-rollout-{i}",
            )
            t.start()
            self._threads.append(t)
        _log.info("worker started threads=%d", len(self._threads))

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)
        with self._lock:
            self._started = False
            self._threads = []
        self._destroy_model_update_group()

    def pause(self) -> None:
        self._pause.set()
        if self._model_update_group is None:
            return
        self._post_json(
            "/pause",
            timeout=60,
            params={"mode": "keep"},
        )

    def resume(self) -> None:
        if self._model_update_group is not None:
            self._post_json("/resume", timeout=60)
        self._pause.clear()

    def send_weights(self, iterator: Iterator[tuple[str, Any]]) -> None:
        # Materialize once so we can derive metadata and send the same
        # tensors through NCCL.
        items = list(iterator)
        if not items:
            return

        if self._model_update_group is None:
            _log.warning(
                "weight sync disabled: NCCL weight-transfer group not initialized"
            )
            return

        items = [(self._vllm_weight_name(name), tensor) for name, tensor in items]
        names = [name for name, _ in items]
        dtype_names = [
            str(getattr(tensor, "dtype", "float32")).split(".")[-1]
            for _, tensor in items
        ]
        shapes = [list(getattr(tensor, "shape", [])) for _, tensor in items]
        update_info = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "packed": True,
            "is_checkpoint_format": True,
        }

        with self._weight_sync_lock:
            post_error: list[Exception] = []

            def _post_update() -> None:
                try:
                    self._post_json(
                        "/update_weights",
                        timeout=1800,
                        json_body={"update_info": update_info},
                    )
                except Exception as exc:  # noqa: BLE001
                    post_error.append(exc)

            t_update = threading.Thread(target=_post_update, daemon=True)
            t_update.start()

            assert NCCLWeightTransferEngine is not None
            assert NCCLTrainerSendWeightsArgs is not None
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(items),
                trainer_args=NCCLTrainerSendWeightsArgs(
                    group=self._model_update_group,
                    packed=True,
                ),
            )

            t_update.join(timeout=1800)
            if t_update.is_alive():
                raise TimeoutError("Timed out waiting for vLLM /update_weights")
            if post_error:
                raise RuntimeError(
                    f"vLLM /update_weights failed: {post_error[0]}"
                ) from post_error[0]

    def update_model_version(self, version: int) -> None:
        with self._lock:
            self._model_version = version

    def _vllm_weight_name(self, name: str) -> str:
        """Map trainer-side text model names to vLLM's served module names."""
        model_id = self._vllm_model.lower().replace("_", "")
        if "qwen3.5" in model_id or "qwen35" in model_id:
            if name.startswith("model."):
                return f"language_model.{name}"
            if name.startswith("lm_head."):
                return f"language_model.{name}"
        return name

    def _post_json(
        self,
        path: str,
        *,
        timeout: float,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> requests.Response:
        response = requests.post(
            f"{self._vllm_base_url}{path}",
            headers={"Authorization": f"Bearer {self._vllm_api_key}"},
            json=json_body,
            params=params,
            timeout=timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"{path} returned {response.status_code}: {response.text[:400]}"
            )
        return response

    def _init_weight_transfer(self) -> None:
        if os.environ.get("SWE_DISABLE_WEIGHT_TRANSFER", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _log.warning("weight sync disabled by SWE_DISABLE_WEIGHT_TRANSFER")
            return

        if (
            NCCLWeightTransferEngine is None
            or NCCLTrainerSendWeightsArgs is None
            or get_ip is None
            or get_open_port is None
        ):
            _log.warning(
                "vLLM NCCL weight-transfer modules unavailable; disabling sync"
            )
            return

        response = requests.get(
            f"{self._vllm_base_url}/get_world_size",
            headers={"Authorization": f"Bearer {self._vllm_api_key}"},
            timeout=10,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "vLLM weight sync requires /get_world_size endpoint; "
                "start vLLM with VLLM_SERVER_DEV_MODE=1 and "
                '--weight-transfer-config \'{"backend":"nccl"}\''
            )

        inference_world_size = int(response.json()["world_size"])
        world_size = inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()

        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }

        post_error: list[Exception] = []

        def _post_init() -> None:
            try:
                self._post_json(
                    "/init_weight_transfer_engine",
                    timeout=120,
                    json_body={"init_info": init_info},
                )
            except Exception as exc:  # noqa: BLE001
                post_error.append(exc)

        t_init = threading.Thread(target=_post_init, daemon=True)
        t_init.start()
        self._model_update_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        t_init.join(timeout=120)
        if t_init.is_alive():
            raise TimeoutError("Timed out waiting for vLLM init_weight_transfer_engine")
        if post_error:
            raise RuntimeError(
                f"vLLM init_weight_transfer_engine failed: {post_error[0]}"
            ) from post_error[0]

        _log.info("initialized NCCL weight-transfer group with vLLM")

    def _destroy_model_update_group(self) -> None:
        group = self._model_update_group
        if group is None:
            return
        try:
            group.group.store = None
            group.group.socket = None
        except Exception:
            pass
        self._model_update_group = None

    # ── Internal ───────────────────────────────────────────────────

    def _next_rollout_assignment(self) -> tuple[SWETask, int, int, int]:
        with self._lock:
            if self._group_replica_idx == 0:
                self._current_group_task = self._tasks[self._task_idx % len(self._tasks)]
                self._task_idx += 1
                self._current_group_id = self._next_group_id
                self._current_group_model_version = self._model_version
                self._next_group_id += 1

            assert self._current_group_task is not None
            assert self._current_group_id is not None
            assert self._current_group_model_version is not None

            task = self._current_group_task
            group_id = self._current_group_id
            model_version = self._current_group_model_version
            replica_idx = self._group_replica_idx

            self._group_replica_idx += 1
            if self._group_replica_idx >= self._cfg.num_generations:
                self._group_replica_idx = 0
                self._current_group_task = None
                self._current_group_id = None
                self._current_group_model_version = None

            return task, group_id, replica_idx, model_version

    def _model_ver(self) -> int:
        with self._lock:
            return self._model_version

    def _collect_group_sample(
        self,
        *,
        group_id: int,
        rollout: PendingRollout,
    ) -> list[RolloutSample] | None:
        with self._lock:
            bucket = self._pending_groups.setdefault(group_id, [])
            bucket.append(rollout)
            if len(bucket) < self._cfg.num_generations:
                return None
            if len(bucket) > self._cfg.num_generations:
                raise RuntimeError(
                    f"received too many rollouts for group {group_id}: "
                    f"{len(bucket)} > {self._cfg.num_generations}"
                )
            group_rollouts = list(bucket)
            del self._pending_groups[group_id]

        return self._finalize_group_rollouts(
            group_id=group_id,
            rollouts=group_rollouts,
        )

    def _finalize_group_rollouts(
        self,
        *,
        group_id: int,
        rollouts: Sequence[PendingRollout],
    ) -> list[RolloutSample]:
        rewards = [float(rollout.reward) for rollout in rollouts]
        advantages, reward_mean, reward_std = _compute_group_advantages(rewards)
        _log.info(
            "rollout group complete: group_id=%d reward_mean=%.4f reward_std=%.4f rewards=%s",
            group_id,
            reward_mean,
            reward_std,
            [round(reward, 4) for reward in rewards],
        )

        samples: list[RolloutSample] = []
        for rollout, advantage in zip(rollouts, advantages, strict=True):
            metrics = dict(rollout.metrics)
            metrics["reward"] = float(rollout.reward)
            metrics["reward_mean"] = reward_mean
            metrics["reward_std"] = reward_std
            metrics["group_size"] = float(len(rollouts))
            samples.append(
                RolloutSample(
                    input_ids=rollout.input_ids,
                    completion_mask=rollout.completion_mask,
                    old_log_probs=rollout.old_log_probs,
                    advantage=advantage,
                    model_version=rollout.model_version,
                    metrics=metrics,
                )
            )
        return samples

    def _loop(self, idx: int) -> None:
        while not self._stop.is_set():
            while self._pause.is_set() and not self._stop.is_set():
                time.sleep(0.05)
            if self._stop.is_set():
                return

            task, group_id, replica_idx, model_version = self._next_rollout_assignment()
            eid = f"swe-{idx}-g{group_id}-r{replica_idx}-{uuid.uuid4().hex[:8]}"
            rollout_t0 = time.time()
            failed = False
            sample: PendingRollout | None = None
            last_exc: Exception | None = None

            for attempt in range(1, self._cfg.max_rollout_attempts + 1):
                try:
                    sample = asyncio.run(
                        self._rollout(task, eid, model_version=model_version)
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    retriable = (
                        attempt < self._cfg.max_rollout_attempts
                        and _is_retriable_rollout_error(exc)
                    )
                    if retriable:
                        _log.warning(
                            "rollout failed worker=%d id=%s attempt=%d/%d; retrying: %s",
                            idx,
                            task.instance_id,
                            attempt,
                            self._cfg.max_rollout_attempts,
                            exc,
                        )
                        time.sleep(self._cfg.failure_backoff_s * attempt)
                        continue

                    _log.exception(
                        "rollout failed worker=%d id=%s attempts=%d/%d",
                        idx,
                        task.instance_id,
                        attempt,
                        self._cfg.max_rollout_attempts,
                    )
                    sample = self._build_failed_rollout(
                        task=task,
                        elapsed_s=time.time() - rollout_t0,
                        exc=exc,
                        model_version=model_version,
                    )
                    failed = True
                    break

            if sample is None:
                if last_exc is None:
                    time.sleep(self._cfg.idle_backoff_s)
                    continue
                raise RuntimeError(
                    f"rollout loop exhausted without sample for {task.instance_id}"
                ) from last_exc

            if sample is None:
                time.sleep(self._cfg.idle_backoff_s)
                continue

            group_samples = self._collect_group_sample(
                group_id=group_id,
                rollout=sample,
            )
            if group_samples is not None:
                for group_sample in group_samples:
                    try:
                        self.rollout_buffer.put(group_sample, timeout=2.0)
                    except queue.Full:
                        _log.warning("queue full, dropping group %d", group_id)
                        break

            if failed:
                time.sleep(self._cfg.failure_backoff_s)

    def _build_failed_rollout(
        self,
        *,
        task: SWETask,
        elapsed_s: float,
        exc: Exception,
        model_version: int,
    ) -> PendingRollout:
        """Return a zero-reward rollout when infrastructure fails before rollout.

        Remote sandbox providers can transiently refuse new jobs. Producing a
        neutral sample keeps distributed trainer ranks moving together instead
        of letting nonzero ranks block in Accelerate's dataloader broadcast.
        """
        prompt_ids = self._render_prompt_ids(
            [{"role": "user", "content": task.instruction}],
            None,
        )
        completion_id = (
            getattr(self._tokenizer, "eos_token_id", None)
            or getattr(self._tokenizer, "pad_token_id", None)
            or 0
        )
        input_ids = [*prompt_ids, int(completion_id)]
        completion_mask = [0] * len(prompt_ids) + [1]
        old_log_probs = [0.0] * len(input_ids)
        exc_name = type(exc).__name__
        return PendingRollout(
            input_ids=input_ids,
            completion_mask=completion_mask,
            old_log_probs=old_log_probs,
            reward=0.0,
            model_version=model_version,
            metrics={
                "reward": 0.0,
                "turns": 0.0,
                "answer_called": 0.0,
                "terminal_idle_stop": 0.0,
                "wall_s": round(elapsed_s, 3),
                "n_tokens": float(len(input_ids)),
                "rollout_error": 1.0,
                "sandbox_create_error": float(
                    "sandbox" in exc_name.lower()
                    or "sandbox" in str(exc).lower()
                ),
            },
        )

    # ── Single rollout ─────────────────────────────────────────────

    async def _rollout(
        self,
        task: SWETask,
        episode_id: str,
        *,
        model_version: int,
    ) -> PendingRollout | None:
        session = self._factory.create(task=task, episode_id=episode_id)

        # Accumulate the full token sequence across turns, matching TRL's
        # _generate_one pattern:
        #   input_ids      = initial_prompt_ids + turn1_ids + suffix1_ids + turn2_ids + ...
        #   completion_mask = [0]*prompt         + [1]*turn1 + [0]*suffix1 + [1]*turn2 + ...
        #   old_log_probs  = [0.0]*prompt        + lp1       + [0.0]*suf1  + lp2       + ...
        all_ids: list[int] = []
        all_mask: list[int] = []
        all_lps: list[float] = []

        initial_prompt_ids: list[int] | None = None
        prev_prompt_ids: list[int] | None = None

        turns = 0
        answer_called = False
        pending_intercept: dict[str, Any] | None = None
        rollout_stop_reason = "max_turns"
        t0 = time.time()

        try:
            while turns < self._cfg.max_turns and not self._stop.is_set():
                if pending_intercept is not None:
                    intercept = pending_intercept
                    pending_intercept = None
                else:
                    intercept = await session.next_request(
                        timeout_s=self._cfg.request_timeout_s,
                    )
                if intercept is None:
                    rollout_stop_reason = "agent_exit_detected"
                    break

                # ── Tokenize this turn's full prompt ──────────────
                messages = _get_messages(intercept)
                tools = _get_tools(intercept)
                (
                    current_prompt_ids,
                    turn_ids,
                    turn_lps,
                    chat_resp,
                    _finish_reason,
                ) = self._generate(
                    intercept=intercept,
                    messages=messages,
                    tools=tools,
                )
                if not current_prompt_ids:
                    current_prompt_ids = self._render_prompt_ids(messages, tools)

                if initial_prompt_ids is None:
                    # First turn: the entire prompt is non-completion tokens.
                    initial_prompt_ids = current_prompt_ids
                    all_ids.extend(current_prompt_ids)
                    all_mask.extend([0] * len(current_prompt_ids))
                    all_lps.extend([0.0] * len(current_prompt_ids))
                elif prev_prompt_ids is not None:
                    # Subsequent turns: the delta between prev generation end
                    # and this turn's prompt is the tool-result suffix.
                    # prev_prompt_ids + prev_turn_ids = end of last generation
                    # current_prompt_ids = prev_prompt_ids + prev_turn_ids + suffix_ids
                    prev_len = len(prev_prompt_ids)
                    suffix_ids = current_prompt_ids[prev_len:]
                    all_ids.extend(suffix_ids)
                    all_mask.extend([0] * len(suffix_ids))
                    all_lps.extend([0.0] * len(suffix_ids))

                turns += 1

                all_ids.extend(turn_ids)
                all_mask.extend([1] * len(turn_ids))
                all_lps.extend(turn_lps)

                # For next turn's suffix computation:
                prev_prompt_ids = current_prompt_ids + turn_ids

                # ── Check for answer tool call ────────────────────
                if _has_answer_call(chat_resp):
                    answer_called = True

                await session.deliver(intercept, chat_resp)

                # Host-side answer handler marks this as soon as Pi executes answer().
                if bool(getattr(session, "answer_called", False)):
                    answer_called = True

                if answer_called:
                    rollout_stop_reason = "answer_called"
                    break

                # Semantic completion fast-path (especially important on HF
                # Sandbox, where process-exit detection can be delayed).
                if (
                    self._cfg.stop_on_idle_terminal_response
                    and self._cfg.post_response_grace_s > 0
                    and _is_terminal_non_tool_response(chat_resp)
                ):
                    try:
                        maybe_next = await session.next_request(
                            timeout_s=self._cfg.post_response_grace_s,
                        )
                    except TimeoutError:
                        rollout_stop_reason = "idle_after_terminal_stop"
                        _log.info(
                            "rollout terminal idle-stop: instance_id=%s turns=%d",
                            task.instance_id,
                            turns,
                        )
                        break

                    if maybe_next is None:
                        rollout_stop_reason = "agent_exit_after_terminal_stop"
                        break
                    pending_intercept = maybe_next

            # ── Reward ────────────────────────────────────────────
            vr = session.verify(transcript=[])
            reward = float(getattr(vr, "env_reward", 0.0) or 0.0)

            if not all_lps:
                pad = getattr(self._tokenizer, "pad_token_id", 0) or 0
                all_ids = [pad]
                all_mask = [1]
                all_lps = [0.0]

            return PendingRollout(
                input_ids=all_ids,
                completion_mask=all_mask,
                old_log_probs=all_lps,
                reward=reward,
                model_version=model_version,
                metrics={
                    "reward": reward,
                    "turns": float(turns),
                    "answer_called": float(answer_called),
                    "terminal_idle_stop": float(
                        rollout_stop_reason
                        in {
                            "idle_after_terminal_stop",
                            "agent_exit_after_terminal_stop",
                        }
                    ),
                    "wall_s": round(time.time() - t0, 3),
                    "n_tokens": float(len(all_ids)),
                },
            )
        finally:
            session.close()

    # ── vLLM call ─────────────────────────────────────────────────

    def _render_prompt_ids(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "return_dict": False,
        }
        # ``tools=[]`` can trigger unwanted boilerplate in some templates.
        if tools:
            kwargs["tools"] = tools
        try:
            ids = self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            ids = self._tokenizer.apply_chat_template(messages, **kwargs)
        return cast(list[int], ids)

    def _generate(
        self,
        *,
        intercept: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[int], list[int], list[float], dict[str, Any], str | None]:
        """POST /v1/chat/completions and return prompt/output tokens."""
        (
            messages,
            rendered_prompt_ids,
        ) = _fit_messages_to_context_window(
            messages=messages,
            tools=tools,
            render_prompt_ids=self._render_prompt_ids,
            requested_completion_tokens=self._cfg.max_completion_tokens,
            max_model_len=self._cfg.max_model_len,
            max_tool_message_chars=self._cfg.max_tool_message_chars,
            min_tool_message_chars=self._cfg.min_tool_message_chars,
            max_assistant_message_chars=self._cfg.max_assistant_message_chars,
            min_assistant_message_chars=self._cfg.min_assistant_message_chars,
        )
        raw_body = intercept.get("body")
        body = dict(raw_body) if isinstance(raw_body, dict) else {}
        body["model"] = self._vllm_model
        body["messages"] = messages
        if tools is not None:
            body["tools"] = tools
        else:
            body.pop("tools", None)
        body.pop("stream", None)
        body.pop("stream_options", None)
        body.pop("max_tokens", None)
        body["max_completion_tokens"] = _clamp_max_completion_tokens(
            prompt_len=len(rendered_prompt_ids),
            requested=self._cfg.max_completion_tokens,
            max_model_len=self._cfg.max_model_len,
        )
        body["temperature"] = self._cfg.temperature
        body["n"] = 1
        body["logprobs"] = True
        body["top_logprobs"] = 0
        body["return_token_ids"] = True

        if os.environ.get("SWE_LOG_PROMPT_TOKENS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _log.info(
                "rollout prompt window: prompt_tokens=%d max_completion_tokens=%d tools=%d",
                len(rendered_prompt_ids),
                int(body["max_completion_tokens"]),
                len(tools or []),
            )

        chat_template_kwargs = body.get("chat_template_kwargs")
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        model_id = self._vllm_model.lower().replace("_", "")
        if (
            ("qwen3" in model_id or "deepseek" in model_id)
            and "enable_thinking" not in chat_template_kwargs
        ):
            chat_template_kwargs["enable_thinking"] = False
        if chat_template_kwargs:
            body["chat_template_kwargs"] = chat_template_kwargs

        if tools and not body.get("tool_choice"):
            body["tool_choice"] = "auto"

        for attempt in range(2):
            resp = requests.post(
                f"{self._vllm_base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._vllm_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self._cfg.request_timeout_s,
            )
            if resp.status_code == 200:
                break

            retry_budget = _retry_completion_tokens_from_context_error(resp.text)
            if attempt == 0 and retry_budget is not None:
                requested = int(body.get("max_completion_tokens") or 0)
                clamped_tokens, max_model_len, prompt_tokens = retry_budget
                if clamped_tokens < requested:
                    _log.info(
                        "clamped max_completion_tokens from %d to %d "
                        "(prompt_tokens=%d max_model_len=%d)",
                        requested,
                        clamped_tokens,
                        prompt_tokens,
                        max_model_len,
                    )
                    body["max_completion_tokens"] = clamped_tokens
                    continue

            raise RuntimeError(f"vllm {resp.status_code}: {resp.text[:400]}")

        payload = resp.json()
        choice = payload["choices"][0]
        turn_ids = _coerce_token_ids(choice.get("token_ids"))
        choice["message"] = _normalize_chat_choice_message(
            tokenizer=self._tokenizer,
            choice=choice,
            completion_ids=turn_ids,
        )
        chat_resp = dict(payload)
        chat_resp["choices"] = [choice]
        return (
            _extract_prompt_token_ids(payload) or rendered_prompt_ids,
            turn_ids,
            _extract_chat_choice_logprobs(choice, expected_len=len(turn_ids)),
            chat_resp,
            choice.get("finish_reason"),
        )


def _compute_group_advantages(
    rewards: Sequence[float],
    *,
    eps: float = 1e-8,
) -> tuple[list[float], float, float]:
    """Return z-scored group advantages, mean reward, and reward stddev."""
    if not rewards:
        raise ValueError("rewards must not be empty")

    reward_mean = sum(rewards) / len(rewards)
    reward_var = sum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards)
    reward_std = math.sqrt(reward_var)
    denom = reward_std + eps
    advantages = [float((reward - reward_mean) / denom) for reward in rewards]
    return advantages, float(reward_mean), float(reward_std)


# ── Helpers ────────────────────────────────────────────────────────────


def _get_messages(intercept: dict[str, Any]) -> list[dict[str, Any]]:
    msgs = intercept.get("messages")
    if isinstance(msgs, list) and msgs:
        return msgs
    body = intercept.get("body") or {}
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        return msgs
    raise RuntimeError("intercept has no messages")


def _get_tools(intercept: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = intercept.get("tools")
    if isinstance(tools, list):
        return [t for t in tools if isinstance(t, dict)]
    body = intercept.get("body") or {}
    tools = body.get("tools")
    if isinstance(tools, list):
        return [t for t in tools if isinstance(t, dict)]
    return None


def _coerce_token_ids(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    token_ids: list[int] = []
    for token_id in raw:
        try:
            token_ids.append(int(token_id))
        except (TypeError, ValueError):
            return []
    return token_ids


def _extract_prompt_token_ids(payload: dict[str, Any]) -> list[int]:
    return _coerce_token_ids(payload.get("prompt_token_ids"))


def _extract_chat_choice_logprobs(
    choice: dict[str, Any],
    *,
    expected_len: int,
) -> list[float]:
    content = (choice.get("logprobs") or {}).get("content") or []
    values: list[float] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                values.append(0.0)
                continue
            raw = item.get("logprob")
            values.append(float(raw) if isinstance(raw, (int, float)) else 0.0)

    if len(values) < expected_len:
        values.extend([0.0] * (expected_len - len(values)))
    return values[:expected_len]


def _clamp_max_completion_tokens(
    *,
    prompt_len: int,
    requested: int,
    max_model_len: int,
    safety_margin: int = 16,
) -> int:
    """Clamp completion tokens so prompt + generation fit in vLLM context."""
    available = max_model_len - prompt_len - safety_margin
    return max(1, min(int(requested), int(available)))


_TRUNCATION_MARKER = "\n...[truncated]...\n"
_OMITTED_TOOL_OUTPUT_MARKER = "[tool output omitted]"
_OMITTED_ASSISTANT_TEXT_MARKER = "[omitted]"


def _truncate_text_middle(
    text: str,
    *,
    max_chars: int,
    marker: str = _TRUNCATION_MARKER,
) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False

    usable = max_chars - len(marker)
    if usable <= 8:
        return marker[: max(1, max_chars)], True

    head = max(1, int(usable * 0.75))
    tail = max(1, usable - head)
    return text[:head].rstrip() + marker + text[-tail:].lstrip(), True


def _truncate_messages_for_prompt_budget(
    messages: Sequence[dict[str, Any]],
    *,
    max_tool_message_chars: int,
    max_assistant_message_chars: int,
) -> tuple[list[dict[str, Any]], int, int]:
    truncated_messages: list[dict[str, Any]] = []
    tool_truncations = 0
    assistant_truncations = 0

    for message in messages:
        updated = dict(message)
        content = updated.get("content")
        if not isinstance(content, str):
            truncated_messages.append(updated)
            continue

        role = str(updated.get("role") or "")
        if role == "tool":
            content, changed = _truncate_text_middle(
                content,
                max_chars=max_tool_message_chars,
            )
            if changed:
                tool_truncations += 1
        elif role == "assistant":
            content, changed = _truncate_text_middle(
                content,
                max_chars=max_assistant_message_chars,
            )
            if changed:
                assistant_truncations += 1

        updated["content"] = content
        truncated_messages.append(updated)

    return truncated_messages, tool_truncations, assistant_truncations


def _replace_oldest_message_content(
    messages: Sequence[dict[str, Any]],
    *,
    role: str,
    replacement: str,
) -> tuple[list[dict[str, Any]], bool]:
    updated_messages = [dict(message) for message in messages]
    for idx, message in enumerate(updated_messages):
        if message.get("role") != role:
            continue
        content = message.get("content")
        if not isinstance(content, str) or content == replacement:
            continue
        if len(replacement) >= len(content):
            continue
        message["content"] = replacement
        updated_messages[idx] = message
        return updated_messages, True
    return updated_messages, False


def _fit_messages_to_context_window(
    *,
    messages: Sequence[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    render_prompt_ids: Callable[
        [list[dict[str, Any]], list[dict[str, Any]] | None],
        list[int],
    ],
    requested_completion_tokens: int,
    max_model_len: int,
    max_tool_message_chars: int,
    min_tool_message_chars: int,
    max_assistant_message_chars: int,
    min_assistant_message_chars: int,
    safety_margin: int = 16,
) -> tuple[list[dict[str, Any]], list[int]]:
    prompt_budget = max(
        1,
        max_model_len - max(1, int(requested_completion_tokens)) - safety_margin,
    )
    tool_char_budget = max(1, int(max_tool_message_chars))
    assistant_char_budget = max(1, int(max_assistant_message_chars))
    min_tool_chars = max(1, int(min_tool_message_chars))
    min_assistant_chars = max(1, int(min_assistant_message_chars))

    base_messages = [dict(message) for message in messages]
    prepared_messages = base_messages
    prompt_ids = render_prompt_ids(prepared_messages, tools)
    tool_truncations = 0
    assistant_truncations = 0

    while True:
        prepared_messages, tool_truncations, assistant_truncations = (
            _truncate_messages_for_prompt_budget(
                base_messages,
                max_tool_message_chars=tool_char_budget,
                max_assistant_message_chars=assistant_char_budget,
            )
        )
        prompt_ids = render_prompt_ids(prepared_messages, tools)
        if len(prompt_ids) <= prompt_budget:
            break
        if tool_char_budget > min_tool_chars:
            tool_char_budget = max(min_tool_chars, tool_char_budget // 2)
            continue
        if assistant_char_budget > min_assistant_chars:
            assistant_char_budget = max(
                min_assistant_chars,
                assistant_char_budget // 2,
            )
            continue
        break

    omitted_tool_messages = 0
    omitted_assistant_messages = 0
    while len(prompt_ids) > prompt_budget:
        prepared_messages, changed = _replace_oldest_message_content(
            prepared_messages,
            role="tool",
            replacement=_OMITTED_TOOL_OUTPUT_MARKER,
        )
        if changed:
            omitted_tool_messages += 1
            prompt_ids = render_prompt_ids(prepared_messages, tools)
            continue

        prepared_messages, changed = _replace_oldest_message_content(
            prepared_messages,
            role="assistant",
            replacement=_OMITTED_ASSISTANT_TEXT_MARKER,
        )
        if not changed:
            break
        omitted_assistant_messages += 1
        prompt_ids = render_prompt_ids(prepared_messages, tools)

    if (
        tool_truncations
        or assistant_truncations
        or omitted_tool_messages
        or omitted_assistant_messages
    ):
        _log.info(
            "trimmed intercepted prompt: prompt_tokens=%d/%d tool_truncations=%d "
            "assistant_truncations=%d omitted_tool_messages=%d "
            "omitted_assistant_messages=%d",
            len(prompt_ids),
            prompt_budget,
            tool_truncations,
            assistant_truncations,
            omitted_tool_messages,
            omitted_assistant_messages,
        )

    return prepared_messages, prompt_ids


_CONTEXT_LIMIT_RE = re.compile(
    r"maximum context length is (?P<max_model_len>[\d,]+) tokens.*?"
    r"requested (?P<requested>[\d,]+) output tokens and your prompt contains "
    r"at least (?P<prompt_tokens>[\d,]+) input tokens",
    flags=re.IGNORECASE | re.DOTALL,
)


def _retry_completion_tokens_from_context_error(
    error_text: str,
    *,
    safety_margin: int = 16,
) -> tuple[int, int, int] | None:
    """Return a smaller completion budget after a vLLM context-window error."""
    match = _CONTEXT_LIMIT_RE.search(error_text or "")
    if match is None:
        return None

    max_model_len = int(match.group("max_model_len").replace(",", ""))
    prompt_tokens = int(match.group("prompt_tokens").replace(",", ""))
    clamped_tokens = max(1, max_model_len - prompt_tokens - safety_margin)
    return clamped_tokens, max_model_len, prompt_tokens


def _is_context_window_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "maximum context length" in text
        or "parameter=input_tokens" in text
        or "input tokens" in text
        and "output tokens" in text
    )


def _is_retriable_rollout_error(exc: Exception) -> bool:
    if _is_context_window_error(exc):
        return False

    text = str(exc or "").lower()
    retriable_markers = (
        "429",
        "too many requests",
        "timed out",
        "timeout",
        "connection reset",
        "temporarily unavailable",
        "tunnel failed",
        "sandbox",
    )
    return any(marker in text for marker in retriable_markers)


def _normalize_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_tool_calls, list):
        return normalized

    for raw_call in raw_tool_calls:
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments or {}, ensure_ascii=False)

        normalized.append(
            {
                "id": str(raw_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_str,
                },
            }
        )

    return normalized


_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.DOTALL,
)


def _extract_xml_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse Qwen-style XML tool-call blocks from assistant text.

    vLLM's Qwen3 XML parser can return tool calls only inside
    ``message.content``:

        <tool_call>
        {"name": "answer", "arguments": {}}
        </tool_call>

    When that happens we need to recover structured ``tool_calls`` before
    sending the response back to Pi, otherwise the harness treats the reply as
    plain terminal text and the rollout dies after one turn.
    """
    if not text:
        return "", []

    tool_calls: list[dict[str, Any]] = []
    cursor = 0
    content_parts: list[str] = []

    for match in _TOOL_CALL_XML_RE.finditer(text):
        start, end = match.span()
        if start > cursor:
            content_parts.append(text[cursor:start])

        raw_block = match.group(1).strip()
        parsed_block: Any
        try:
            parsed_block = json.loads(raw_block)
        except json.JSONDecodeError:
            content_parts.append(text[start:end])
            cursor = end
            continue

        blocks = parsed_block if isinstance(parsed_block, list) else [parsed_block]
        parsed_any = False
        for block in blocks:
            if not isinstance(block, dict):
                continue
            name = block.get("name")
            if not isinstance(name, str) or not name:
                continue
            parsed_any = True
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            block.get("arguments") or {},
                            ensure_ascii=False,
                        ),
                    },
                }
            )

        if not parsed_any:
            content_parts.append(text[start:end])
        cursor = end

    if cursor < len(text):
        content_parts.append(text[cursor:])

    return "".join(content_parts).strip(), tool_calls


def _parse_assistant_message(
    *,
    tokenizer: Any,
    completion_ids: list[int],
    fallback_text: str,
) -> dict[str, Any]:
    parsed: dict[str, Any] | None = None
    if parse_response is not None:
        try:
            maybe = parse_response(tokenizer, completion_ids)
            if isinstance(maybe, dict):
                parsed = maybe
        except Exception:
            parsed = None

    if parsed is None:
        return {"role": "assistant", "content": fallback_text}

    content = parsed.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    tool_calls = _normalize_tool_calls(parsed.get("tool_calls"))
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _normalize_chat_choice_message(
    *,
    tokenizer: Any,
    choice: dict[str, Any],
    completion_ids: list[int],
) -> dict[str, Any]:
    raw_message = choice.get("message")
    if isinstance(raw_message, dict):
        message = dict(raw_message)
    else:
        message = {"role": "assistant", "content": ""}

    fallback_text = message.get("content")
    if not isinstance(fallback_text, str):
        fallback_text = ""

    parsed = _parse_assistant_message(
        tokenizer=tokenizer,
        completion_ids=completion_ids,
        fallback_text=fallback_text,
    )

    if not isinstance(message.get("role"), str):
        message["role"] = "assistant"
    if not isinstance(message.get("content"), str):
        message["content"] = parsed.get("content", "")
    if not (message.get("tool_calls") or []):
        tool_calls = parsed.get("tool_calls") or []
        if not tool_calls and fallback_text:
            text_content, xml_tool_calls = _extract_xml_tool_calls(fallback_text)
            if xml_tool_calls:
                tool_calls = xml_tool_calls
                message["content"] = text_content
        if tool_calls:
            message["tool_calls"] = tool_calls
    return message


def _make_chat_response(
    assistant_message: dict[str, Any],
    model: str,
    *,
    finish_reason: str | None = "stop",
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": assistant_message,
                "finish_reason": finish_reason or "stop",
            }
        ],
    }


def _is_terminal_non_tool_response(resp: dict[str, Any]) -> bool:
    for choice in resp.get("choices") or []:
        message = (choice or {}).get("message") or {}
        if message.get("tool_calls") or []:
            return False
        if (choice or {}).get("finish_reason") == "stop":
            return True
    return False


def _has_answer_call(resp: dict[str, Any]) -> bool:
    for choice in resp.get("choices") or []:
        for tc in ((choice or {}).get("message") or {}).get("tool_calls") or []:
            if ((tc or {}).get("function") or {}).get("name") == "answer":
                return True
    return False
