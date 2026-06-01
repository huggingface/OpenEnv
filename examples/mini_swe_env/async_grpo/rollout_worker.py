"""Custom rollout worker for Pi-in-sandbox SWE training.

Implements ``RolloutWorkerProtocol`` from TRL's ``AsyncGRPOTrainer``.

Architecture:
    Pi (sandbox) → InterceptionServer → this worker → vLLM /v1/completions
                                                    ← chat response back to Pi

Pi drives the generation loop inside the sandbox.  This worker:

1. Dequeues each intercepted LLM request from Pi.
2. Tokenizes the messages with ``apply_chat_template``.
3. Calls vLLM ``/v1/completions`` with ``prompt=token_ids``,
   ``return_token_ids=True``, ``logprobs=0`` — same as TRL's own
   ``AsyncRolloutWorker._generate_one_turn``.
4. Gets exact ``completion_ids`` and ``completion_logprobs`` from vLLM.
5. Wraps the completion text as a chat response and delivers it back to Pi.
6. Tracks multi-turn token sequences matching TRL's pattern:
   ``input_ids = initial_prompt_ids + [turn_ids + suffix_ids]*N``
   ``completion_mask = [0]*prompt + [1]*turn + [0]*suffix + ...``
7. On ``answer()``, bridges to host-side grading.
8. Assembles the final ``RolloutSample`` and pushes to ``rollout_buffer``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence, cast

import requests

try:
    from trl.chat_template_utils import (
        add_response_schema,
        get_training_chat_template,
        is_chat_template_prefix_preserving,
        parse_response,
    )
except Exception:  # pragma: no cover - defensive for older TRL versions
    add_response_schema = None
    get_training_chat_template = None
    is_chat_template_prefix_preserving = None
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
    """Matches the fields TRL's ``RolloutQueueDataset`` reads.

    Note: TRL's own RolloutSample also has ``prompt`` and ``completion``
    (message-level) but those are only used by TRL's built-in reward_funcs.
    We compute rewards via ``session.verify()`` so they're unnecessary.
    """

    input_ids: list[int]
    completion_mask: list[int]
    old_log_probs: list[float]
    advantage: float
    model_version: int
    metrics: dict[str, Any] = field(default_factory=dict)


# ── Config ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkerConfig:
    max_inflight: int = 2
    queue_maxsize: int = 64
    request_timeout_s: float = 600.0
    max_turns: int = 50
    max_completion_tokens: int = 2048
    temperature: float = 1.0
    # Number of rollouts per prompt for group-relative advantage
    # normalization (true GRPO). Polar used 16.
    num_generations: int = 16
    # After returning a terminal plain-text response (finish_reason=stop,
    # no tool_calls), wait briefly for a follow-up request before treating
    # the rollout as complete. This avoids 600s stalls when agent exit
    # detection is delayed on remote backends.
    post_response_grace_s: float = 10.0
    stop_on_idle_terminal_response: bool = True
    idle_backoff_s: float = 0.5
    # vLLM max_model_len for pre-generation guard.  If the prompt +
    # max_completion_tokens exceeds this, the rollout terminates early
    # with reward=0 rather than crashing the engine.  Set to match
    # vLLM's --max-model-len.
    max_model_len: int = 40960
    # Number of context overflow failures before a task is blacklisted
    # for the remainder of the training run.  Prevents infinite retry
    # loops on tasks whose prompts inherently exceed the context window.
    max_context_overflow_per_task: int = 3
    # Message truncation settings for context window management.
    # Instead of crashing on overflow, progressively truncate long tool
    # outputs and assistant messages to fit within context budget.
    max_tool_message_chars: int = 6000
    min_tool_message_chars: int = 256
    max_assistant_message_chars: int = 4000
    min_assistant_message_chars: int = 256
    # Retry config for transient sandbox/network errors.
    max_rollout_attempts: int = 4
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

        # Use TRL's prefix-preserving training template if available.
        # The stock Qwen3.5 template is NOT prefix-preserving (it conditionally
        # renders <think> tags based on message position), which breaks our
        # suffix computation.  The training template fixes this.
        self._chat_template: str | None = None
        if get_training_chat_template is not None:
            try:
                self._chat_template = get_training_chat_template(self._tokenizer)
                if self._chat_template:
                    _log.info("using TRL prefix-preserving training chat template")
            except (ValueError, TypeError) as exc:
                _log.debug("get_training_chat_template failed: %s", exc)
        self._vllm_base_url = vllm_base_url.rstrip("/")
        self._vllm_api_key = vllm_api_key
        self._vllm_model = vllm_model
        self._cfg = config or WorkerConfig()

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

        # ── Group-relative advantage (GRPO) ──────────────────────
        # Accumulate rollout samples by task until num_generations are
        # collected, then normalize advantages within the group and push
        # all samples to the rollout buffer.
        self._group_lock = threading.Lock()
        # task_id → list of (RolloutSample with raw reward in .advantage)
        self._pending_groups: dict[str, list[RolloutSample]] = {}

        # ── Context overflow tracking ────────────────────────────
        # Track tasks that repeatedly exceed context to avoid infinite
        # retry loops.  Failed sessions get reward=0 and are excluded
        # from group normalization in _score_and_push_group.
        self._overflow_counts: dict[str, int] = {}
        self._overflow_lock = threading.Lock()

        # Prefix to prepend to trainer parameter names so they match vLLM's
        # model architecture.  For VLM models like Qwen3_5ForConditionalGeneration
        # served with --language-model-only, vLLM parameters are at
        # "language_model.model.layers.X" but the trainer (AutoModelForCausalLM)
        # produces "model.layers.X".  With is_checkpoint_format=True (vLLM 0.21+),
        # vLLM's load_weights() handles this remapping automatically — no manual
        # prefix is needed.

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

        names = [name for name, _ in items]

        # With is_checkpoint_format=True (vLLM 0.21+), vLLM's load_weights()
        # handles the name remapping internally (e.g. model.layers.X →
        # language_model.model.layers.X for VLM architectures like Qwen3.5).
        # Send HuggingFace checkpoint-format names as-is.

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
        }

        with self._weight_sync_lock:
            # vLLM 0.21+ four-phase protocol:
            #   start_weight_update(is_checkpoint_format=True) → update_weights → finish_weight_update
            # is_checkpoint_format=True tells vLLM to use model.load_weights() which
            # handles parameter name remapping (e.g. CausalLM → VLM prefixes).
            self._post_json(
                "/start_weight_update",
                timeout=60,
                json_body={"is_checkpoint_format": True},
            )

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

            # Finalize layerwise reload / quantization.
            self._post_json("/finish_weight_update", timeout=120)

    def update_model_version(self, version: int) -> None:
        with self._lock:
            self._model_version = version

    def check_health(self, stale_after_s: float) -> None:
        """Health check called by the trainer when the queue is empty.

        Required by TRL's RolloutWorkerProtocol. Can be used to detect
        stuck workers. Currently a no-op — our workers self-recover via
        retry logic and idle backoff.
        """
        pass

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

    def _next_task(self) -> SWETask:
        with self._lock:
            t = self._tasks[self._task_idx % len(self._tasks)]
            self._task_idx += 1
            return t

    def _model_ver(self) -> int:
        with self._lock:
            return self._model_version

    def _is_task_blacklisted(self, task_id: str) -> bool:
        """Check if a task has been blacklisted due to repeated context overflow."""
        with self._overflow_lock:
            return (
                self._overflow_counts.get(task_id, 0)
                >= self._cfg.max_context_overflow_per_task
            )

    def _record_context_overflow(self, task_id: str) -> None:
        """Record a context overflow for a task.  After max attempts,
        abandon the group with reward=0 for all pending samples."""
        abandon_group = False
        with self._overflow_lock:
            self._overflow_counts[task_id] = (
                self._overflow_counts.get(task_id, 0) + 1
            )
            count = self._overflow_counts[task_id]
            if count >= self._cfg.max_context_overflow_per_task:
                _log.warning(
                    "task %s blacklisted after %d context overflows — "
                    "abandoning group with reward=0",
                    task_id,
                    count,
                )
                abandon_group = True

        if abandon_group:
            # Flush the partial group: failed sessions get reward=0
            # and are excluded from group normalization.
            with self._group_lock:
                partial = self._pending_groups.pop(task_id, None)
            if partial:
                for sample in partial:
                    sample.advantage = 0.0
                    sample.metrics["context_overflow_abandoned"] = 1.0
                    try:
                        self.rollout_buffer.put(sample, timeout=1.0)
                    except queue.Full:
                        pass

    def _next_group_task(self) -> SWETask:
        """Pick a task that needs more rollouts to complete its group.

        Prioritizes tasks that already have partial groups (some rollouts
        done but < num_generations). Falls back to a fresh task.
        Skips tasks blacklisted due to repeated context overflow.
        """
        with self._group_lock:
            # Find a task with a partial group that needs more rollouts
            for task_id, samples in self._pending_groups.items():
                if len(samples) < self._cfg.num_generations:
                    if self._is_task_blacklisted(task_id):
                        continue
                    # Return the corresponding task
                    for t in self._tasks:
                        if t.instance_id == task_id:
                            return t
        # No partial groups — start a fresh task (skip blacklisted)
        for _ in range(len(self._tasks)):
            task = self._next_task()
            if not self._is_task_blacklisted(task.instance_id):
                return task
        # All tasks blacklisted — fatal (shouldn't happen with 293 tasks)
        return self._next_task()

    def _submit_to_group(self, task: SWETask, sample: RolloutSample) -> None:
        """Add a completed rollout to its task's group.

        When the group reaches num_generations, normalize advantages
        (GRPO-style) and push all samples to the rollout buffer.
        """
        completed_group: list[RolloutSample] | None = None

        with self._group_lock:
            group = self._pending_groups.setdefault(task.instance_id, [])
            group.append(sample)

            if len(group) >= self._cfg.num_generations:
                completed_group = self._pending_groups.pop(task.instance_id)

        if completed_group is not None:
            self._score_and_push_group(task.instance_id, completed_group)

    def _score_and_push_group(
        self, task_id: str, group: list[RolloutSample]
    ) -> None:
        """Compute group-relative advantages (GRPO) and push to buffer.

        Failed sessions (context_overflow) are excluded from the group
        mean/std and get advantage=0, matching Polar's reward_post_process
        which zeros FAILED/ABORTED trajectories.
        """
        rewards = [s.advantage for s in group]  # raw rewards stored here
        n = len(rewards)

        # Exclude failed sessions from the baseline computation.
        valid_mask = [
            s.metrics.get("context_overflow", 0.0) == 0.0 for s in group
        ]
        valid_rewards = [r for r, v in zip(rewards, valid_mask) if v]

        if not valid_rewards:
            # All sessions failed — no signal.
            for sample in group:
                sample.advantage = 0.0
                sample.metrics["group_reward_mean"] = 0.0
                sample.metrics["group_reward_std"] = 0.0
                sample.metrics["group_size"] = float(n)
                try:
                    self.rollout_buffer.put(sample, timeout=2.0)
                except queue.Full:
                    _log.warning("queue full, dropping sample from group %s", task_id)
            _log.info(
                "group complete (all failed): task=%s n=%d", task_id, n
            )
            return

        mean_r = sum(valid_rewards) / len(valid_rewards)
        var_r = sum((r - mean_r) ** 2 for r in valid_rewards) / len(valid_rewards)
        std_r = var_r**0.5

        for i, sample in enumerate(group):
            if not valid_mask[i]:
                # Failed session: zero advantage, zero gradient.
                sample.advantage = 0.0
            elif std_r > 1e-8:
                sample.advantage = (rewards[i] - mean_r) / std_r
            else:
                # All valid rewards identical — zero advantage
                sample.advantage = 0.0

            sample.metrics["group_reward_mean"] = mean_r
            sample.metrics["group_reward_std"] = std_r
            sample.metrics["group_size"] = float(n)

            try:
                self.rollout_buffer.put(sample, timeout=2.0)
            except queue.Full:
                _log.warning(
                    "queue full, dropping sample from group %s", task_id
                )

        _log.info(
            "group complete: task=%s n=%d valid=%d reward_mean=%.3f reward_std=%.3f",
            task_id, n, len(valid_rewards), mean_r, std_r,
        )

    def _loop(self, idx: int) -> None:
        while not self._stop.is_set():
            while self._pause.is_set() and not self._stop.is_set():
                time.sleep(0.05)
            if self._stop.is_set():
                return

            task = self._next_group_task()
            eid = f"swe-{idx}-{uuid.uuid4().hex[:8]}"
            try:
                sample = asyncio.run(self._rollout(task, eid))
            except SWERolloutWorker.ContextOverflowError as exc:
                # Record the overflow and submit a zero-reward sample so
                # the group can still progress.  After max_context_overflow_per_task
                # failures the task is blacklisted.
                _log.warning(
                    "context overflow worker=%d task=%s: %s",
                    idx,
                    task.instance_id,
                    exc,
                )
                self._record_context_overflow(task.instance_id)
                pad = getattr(self._tokenizer, "pad_token_id", 0) or 0
                overflow_sample = RolloutSample(
                    input_ids=[pad],
                    completion_mask=[1],
                    old_log_probs=[0.0],
                    advantage=0.0,
                    model_version=self._model_ver(),
                    metrics={
                        "reward": 0.0,
                        "turns": 0.0,
                        "context_overflow": 1.0,
                        "answer_called": 0.0,
                    },
                )
                self._submit_to_group(task, overflow_sample)
                time.sleep(self._cfg.idle_backoff_s)
                continue
            except Exception:
                _log.exception("rollout failed worker=%d id=%s", idx, task.instance_id)
                time.sleep(self._cfg.idle_backoff_s)
                continue

            if sample is None:
                time.sleep(self._cfg.idle_backoff_s)
                continue

            self._submit_to_group(task, sample)

    # ── Single rollout ─────────────────────────────────────────────

    async def _rollout(self, task: SWETask, episode_id: str) -> RolloutSample | None:
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
        # prev_prompt_ids: the FULL prompt (with add_generation_prompt=True)
        # used for the most recently completed turn.  Used for EOT-based
        # interstitial extraction following Polar's prefix_merging approach
        # (arXiv:2605.24220 §3.4.2).  The next turn's canonical_tail =
        # next_prompt[len(prev_prompt_ids):] contains the canonical response
        # copy + interstitial; we split at EOT to get only the interstitial.
        prev_prompt_ids: list[int] | None = None
        # The raw turn_ids from the most recent generation, needed to detect
        # whether the response already ends with EOT (for correct slicing).
        prev_turn_ids: list[int] | None = None

        turns = 0
        answer_called = False
        pending_intercept: dict[str, Any] | None = None
        turn_limit: int | None = self._cfg.max_turns if self._cfg.max_turns > 0 else None
        rollout_stop_reason = "max_turns" if turn_limit is not None else "running"
        t0 = time.time()

        try:
            while (turn_limit is None or turns < turn_limit) and not self._stop.is_set():
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

                # Trim messages to fit within context budget before tokenizing.
                messages, _ = _fit_messages_to_context_window(
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

                current_prompt_ids = self._render_prompt_ids(messages, tools)

                if initial_prompt_ids is None:
                    # First turn: the entire prompt is non-completion tokens.
                    initial_prompt_ids = current_prompt_ids
                    all_ids.extend(current_prompt_ids)
                    all_mask.extend([0] * len(current_prompt_ids))
                    all_lps.extend([0.0] * len(current_prompt_ids))
                elif prev_prompt_ids is not None:
                    # Subsequent turns: compute the interstitial (tool-result
                    # suffix) between the last generation and this turn's prompt.
                    #
                    # Following Polar's prefix_merging (arXiv:2605.24220 §3.4.2)
                    # and TRL's _get_tool_suffix_ids approach:
                    #
                    # canonical_tail = next_prompt[len(prev_prompt):]  contains:
                    #   1. Canonical copy of prev response (may differ from raw
                    #      turn_ids due to BPE non-canonicality)
                    #   2. EOT token (<|im_end|>)
                    #   3. True interstitial (tool result, user msg, gen prompt)
                    #
                    # We split at EOT to extract only the interstitial (#3),
                    # avoiding the ~2× sequence bloat from duplicating responses.
                    #
                    # If the prefix breaks (due to message truncation changing
                    # earlier content between turns), we find the longest common
                    # prefix first, then EOT-split the remainder.
                    prev_len = len(prev_prompt_ids)
                    if (
                        len(current_prompt_ids) >= prev_len
                        and current_prompt_ids[:prev_len] == prev_prompt_ids
                    ):
                        canonical_tail = current_prompt_ids[prev_len:]
                    else:
                        # Message truncation changed earlier content — find
                        # the actual divergence point.
                        common_len = 0
                        for a, b in zip(current_prompt_ids, prev_prompt_ids):
                            if a != b:
                                break
                            common_len += 1
                        canonical_tail = current_prompt_ids[common_len:]
                        if common_len < prev_len:
                            _log.debug(
                                "prefix diverged at turn %d: expected %d, "
                                "common=%d (likely truncation drift)",
                                turns, prev_len, common_len,
                            )

                    suffix_ids = _extract_interstitial_after_eot(
                        canonical_tail=canonical_tail,
                        prev_turn_ids=prev_turn_ids,
                        eot_id=self._tokenizer.eos_token_id,
                    )

                    all_ids.extend(suffix_ids)
                    all_mask.extend([0] * len(suffix_ids))
                    all_lps.extend([0.0] * len(suffix_ids))

                turns += 1

                # ── Generate via /v1/completions ──────────────────
                # If context overflows on turn 1, let it propagate to _loop()
                # which emits a zero-reward placeholder.  On later turns,
                # terminate the rollout with the partial trajectory (reward=0)
                # so the session still contributes to the group.
                try:
                    turn_ids, turn_lps, text, finish_reason = self._generate(
                        current_prompt_ids
                    )
                except SWERolloutWorker.ContextOverflowError:
                    if turns <= 1:
                        raise
                    _log.warning(
                        "context overflow at turn %d for task=%s — "
                        "terminating with partial trajectory (%d tokens)",
                        turns,
                        task.instance_id,
                        len(all_ids),
                    )
                    rollout_stop_reason = "context_overflow"
                    self._record_context_overflow(task.instance_id)
                    break

                all_ids.extend(turn_ids)
                all_mask.extend([1] * len(turn_ids))
                all_lps.extend(turn_lps)

                # For next turn's suffix computation: save this turn's full
                # prompt (with add_generation_prompt=True) as the reference.
                # The next turn's prompt will extend this prefix (same messages
                # plus the response + tool result), so canonical_tail =
                # next_prompt[len(prev_prompt):] gives us the new content.
                #
                # This is Polar/TRL's approach: comparing canonical-vs-canonical
                # server tokenizations avoids BPE drift from raw tokens.
                prev_prompt_ids = current_prompt_ids
                prev_turn_ids = turn_ids

                # ── Build chat response for Pi ────────────────────
                assistant_message = _parse_assistant_message(
                    tokenizer=self._tokenizer,
                    completion_ids=turn_ids,
                    fallback_text=text,
                )
                chat_resp = _make_chat_response(
                    assistant_message,
                    self._vllm_model,
                    finish_reason=finish_reason,
                )

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
            # Context overflow means the agent didn't finish; assign reward=0.
            if rollout_stop_reason == "context_overflow":
                reward = 0.0
            else:
                vr = session.verify(transcript=[])
                reward = float(getattr(vr, "env_reward", 0.0) or 0.0)

            if not all_lps:
                pad = getattr(self._tokenizer, "pad_token_id", 0) or 0
                all_ids = [pad]
                all_mask = [1]
                all_lps = [0.0]

            return RolloutSample(
                input_ids=all_ids,
                completion_mask=all_mask,
                old_log_probs=all_lps,
                advantage=reward,
                model_version=self._model_ver(),
                metrics={
                    "reward": reward,
                    "turns": float(turns),
                    "answer_called": float(answer_called),
                    "context_overflow": float(
                        rollout_stop_reason == "context_overflow"
                    ),
                    "terminal_idle_stop": float(
                        rollout_stop_reason
                        in {
                            "idle_after_terminal_stop",
                            "agent_exit_after_terminal_stop",
                        }
                    ),
                    "request_idle_timeout": float(
                        rollout_stop_reason == "request_idle_timeout"
                    ),
                    "wall_s": round(time.time() - t0, 3),
                    "n_tokens": float(len(all_ids)),
                },
            )
        finally:
            session.close()

    # ── vLLM call (matches TRL's _generate_one_turn exactly) ──────

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
        # Use TRL's prefix-preserving training template when available.
        if self._chat_template:
            kwargs["chat_template"] = self._chat_template

        # Qwen3.5's chat template iterates tool_call.arguments with |items,
        # expecting a dict.  The OpenAI API spec (and our _normalize_tool_calls)
        # stores arguments as a JSON *string*.  Parse them into dicts here so
        # the Jinja template can iterate key-value pairs.
        messages = _ensure_tool_call_arguments_parsed(messages)

        try:
            ids = self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            ids = self._tokenizer.apply_chat_template(messages, **kwargs)
        return cast(list[int], ids)

    class ContextOverflowError(RuntimeError):
        """Raised when prompt + max_tokens exceeds max_model_len.

        Caught in _loop() to terminate the session gracefully with
        reward=0, avoiding the infinite retry loop.
        """

    def _generate(
        self,
        prompt_ids: list[int],
    ) -> tuple[list[int], list[float], str, str | None]:
        """POST /v1/completions with token IDs.

        Returns: ``(token_ids, token_logprobs, text, finish_reason)``.

        Raises ContextOverflowError if the prompt would exceed
        max_model_len.  This is caught in _loop() for graceful
        termination with reward=0.
        """
        # Pre-generation guard: check before sending to vLLM so we don't
        # waste a round trip and risk the engine going idle.
        total_needed = len(prompt_ids) + self._cfg.max_completion_tokens
        if total_needed > self._cfg.max_model_len:
            raise SWERolloutWorker.ContextOverflowError(
                f"prompt ({len(prompt_ids)}) + max_tokens "
                f"({self._cfg.max_completion_tokens}) = {total_needed} > "
                f"max_model_len ({self._cfg.max_model_len})"
            )

        body = {
            "model": self._vllm_model,
            "prompt": prompt_ids,
            "max_tokens": self._cfg.max_completion_tokens,
            "temperature": self._cfg.temperature,
            "n": 1,
            "return_token_ids": True,
            "logprobs": 0,
        }
        resp = requests.post(
            f"{self._vllm_base_url}/v1/completions",
            headers={
                "Authorization": f"Bearer {self._vllm_api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self._cfg.request_timeout_s,
        )
        if resp.status_code != 200:
            error_text = resp.text[:400]
            # Detect context overflow from vLLM's 400 response
            if resp.status_code == 400 and "max_model_len" in error_text:
                raise SWERolloutWorker.ContextOverflowError(
                    f"vLLM rejected: {error_text}"
                )
            raise RuntimeError(f"vllm {resp.status_code}: {error_text}")

        choice = resp.json()["choices"][0]
        return (
            choice["token_ids"],
            choice["logprobs"]["token_logprobs"],
            choice.get("text", ""),
            choice.get("finish_reason"),
        )


# ── Helpers ────────────────────────────────────────────────────────────


def _extract_interstitial_after_eot(
    *,
    canonical_tail: list[int],
    prev_turn_ids: list[int] | None,
    eot_id: int | None,
) -> list[int]:
    """Extract the true interstitial from a canonical tail using EOT splitting.

    Following Polar's ``PrefixMergingBuilder._slice_interstitial`` and TRL's
    ``_get_tool_suffix_ids`` EOS-trimming approach:

    ``canonical_tail`` = next_prompt[len(prev_prompt):] contains:
      1. Canonical re-rendering of the previous assistant response
      2. EOT token (``<|im_end|>`` for Qwen/ChatML)
      3. True interstitial (tool result, user turn, gen prompt)

    We split at the first EOT to skip the canonical response copy (#1-#2)
    and return only the interstitial (#3).

    If ``prev_turn_ids`` already ends with EOT (natural stop), the EOT in
    the canonical tail is a duplicate — skip it.  Otherwise (truncation),
    include it so the stream closes the assistant turn.

    Parameters
    ----------
    canonical_tail:
        Tokens from next_prompt that come after prev_prompt.
    prev_turn_ids:
        The raw response token IDs from the previous generation.
    eot_id:
        The end-of-turn token ID (e.g. ``<|im_end|>`` / eos_token_id).

    Returns
    -------
    list[int]
        Only the interstitial tokens (tool results, user messages,
        generation prompt for the next turn).
    """
    if not canonical_tail:
        return canonical_tail

    if eot_id is None:
        raise ValueError(
            "Cannot extract interstitial without eot_id (tokenizer.eos_token_id). "
            "This would duplicate the previous response in the training stream."
        )

    # Find the first EOT in the canonical tail.  This marks the end of the
    # canonical copy of the previous response.
    try:
        eot_pos = canonical_tail.index(eot_id)
    except ValueError:
        # No EOT found — edge case (e.g. truncated response without stop token).
        # Return full tail as interstitial (conservative).
        return canonical_tail

    # If the raw response already ended with EOT (model emitted stop token),
    # the EOT in canonical_tail is a duplicate — skip past it.
    # Otherwise (response was truncated), include the EOT to properly close
    # the assistant turn in the training stream.
    if prev_turn_ids and prev_turn_ids[-1] == eot_id:
        return canonical_tail[eot_pos + 1:]
    else:
        return canonical_tail[eot_pos:]


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


def _ensure_tool_call_arguments_parsed(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deep-copy messages, parsing tool_call arguments from JSON strings to dicts.

    Qwen3.5's chat template iterates ``tool_call.arguments|items`` expecting a
    mapping.  The OpenAI API (and our interception layer) stores arguments as a
    JSON string.  This bridges the gap.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list):
            out.append(msg)
            continue
        new_tcs: list[dict[str, Any]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                new_tcs.append(tc)
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                new_tcs.append(tc)
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            new_tcs.append(
                {**tc, "function": {**fn, "arguments": args if isinstance(args, dict) else {}}}
            )
        out.append({**msg, "tool_calls": new_tcs})
    return out


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


# ── Context window message trimming ────────────────────────────────────

_TRUNCATION_MARKER = "\n...[truncated]...\n"
_OMITTED_TOOL_OUTPUT_MARKER = "[tool output omitted]"
_OMITTED_ASSISTANT_TEXT_MARKER = "[omitted]"


def _truncate_text_middle(
    text: str,
    *,
    max_chars: int,
    marker: str = _TRUNCATION_MARKER,
) -> tuple[str, bool]:
    """Truncate text in the middle, preserving head and tail."""
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
    """Truncate long tool/assistant messages in-place."""
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
    """Replace the oldest message of a given role with a short placeholder."""
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
    """Progressively truncate messages to fit prompt within context budget.

    Strategy:
    1. Truncate tool outputs to max_tool_message_chars, halving until min.
    2. Truncate assistant messages similarly.
    3. Replace oldest tool messages entirely with a placeholder.
    4. Replace oldest assistant messages entirely with a placeholder.

    Returns (trimmed_messages, prompt_token_ids).
    """
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

    # Phase 1: progressive truncation
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

    # Phase 2: replace entire oldest messages with placeholders
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
