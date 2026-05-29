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
from typing import Any, Iterator, Sequence, cast

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


def _vllm_version() -> tuple[int, ...]:
    """Return parsed vLLM version as a tuple, e.g. (0, 20, 2).

    Returns (0, 0, 0) if vLLM is not installed or version cannot be parsed.
    """
    try:
        import vllm  # noqa: F811

        parts = vllm.__version__.split(".")
        return tuple(int(p) for p in parts[:3])
    except Exception:
        return (0, 0, 0)


# vLLM 0.21+ uses a four-phase weight transfer protocol:
#   init_weight_transfer_engine → start_weight_update → update_weights → finish_weight_update
# start_weight_update(is_checkpoint_format=True) routes through model.load_weights()
# which handles parameter name remapping (e.g. CausalLM → VLM prefixes).
_VLLM_NEEDS_WEIGHT_UPDATE_LIFECYCLE = _vllm_version() >= (0, 21, 0)


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

        # Prefix to prepend to trainer parameter names so they match vLLM's
        # model architecture.  For VLM models like Qwen3_5ForConditionalGeneration
        # served with --language-model-only, vLLM parameters are at
        # "language_model.model.layers.X" but the trainer (AutoModelForCausalLM)
        # produces "model.layers.X".  Set to "" to disable remapping.
        #
        # Only Qwen3.5 models need this prefix — they are unified VLMs where
        # vLLM resolves Qwen3_5ForConditionalGeneration even in text-only mode.
        # Standard CausalLM models (Qwen3, Llama, etc.) have matching param names.
        self._vllm_weight_prefix = (
            "language_model."
            if "qwen3.5" in vllm_model.lower() or "qwen3_5" in vllm_model.lower()
            else ""
        )

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

        # VLM models (e.g. Qwen3_5ForConditionalGeneration) have parameters at
        # language_model.model.layers.X but the trainer (AutoModelForCausalLM)
        # produces names like model.layers.X.  Prepend the prefix so names match.
        if names and not names[0].startswith("language_model."):
            if self._vllm_weight_prefix:
                names = [f"{self._vllm_weight_prefix}{n}" for n in names]

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
            if _VLLM_NEEDS_WEIGHT_UPDATE_LIFECYCLE:
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

            # vLLM >= 0.21: finalize layerwise reload / quantization.
            if _VLLM_NEEDS_WEIGHT_UPDATE_LIFECYCLE:
                self._post_json("/finish_weight_update", timeout=120)

    def update_model_version(self, version: int) -> None:
        with self._lock:
            self._model_version = version

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

    def _next_group_task(self) -> SWETask:
        """Pick a task that needs more rollouts to complete its group.

        Prioritizes tasks that already have partial groups (some rollouts
        done but < num_generations). Falls back to a fresh task.
        """
        with self._group_lock:
            # Find a task with a partial group that needs more rollouts
            for task_id, samples in self._pending_groups.items():
                if len(samples) < self._cfg.num_generations:
                    # Return the corresponding task
                    for t in self._tasks:
                        if t.instance_id == task_id:
                            return t
        # No partial groups — start a fresh task
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
        """Compute group-relative advantages (GRPO) and push to buffer."""
        rewards = [s.advantage for s in group]  # raw rewards stored here
        n = len(rewards)
        mean_r = sum(rewards) / n
        var_r = sum((r - mean_r) ** 2 for r in rewards) / n
        std_r = var_r**0.5

        for i, sample in enumerate(group):
            if std_r > 1e-8:
                sample.advantage = (rewards[i] - mean_r) / std_r
            else:
                # All rewards identical (e.g. all 0 or all 1) — zero advantage
                sample.advantage = 0.0

            # Add group metrics
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
            "group complete: task=%s n=%d reward_mean=%.3f reward_std=%.3f",
            task_id, n, mean_r, std_r,
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

                # ── Generate via /v1/completions ──────────────────
                turn_ids, turn_lps, text, finish_reason = self._generate(
                    current_prompt_ids
                )

                all_ids.extend(turn_ids)
                all_mask.extend([1] * len(turn_ids))
                all_lps.extend(turn_lps)

                # For next turn's suffix computation:
                prev_prompt_ids = current_prompt_ids + turn_ids

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
        try:
            ids = self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            ids = self._tokenizer.apply_chat_template(messages, **kwargs)
        return cast(list[int], ids)

    def _generate(
        self,
        prompt_ids: list[int],
    ) -> tuple[list[int], list[float], str, str | None]:
        """POST /v1/completions with token IDs.

        Returns: ``(token_ids, token_logprobs, text, finish_reason)``.
        """
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
            raise RuntimeError(f"vllm {resp.status_code}: {resp.text[:400]}")

        choice = resp.json()["choices"][0]
        return (
            choice["token_ids"],
            choice["logprobs"]["token_logprobs"],
            choice.get("text", ""),
            choice.get("finish_reason"),
        )


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
