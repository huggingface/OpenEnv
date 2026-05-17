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
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence, cast

import requests

from ..models import SWETask

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


# ── Config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkerConfig:
    max_inflight: int = 2
    queue_maxsize: int = 64
    request_timeout_s: float = 600.0
    max_turns: int = 50
    max_completion_tokens: int = 2048
    temperature: float = 1.0
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
        self._threads: list[threading.Thread] = []
        self._task_idx = 0
        self._model_version = 0
        self._started = False

    # ── RolloutWorkerProtocol ──────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._stop.clear()
        for i in range(max(1, self._cfg.max_inflight)):
            t = threading.Thread(
                target=self._loop, args=(i,), daemon=True,
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

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def send_weights(self, iterator: Iterator[tuple[str, Any]]) -> None:
        # Consume iterator (required by protocol). Real NCCL sync is future work.
        for _ in iterator:
            pass

    def update_model_version(self, version: int) -> None:
        with self._lock:
            self._model_version = version

    # ── Internal ───────────────────────────────────────────────────

    def _next_task(self) -> SWETask:
        with self._lock:
            t = self._tasks[self._task_idx % len(self._tasks)]
            self._task_idx += 1
            return t

    def _model_ver(self) -> int:
        with self._lock:
            return self._model_version

    def _loop(self, idx: int) -> None:
        while not self._stop.is_set():
            while self._pause.is_set() and not self._stop.is_set():
                time.sleep(0.05)
            if self._stop.is_set():
                return

            task = self._next_task()
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

            try:
                self.rollout_buffer.put(sample, timeout=2.0)
            except queue.Full:
                _log.warning("queue full, dropping %s", task.instance_id)

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
        t0 = time.time()

        try:
            while turns < self._cfg.max_turns and not self._stop.is_set():
                intercept = await session.next_request(
                    timeout_s=self._cfg.request_timeout_s,
                )
                if intercept is None:
                    break

                # ── Tokenize this turn's full prompt ──────────────
                messages = _get_messages(intercept)
                current_prompt_ids = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_dict=False,
                )

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
                turn_ids, turn_lps, text = self._generate(current_prompt_ids)

                all_ids.extend(turn_ids)
                all_mask.extend([1] * len(turn_ids))
                all_lps.extend(turn_lps)

                # For next turn's suffix computation:
                prev_prompt_ids = current_prompt_ids + turn_ids

                # ── Build chat response for Pi ────────────────────
                chat_resp = _make_chat_response(text, self._vllm_model)

                # ── Check for answer tool call ────────────────────
                if _has_answer_call(chat_resp):
                    answer_called = True

                await session.deliver(intercept, chat_resp)

                if answer_called:
                    break

            # ── Reward ────────────────────────────────────────────
            vr = session.verify(transcript=[])
            reward = float(getattr(vr, "env_reward", 0.0) or 0.0)
            metrics = dict(getattr(vr, "metrics", {}) or {})

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
                    **metrics,
                    "reward": reward,
                    "instance_id": task.instance_id,
                    "turns": turns,
                    "answer_called": answer_called,
                    "wall_s": round(time.time() - t0, 3),
                    "n_tokens": len(all_ids),
                },
            )
        finally:
            session.close()

    # ── vLLM call (matches TRL's _generate_one_turn exactly) ──────

    def _generate(
        self, prompt_ids: list[int],
    ) -> tuple[list[int], list[float], str]:
        """POST /v1/completions with token IDs. Returns (ids, logprobs, text)."""
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


def _make_chat_response(text: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


def _has_answer_call(resp: dict[str, Any]) -> bool:
    for choice in (resp.get("choices") or []):
        for tc in ((choice or {}).get("message") or {}).get("tool_calls") or []:
            if ((tc or {}).get("function") or {}).get("name") == "answer":
                return True
    return False
