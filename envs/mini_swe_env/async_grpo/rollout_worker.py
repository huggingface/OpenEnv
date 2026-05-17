from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterator, Protocol, Sequence, cast

import requests

from ..models import SWETask


_log = logging.getLogger(__name__)

VLLMForwarder = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]
AnswerToolInvoker = Callable[
    [dict[str, Any]],
    dict[str, Any] | Awaitable[dict[str, Any]],
]


class _SessionProtocol(Protocol):
    swe_task: SWETask

    async def next_request(self, timeout_s: float | None = None) -> dict[str, Any] | None:
        ...

    async def deliver(self, intercept: dict[str, Any], response_dict: dict[str, Any]) -> None:
        ...

    def verify(self, transcript: list[dict[str, Any]], final_state: Any | None = None) -> Any:
        ...

    def close(self) -> None:
        ...


class _SessionFactoryProtocol(Protocol):
    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> _SessionProtocol:
        ...


@dataclass(frozen=True)
class SWEAsyncRolloutWorkerConfig:
    max_inflight_tasks: int = 1
    queue_maxsize: int = 64
    request_timeout_s: float = 180.0
    max_turns: int = 50
    idle_backoff_s: float = 0.2
    queue_put_timeout_s: float = 1.0


@dataclass
class SWEAsyncRolloutSample:
    """One rollout item produced by the async worker.

    Field names are chosen to align with AsyncGRPO queue consumption.
    """

    input_ids: list[int]
    completion_mask: list[int]
    old_log_probs: list[float]
    advantages: float
    model_version: int
    metrics: dict[str, Any] = field(default_factory=dict)


class SWEAsyncRolloutWorker:
    """Background rollout producer compatible with AsyncGRPO's worker protocol.

    The worker continuously:
    - selects a SWE task,
    - creates an interception-gated SWE session,
    - forwards intercepted LLM requests to a vLLM/OpenAI-compatible endpoint,
    - bridges ``answer`` tool calls to host-side grading,
    - verifies reward and enqueues a rollout sample.
    """

    def __init__(
        self,
        *,
        session_factory: _SessionFactoryProtocol,
        tasks: Sequence[SWETask],
        vllm_forwarder: VLLMForwarder,
        answer_tool_invoker: AnswerToolInvoker | None = None,
        config: SWEAsyncRolloutWorkerConfig | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("tasks must be non-empty")

        self._session_factory = session_factory
        self._tasks = list(tasks)
        self._vllm_forwarder = vllm_forwarder
        self._answer_tool_invoker = answer_tool_invoker
        self.config = config or SWEAsyncRolloutWorkerConfig()

        self.rollout_buffer: queue.Queue[SWEAsyncRolloutSample] = queue.Queue(
            maxsize=self.config.queue_maxsize
        )

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._task_index = 0
        self._state_lock = threading.Lock()
        self._started = False
        self._model_version = 0
        self._last_weight_sync_param_count = 0

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._started = True

        self._stop_event.clear()
        self._threads = []
        for idx in range(max(1, self.config.max_inflight_tasks)):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(idx,),
                daemon=True,
                name=f"swe-async-rollout-{idx}",
            )
            thread.start()
            self._threads.append(thread)

        _log.info("swe_async_rollout_worker_started threads=%d", len(self._threads))

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        with self._state_lock:
            self._started = False
            self._threads = []
        _log.info("swe_async_rollout_worker_stopped")

    def pause(self) -> None:
        self._pause_event.set()
        _log.info("swe_async_rollout_worker_paused")

    def resume(self) -> None:
        self._pause_event.clear()
        _log.info("swe_async_rollout_worker_resumed")

    def send_weights(self, iterator: Iterator[tuple[str, Any]]) -> None:
        count = 0
        for _ in iterator:
            count += 1
        self._last_weight_sync_param_count = count
        _log.info("swe_async_rollout_worker_weight_sync_received params=%d", count)

    def update_model_version(self, version: int) -> None:
        with self._state_lock:
            self._model_version = int(version)

    @property
    def last_weight_sync_param_count(self) -> int:
        return self._last_weight_sync_param_count

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "started": self._started,
                "paused": self._pause_event.is_set(),
                "queue_size": self.rollout_buffer.qsize(),
                "max_queue_size": self.config.queue_maxsize,
                "model_version": self._model_version,
                "worker_threads": len(self._threads),
                "last_weight_sync_param_count": self._last_weight_sync_param_count,
            }

    def _current_model_version(self) -> int:
        with self._state_lock:
            return self._model_version

    def _next_task(self) -> SWETask:
        with self._state_lock:
            task = self._tasks[self._task_index % len(self._tasks)]
            self._task_index += 1
            return task

    def _wait_if_paused(self) -> bool:
        while self._pause_event.is_set() and not self._stop_event.is_set():
            time.sleep(0.05)
        return not self._stop_event.is_set()

    def _worker_loop(self, worker_idx: int) -> None:
        while not self._stop_event.is_set():
            if not self._wait_if_paused():
                return

            task = self._next_task()
            episode_id = f"swe-async-{worker_idx}-{uuid.uuid4().hex[:8]}"
            try:
                sample = asyncio.run(self._run_one_rollout(task, episode_id=episode_id))
            except Exception:
                _log.exception(
                    "swe_async_rollout_failed worker=%d instance_id=%s",
                    worker_idx,
                    task.instance_id,
                )
                time.sleep(self.config.idle_backoff_s)
                continue

            if sample is None:
                time.sleep(self.config.idle_backoff_s)
                continue

            try:
                self.rollout_buffer.put(sample, timeout=self.config.queue_put_timeout_s)
            except queue.Full:
                _log.warning(
                    "swe_async_rollout_queue_full dropping instance_id=%s", task.instance_id
                )
                continue

    async def _run_one_rollout(
        self,
        task: SWETask,
        *,
        episode_id: str,
    ) -> SWEAsyncRolloutSample | None:
        session = cast(
            _SessionProtocol,
            self._session_factory.create(task=task, episode_id=episode_id),
        )

        turns = 0
        answer_called = False
        answer_bridged = False
        all_token_ids: list[int] = []
        all_logprobs: list[float] = []
        t0 = time.time()

        try:
            while turns < self.config.max_turns and not self._stop_event.is_set():
                intercept = await session.next_request(
                    timeout_s=self.config.request_timeout_s,
                )
                if intercept is None:
                    break

                turns += 1
                response = await self._maybe_call_forwarder(self._vllm_forwarder, intercept)
                response = copy.deepcopy(response)

                if _response_has_answer_tool_call(response):
                    answer_called = True
                    if self._answer_tool_invoker is not None:
                        await self._maybe_call_forwarder(self._answer_tool_invoker, intercept)
                        answer_bridged = True
                    _strip_answer_tool_calls(response)

                token_ids, logprobs = _extract_logprob_trace(response)
                all_token_ids.extend(token_ids)
                all_logprobs.extend(logprobs)

                await session.deliver(intercept, response)

                if answer_bridged:
                    break

            verify = session.verify(transcript=[])
            reward = float(getattr(verify, "env_reward", 0.0) or 0.0)
            verify_metrics = dict(getattr(verify, "metrics", {}) or {})

            if not all_logprobs:
                all_logprobs = [0.0]
                all_token_ids = [0]

            model_version = self._current_model_version()
            sample = SWEAsyncRolloutSample(
                input_ids=all_token_ids,
                completion_mask=[1] * len(all_logprobs),
                old_log_probs=all_logprobs,
                advantages=reward,
                model_version=model_version,
                metrics={
                    **verify_metrics,
                    "instance_id": task.instance_id,
                    "turns": turns,
                    "answer_called": answer_called,
                    "answer_bridged": answer_bridged,
                    "rollout_wall_time_s": round(time.time() - t0, 3),
                },
            )
            return sample
        finally:
            session.close()

    @staticmethod
    async def _maybe_call_forwarder(
        forwarder: Callable[[dict[str, Any]], Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        result = forwarder(payload)
        if inspect.isawaitable(result):
            awaited = await cast(Awaitable[dict[str, Any]], result)
            return awaited
        return cast(dict[str, Any], result)


def build_openai_chat_forwarder(
    *,
    base_url: str,
    api_key: str,
    model: str,
    request_timeout_s: float = 180.0,
) -> VLLMForwarder:
    """Create a sync forwarder for OpenAI-compatible chat completions."""

    def _forward(intercept: dict[str, Any]) -> dict[str, Any]:
        body = dict(intercept.get("body") or {})
        body["model"] = model
        body["logprobs"] = True
        body["top_logprobs"] = int(body.get("top_logprobs") or 5)
        body.pop("stream", None)
        body.pop("stream_options", None)

        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=request_timeout_s,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"vllm forward error {response.status_code}: {response.text[:400]}"
            )
        return cast(dict[str, Any], response.json())

    return _forward


def build_answer_tool_invoker(
    *,
    interception_base_url: str,
    interception_auth_token: str,
    request_timeout_s: float = 180.0,
    extra_headers: dict[str, str] | None = None,
) -> AnswerToolInvoker:
    """Create a sync invoker for host-side ``answer`` tool bridging."""

    def _invoke(intercept: dict[str, Any]) -> dict[str, Any]:
        rollout_id = str(intercept.get("rollout_id") or "")
        if not rollout_id:
            raise RuntimeError("intercept missing rollout_id for answer invocation")

        headers = {
            "Authorization": f"Bearer {interception_auth_token}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        response = requests.post(
            f"{interception_base_url.rstrip('/')}/rollout/{rollout_id}/v1/tools/answer",
            headers=headers,
            json={"arguments": {}},
            timeout=request_timeout_s,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"answer bridge error {response.status_code}: {response.text[:400]}"
            )
        return cast(dict[str, Any], response.json())

    return _invoke


def _response_has_answer_tool_call(response: dict[str, Any]) -> bool:
    choices = response.get("choices") or []
    if not choices:
        return False
    first = choices[0] or {}
    message = first.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        function = (tool_call or {}).get("function") or {}
        if function.get("name") == "answer":
            return True
    return False


def _strip_answer_tool_calls(response: dict[str, Any]) -> None:
    choices = list(response.get("choices") or [])
    if not choices:
        return

    first = dict(choices[0])
    message = dict(first.get("message") or {})
    message.pop("tool_calls", None)
    current = str(message.get("content") or "").strip()
    suffix = "Submission received and graded on host."
    message["content"] = f"{current}\n{suffix}".strip()
    first["message"] = message
    first["finish_reason"] = "stop"
    choices[0] = first
    response["choices"] = choices


def _extract_logprob_trace(response: dict[str, Any]) -> tuple[list[int], list[float]]:
    token_ids: list[int] = []
    logprobs: list[float] = []

    choices = response.get("choices") or []
    if not choices:
        return token_ids, logprobs

    first = choices[0] or {}
    content = ((first.get("logprobs") or {}).get("content") or [])
    for row in content:
        if not isinstance(row, dict):
            continue
        lp = row.get("logprob")
        if isinstance(lp, (int, float)):
            logprobs.append(float(lp))
        token_id = row.get("token_id")
        token_ids.append(int(token_id) if isinstance(token_id, int) else 0)

    return token_ids, logprobs


__all__ = [
    "AnswerToolInvoker",
    "SWEAsyncRolloutSample",
    "SWEAsyncRolloutWorker",
    "SWEAsyncRolloutWorkerConfig",
    "VLLMForwarder",
    "build_answer_tool_invoker",
    "build_openai_chat_forwarder",
]
