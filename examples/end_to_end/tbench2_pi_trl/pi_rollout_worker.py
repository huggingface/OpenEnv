"""Pi rollout worker for the Terminus async GRPO example."""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator, Sequence, cast

import requests
from openenv.core.harness import HarnessRunLimits
from openenv.core.harness.pi_cli import PiCLIHarnessAdapter

try:
    from terminus_env.harness import build_terminal_tool_call
except Exception:  # pragma: no cover - optional outside the Terminus example
    build_terminal_tool_call = None

try:
    from trl.chat_template_utils import (
        add_response_schema,
        get_training_chat_template,
        is_chat_template_prefix_preserving,
        parse_response,
    )
except Exception:  # pragma: no cover - optional across TRL revisions
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
except Exception:  # pragma: no cover - optional outside training runtime
    NCCLTrainerSendWeightsArgs = None
    NCCLWeightTransferEngine = None
    get_ip = None
    get_open_port = None

logger = logging.getLogger(__name__)


def _vllm_version() -> tuple[int, ...]:
    try:
        import vllm

        return tuple(int(part) for part in vllm.__version__.split(".")[:3])
    except Exception:
        return (0, 0, 0)


_VLLM_NEEDS_WEIGHT_UPDATE_LIFECYCLE = _vllm_version() >= (0, 21, 0)


@dataclass
class RolloutSample:
    input_ids: list[int]
    completion_mask: list[int]
    old_log_probs: list[float]
    advantage: float
    model_version: int
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerConfig:
    max_inflight: int = 2
    queue_maxsize: int = 64
    max_turns: int = 8
    max_completion_tokens: int = 512
    temperature: float = 1.0
    request_timeout_s: float = 600.0
    server_timeout_s: float = 600.0
    idle_sleep_s: float = 0.25


class InterceptionServer:
    """Minimal OpenAI-compatible gate for PI chat completion requests."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0, secret: str = "openenv"):
        self.host = host
        self.port = port
        self.secret = secret
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._rollouts: dict[str, queue.Queue[str]] = {}
        self._intercepts: dict[str, dict[str, Any]] = {}

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("interception server is not running")
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self._server is not None:
            return
        server = ThreadingHTTPServer((self.host, self.port), self._handler())
        self._server = server
        self.port = int(server.server_port)
        self._thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="terminus-pi-interception",
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def register_rollout(self, rollout_id: str) -> queue.Queue[str]:
        request_queue: queue.Queue[str] = queue.Queue()
        with self._lock:
            self._rollouts[rollout_id] = request_queue
        return request_queue

    def unregister_rollout(self, rollout_id: str) -> None:
        with self._lock:
            self._rollouts.pop(rollout_id, None)
            intercepts = [
                key
                for key, intercept in self._intercepts.items()
                if intercept.get("rollout_id") == rollout_id
            ]
            for key in intercepts:
                intercept = self._intercepts.pop(key)
                intercept["response"] = _error_response("rollout cancelled")
                intercept["event"].set()

    def get_intercept(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._intercepts.get(request_id)

    def deliver(self, intercept: dict[str, Any], response: dict[str, Any]) -> None:
        intercept["response"] = response
        intercept["event"].set()

    def _authorized(self, headers: Any) -> bool:
        auth = headers.get("Authorization", "")
        api_key = headers.get("x-api-key", "")
        return auth == f"Bearer {self.secret}" or api_key == self.secret

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return None

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._json({"status": "ok"})
                    return
                self._json({"error": "not found"}, status=404)

            def do_POST(self) -> None:
                if not outer._authorized(self.headers):
                    self._json({"error": "unauthorized"}, status=401)
                    return
                match = re.fullmatch(
                    r"/rollout/([^/]+)/v1/chat/completions",
                    self.path.split("?", 1)[0],
                )
                if match is None:
                    self._json({"error": "not found"}, status=404)
                    return
                rollout_id = match.group(1)
                try:
                    length = int(self.headers.get("content-length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception as exc:
                    self._json({"error": f"invalid JSON: {exc}"}, status=400)
                    return

                with outer._lock:
                    request_queue = outer._rollouts.get(rollout_id)
                if request_queue is None:
                    self._json({"error": "rollout not found"}, status=404)
                    return

                request_id = f"req_{uuid.uuid4().hex[:8]}"
                intercept = {
                    "request_id": request_id,
                    "rollout_id": rollout_id,
                    "messages": body.get("messages"),
                    "tools": body.get("tools"),
                    "body": body,
                    "event": threading.Event(),
                    "response": None,
                }
                with outer._lock:
                    outer._intercepts[request_id] = intercept
                request_queue.put(request_id)

                if not intercept["event"].wait(timeout=900):
                    self._json({"error": "interception timeout"}, status=504)
                    return

                with outer._lock:
                    outer._intercepts.pop(request_id, None)
                response = intercept["response"] or _error_response("empty response")
                self._json(response)

            def _json(self, payload: dict[str, Any], *, status: int = 200) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


class TerminusPiRolloutWorker:
    """TRL rollout worker that lets PI drive tools while trainer owns generation."""

    def __init__(
        self,
        *,
        session_factory: Any,
        tasks: Sequence[Any],
        tokenizer: Any,
        vllm_base_url: str,
        vllm_model: str,
        vllm_api_key: str = "openenv",
        config: WorkerConfig | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        pi_command: str = "pi",
        command_runner: Callable[..., Any] | None = None,
    ):
        self._session_factory = session_factory
        self._tasks = list(tasks)
        if not self._tasks:
            raise ValueError("tasks must not be empty")
        self._tokenizer = tokenizer
        if add_response_schema is not None:
            try:
                self._tokenizer = add_response_schema(tokenizer)
            except Exception:
                logger.debug("could not add response schema to tokenizer", exc_info=True)
        self._chat_template = None
        if (
            get_training_chat_template is not None
            and is_chat_template_prefix_preserving is not None
        ):
            try:
                if not is_chat_template_prefix_preserving(self._tokenizer):
                    self._chat_template = get_training_chat_template(self._tokenizer)
            except Exception:
                logger.debug("could not inspect chat template", exc_info=True)
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        self._vllm_base_url = vllm_base_url.rstrip("/")
        self._vllm_model = vllm_model
        self._vllm_api_key = vllm_api_key
        self._config = config or WorkerConfig()
        self._pi_command = pi_command
        self._command_runner = command_runner

        self.rollout_buffer: queue.Queue[RolloutSample] = queue.Queue(
            maxsize=self._config.queue_maxsize,
        )
        self._interception = InterceptionServer(secret=vllm_api_key)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.Lock()
        self._weight_sync_lock = threading.Lock()
        self._task_index = 0
        self._model_version = 0
        self._last_heartbeat_s = time.monotonic()
        self._model_update_group: Any | None = None

        self._wait_for_server_ready()
        self._init_weight_transfer()

    def start(self) -> None:
        self._interception.start()
        self._stop.clear()
        self._last_heartbeat_s = time.monotonic()
        with self._lock:
            if self._threads:
                return
            for index in range(max(1, self._config.max_inflight)):
                thread = threading.Thread(
                    target=self._loop,
                    args=(index,),
                    daemon=True,
                    name=f"terminus-pi-rollout-{index}",
                )
                thread.start()
                self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        with self._lock:
            self._threads = []
        self._interception.stop()
        self._destroy_model_update_group()

    def pause(self) -> None:
        self._pause.set()
        if self._model_update_group is not None:
            self._post_json("/pause", params={"mode": "keep"}, timeout=60)

    def resume(self) -> None:
        if self._model_update_group is not None:
            self._post_json("/resume", timeout=60)
        self._pause.clear()

    def send_weights(self, iterator: Iterator[tuple[str, Any]]) -> None:
        items = list(iterator)
        if not items:
            return
        if self._model_update_group is None:
            raise RuntimeError("vLLM weight-transfer group is not initialized")

        update_info = {
            "names": [name for name, _ in items],
            "dtype_names": [
                str(getattr(tensor, "dtype", "float32")).split(".")[-1]
                for _, tensor in items
            ],
            "shapes": [list(getattr(tensor, "shape", [])) for _, tensor in items],
            "packed": True,
            "is_checkpoint_format": True,
        }

        with self._weight_sync_lock:
            if _VLLM_NEEDS_WEIGHT_UPDATE_LIFECYCLE:
                self._post_json(
                    "/start_weight_update",
                    json_body={"is_checkpoint_format": True},
                    timeout=60,
                )

            post_error: list[Exception] = []

            def post_update() -> None:
                try:
                    self._post_json(
                        "/update_weights",
                        json_body={"update_info": update_info},
                        timeout=1800,
                    )
                except Exception as exc:  # noqa: BLE001
                    post_error.append(exc)

            update_thread = threading.Thread(target=post_update, daemon=True)
            update_thread.start()

            assert NCCLTrainerSendWeightsArgs is not None
            assert NCCLWeightTransferEngine is not None
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(items),
                trainer_args=NCCLTrainerSendWeightsArgs(
                    group=self._model_update_group,
                    packed=True,
                ),
            )

            update_thread.join(timeout=1800)
            if update_thread.is_alive():
                raise TimeoutError("timed out waiting for vLLM /update_weights")
            if post_error:
                raise RuntimeError("vLLM /update_weights failed") from post_error[0]
            if _VLLM_NEEDS_WEIGHT_UPDATE_LIFECYCLE:
                self._post_json("/finish_weight_update", timeout=120)

    def update_model_version(self, version: int) -> None:
        with self._lock:
            self._model_version = version

    def check_health(self, stale_after_s: float) -> None:
        if not self._threads or not any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("Terminus PI rollout worker is not running")
        age = time.monotonic() - self._last_heartbeat_s
        if age > stale_after_s:
            raise RuntimeError(
                f"Terminus PI rollout worker heartbeat stale: {age:.0f}s"
            )

    def _wait_for_server_ready(self) -> None:
        start = time.time()
        while True:
            try:
                response = requests.get(f"{self._vllm_base_url}/health", timeout=5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            if time.time() - start >= self._config.server_timeout_s:
                raise TimeoutError(
                    f"timed out waiting for vLLM server at {self._vllm_base_url}"
                )
            time.sleep(2.0)

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
            NCCLTrainerSendWeightsArgs is None
            or NCCLWeightTransferEngine is None
            or get_ip is None
            or get_open_port is None
        ):
            raise RuntimeError("vLLM NCCL weight-transfer modules are unavailable")

        response = requests.get(
            f"{self._vllm_base_url}/get_world_size",
            headers={"Authorization": f"Bearer {self._vllm_api_key}"},
            timeout=10,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "vLLM weight sync requires /get_world_size. Start vLLM with "
                'VLLM_SERVER_DEV_MODE=1 and --weight-transfer-config \'{"backend":"nccl"}\'.'
            )

        inference_world_size = int(response.json()["world_size"])
        init_info = {
            "master_address": get_ip(),
            "master_port": get_open_port(),
            "rank_offset": 1,
            "world_size": inference_world_size + 1,
        }
        post_error: list[Exception] = []

        def post_init() -> None:
            try:
                self._post_json(
                    "/init_weight_transfer_engine",
                    json_body={"init_info": init_info},
                    timeout=120,
                )
            except Exception as exc:  # noqa: BLE001
                post_error.append(exc)

        init_thread = threading.Thread(target=post_init, daemon=True)
        init_thread.start()
        self._model_update_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": init_info["master_address"],
                "master_port": init_info["master_port"],
                "world_size": init_info["world_size"],
            }
        )
        init_thread.join(timeout=120)
        if init_thread.is_alive():
            raise TimeoutError("timed out waiting for vLLM weight-transfer init")
        if post_error:
            raise RuntimeError("vLLM weight-transfer init failed") from post_error[0]

    def _destroy_model_update_group(self) -> None:
        group = self._model_update_group
        if group is None:
            return
        try:
            group.group.store = None
            group.group.socket = None
        except Exception:
            logger.debug("could not destroy vLLM weight-transfer group", exc_info=True)
        self._model_update_group = None

    def _loop(self, worker_index: int) -> None:
        while not self._stop.is_set():
            while self._pause.is_set() and not self._stop.is_set():
                time.sleep(0.05)
            if self._stop.is_set():
                return
            task = self._next_task()
            try:
                sample = self._rollout(
                    task,
                    f"terminus-{worker_index}-{uuid.uuid4().hex[:8]}",
                )
                self.rollout_buffer.put(sample, timeout=2.0)
                self._last_heartbeat_s = time.monotonic()
            except Exception:
                logger.exception("terminus PI rollout failed")
                time.sleep(self._config.idle_sleep_s)

    def _next_task(self) -> Any:
        with self._lock:
            task = self._tasks[self._task_index % len(self._tasks)]
            self._task_index += 1
            return task

    def _rollout(self, task: Any, rollout_id: str) -> RolloutSample:
        session = self._session_factory.create(
            task=_session_task(task),
            episode_id=rollout_id,
        )
        request_queue = self._interception.register_rollout(rollout_id)
        result_box: dict[str, Any] = {}
        error_box: list[BaseException] = []

        adapter = PiCLIHarnessAdapter(
            pi_command=self._pi_command,
            model=self._vllm_model,
            model_base_url=f"{self._interception.base_url}/rollout/{rollout_id}/v1",
            model_api_key=self._vllm_api_key,
            timeout_s=self._config.request_timeout_s,
            command_runner=self._command_runner,
        )

        def run_pi() -> None:
            try:
                result_box["rollout"] = adapter.run_black_box(
                    session=session,
                    limits=HarnessRunLimits(max_turns=self._config.max_turns),
                )
            except BaseException as exc:  # noqa: BLE001
                error_box.append(exc)

        pi_thread = threading.Thread(target=run_pi, daemon=True, name=f"pi-{rollout_id}")
        pi_thread.start()

        all_ids: list[int] = []
        all_mask: list[int] = []
        all_logprobs: list[float] = []
        previous_prompt_and_turn: list[int] | None = None
        turns = 0

        try:
            while turns < self._config.max_turns:
                self._last_heartbeat_s = time.monotonic()
                if error_box:
                    raise RuntimeError("pi subprocess failed") from error_box[0]
                try:
                    request_id = request_queue.get(timeout=0.5)
                except queue.Empty:
                    if not pi_thread.is_alive():
                        break
                    continue

                intercept = self._interception.get_intercept(request_id)
                if intercept is None:
                    continue
                prompt_ids = self._render_prompt_ids(intercept)
                if previous_prompt_and_turn is None:
                    all_ids.extend(prompt_ids)
                    all_mask.extend([0] * len(prompt_ids))
                    all_logprobs.extend([0.0] * len(prompt_ids))
                else:
                    suffix = prompt_ids[len(previous_prompt_and_turn) :]
                    all_ids.extend(suffix)
                    all_mask.extend([0] * len(suffix))
                    all_logprobs.extend([0.0] * len(suffix))

                turn_ids, turn_logprobs, text, finish_reason = self._generate(prompt_ids)
                all_ids.extend(turn_ids)
                all_mask.extend([1] * len(turn_ids))
                all_logprobs.extend(turn_logprobs)
                previous_prompt_and_turn = prompt_ids + turn_ids
                turns += 1

                assistant_message = _parse_assistant_message(
                    tokenizer=self._tokenizer,
                    completion_ids=turn_ids,
                    fallback_text=text,
                )
                self._interception.deliver(
                    intercept,
                    _chat_response(
                        assistant_message,
                        model=self._vllm_model,
                        finish_reason=finish_reason,
                    ),
                )

            pi_thread.join(timeout=2.0)
            if error_box:
                raise RuntimeError("pi subprocess failed") from error_box[0]
            rollout = result_box.get("rollout")
            verify = session.verify(
                transcript=[] if rollout is None else rollout.messages,
                final_state=None
                if rollout is None
                else {"done": rollout.done, "metrics": dict(rollout.metrics)},
            )
            reward = float(verify.env_reward or 0.0)
            if not all_ids:
                pad_id = getattr(self._tokenizer, "pad_token_id", None) or 0
                all_ids = [pad_id]
                all_mask = [1]
                all_logprobs = [0.0]
            metrics = {"reward": reward, "turns": float(turns)}
            if rollout is not None:
                metrics.update(
                    {
                        "pi/tool_calls": float(len(rollout.tool_trace)),
                        "pi/events": float(rollout.metrics.get("pi_events", 0.0)),
                        "pi/done": float(bool(rollout.done)),
                    }
                )
            for name, value in (verify.metrics or {}).items():
                if isinstance(value, bool):
                    metrics[f"verify/{name}"] = float(value)
                elif isinstance(value, (int, float)):
                    metrics[f"verify/{name}"] = float(value)
            with self._lock:
                model_version = self._model_version
            return RolloutSample(
                input_ids=all_ids,
                completion_mask=all_mask,
                old_log_probs=all_logprobs,
                advantage=reward,
                model_version=model_version,
                metrics=metrics,
            )
        finally:
            self._interception.unregister_rollout(rollout_id)
            pi_thread.join(timeout=1.0)
            session.close()

    def _render_prompt_ids(self, intercept: dict[str, Any]) -> list[int]:
        body = intercept.get("body") or {}
        messages = body.get("messages") or intercept.get("messages")
        if not isinstance(messages, list):
            raise RuntimeError("intercepted request did not include messages")
        messages = _normalize_chat_messages(messages)
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "return_dict": False,
            **self._chat_template_kwargs,
        }
        if self._chat_template is not None:
            kwargs["chat_template"] = self._chat_template
        tools = body.get("tools") or intercept.get("tools")
        if tools:
            kwargs["tools"] = tools
        try:
            return cast(list[int], self._tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("tools", None)
            return cast(list[int], self._tokenizer.apply_chat_template(messages, **kwargs))

    def _generate(self, prompt_ids: list[int]) -> tuple[list[int], list[float], str, str | None]:
        response = requests.post(
            f"{self._vllm_base_url}/v1/completions",
            headers={
                "Authorization": f"Bearer {self._vllm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._vllm_model,
                "prompt": prompt_ids,
                "max_tokens": self._config.max_completion_tokens,
                "temperature": self._config.temperature,
                "n": 1,
                "return_token_ids": True,
                "logprobs": 0,
            },
            timeout=self._config.request_timeout_s,
        )
        if response.status_code != 200:
            raise RuntimeError(f"vLLM {response.status_code}: {response.text[:400]}")
        choice = response.json()["choices"][0]
        token_ids = list(choice["token_ids"])
        logprobs = choice.get("logprobs", {}).get("token_logprobs", [])
        if len(logprobs) != len(token_ids):
            logprobs = [0.0] * len(token_ids)
        return (
            token_ids,
            [0.0 if value is None else float(value) for value in logprobs],
            str(choice.get("text", "")),
            choice.get("finish_reason"),
        )


def _parse_assistant_message(
    *,
    tokenizer: Any,
    completion_ids: list[int],
    fallback_text: str,
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    try:
        if parse_response is not None:
            parsed = parse_response(tokenizer, completion_ids)
    except Exception:
        logger.debug("could not parse TRL response schema", exc_info=True)
    if not isinstance(parsed, dict):
        parsed = {}
    content = str(parsed.get("content") or "")
    tool_calls = _normalize_tool_calls(parsed.get("tool_calls"))
    if not tool_calls:
        tool_calls = _terminal_tool_call_from_text(content or fallback_text)
    if tool_calls:
        return {"role": "assistant", "content": "", "tool_calls": tool_calls}
    return {"role": "assistant", "content": content or fallback_text}


def _normalize_chat_messages(messages: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("text") is not None
            )
        normalized.append(item)
    return normalized


def _session_task(task: Any) -> Any:
    if not isinstance(task, dict) or not isinstance(task.get("prompt"), list):
        return task
    instruction = "\n\n".join(
        str(message.get("content", ""))
        for message in task["prompt"]
        if isinstance(message, dict) and message.get("content")
    )
    if not instruction:
        return task
    return {**task, "instruction": instruction}


def _terminal_tool_call_from_text(text: str) -> list[dict[str, Any]]:
    if build_terminal_tool_call is None or not text.strip():
        return []
    try:
        tool_call = build_terminal_tool_call(
            text,
            call_id=f"call_{uuid.uuid4().hex[:8]}",
        )
    except Exception:
        logger.debug("could not parse Terminus terminal text", exc_info=True)
        return []
    arguments = getattr(tool_call, "args", None) or {}
    name = getattr(tool_call, "name", "")
    if name != "terminal" or not arguments:
        return []
    return [
        {
            "id": str(getattr(tool_call, "id", "") or f"call_{uuid.uuid4().hex[:8]}"),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }
    ]


def _normalize_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []
    tool_calls = []
    for raw_call in raw_tool_calls:
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        arguments = function.get("arguments") or {}
        tool_calls.append(
            {
                "id": str(raw_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": str(function["name"]),
                    "arguments": arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments),
                },
            }
        )
    return tool_calls


def _chat_response(
    assistant_message: dict[str, Any],
    *,
    model: str,
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": assistant_message,
                "finish_reason": finish_reason or "stop",
            }
        ],
    }


def _error_response(message: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "openenv-error",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": message},
                "finish_reason": "stop",
            }
        ],
    }


__all__ = ["TerminusPiRolloutWorker", "WorkerConfig"]
