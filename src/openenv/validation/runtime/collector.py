"""Bounded raw protocol collection in one OpenEnv orchestration session."""

import json
import time
from urllib.parse import urlsplit, urlunsplit

import httpx
from websockets.sync.client import connect

from .contracts import RuntimeEvidence, RuntimePlan, WireExchange

MAX_MESSAGE_BYTES = 1024 * 1024
MAX_TRACE_BYTES = 8 * 1024 * 1024


def collect_runtime_evidence(
    base_url: str,
    plan: RuntimePlan,
    *,
    episode_timeout_s: float,
    request_timeout_s: float = 5.0,
) -> RuntimeEvidence:
    """
    Preserve schema and reset/step/state responses without model coercion.

    The collector never imports submitted code or reconnects midway through an
    episode. Transport failures retain the completed prefix and a bounded reason.
    Graders inspect the original envelopes rather than convenience-client defaults.

    Args:
        base_url (`str`):
            Provider-owned control endpoint.
        plan ([`~openenv.validation.runtime.contracts.RuntimePlan`]):
            Validated, bounded reset and action inputs.
        episode_timeout_s (`float`):
            Deadline for the complete collection, including schema retrieval.
        request_timeout_s (`float`, *optional*, defaults to `5.0`):
            Per-operation deadline, capped by the remaining episode budget.

    Returns:
        [`~openenv.validation.runtime.contracts.RuntimeEvidence`]: raw evidence.
    """
    deadline = time.monotonic() + episode_timeout_s
    exchanges = []
    schema_json = None
    phase = "schema"
    trace_bytes = 0
    completed = False

    def remaining() -> float:
        value = min(request_timeout_s, deadline - time.monotonic())
        if value <= 0:
            raise TimeoutError("episode deadline exceeded")
        return value

    def send(socket, message: str) -> None:
        timeout = remaining()
        transport = socket.socket
        previous_timeout = transport.gettimeout()
        transport.settimeout(timeout)
        try:
            socket.send(message)
        finally:
            try:
                transport.settimeout(previous_timeout)
            except OSError:
                pass

    try:
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            with client.stream(
                "GET",
                base_url.rstrip("/") + "/schema",
                timeout=remaining(),
                headers={"Accept-Encoding": "identity"},
            ) as response:
                response.raise_for_status()
                # iter_bytes() transparently decompresses. Reject compressed
                # bodies before touching the stream so the byte budget also
                # bounds allocation, even when a subject ignores our header.
                if (
                    response.headers.get("Content-Encoding", "identity").strip().lower()
                    != "identity"
                ):
                    raise ValueError("compressed schema responses are not supported")
                payload = bytearray()
                for chunk in response.iter_bytes():
                    remaining()
                    if len(payload) + len(chunk) > MAX_MESSAGE_BYTES:
                        raise ValueError("schema exceeds size bound")
                    payload.extend(chunk)
            schema = json.loads(payload)
            if not isinstance(schema, dict) or "observation" not in schema:
                raise ValueError("missing observation schema")
            schema_json = json.dumps(schema["observation"], allow_nan=False)

        endpoint = urlsplit(base_url)
        ws_url = urlunsplit(
            (
                "wss" if endpoint.scheme == "https" else "ws",
                endpoint.netloc,
                endpoint.path.rstrip("/") + "/ws",
                "",
                "",
            )
        )
        phase = "connect"
        with connect(
            ws_url,
            proxy=None,
            open_timeout=remaining(),
            close_timeout=1,
            max_size=MAX_MESSAGE_BYTES,
            max_queue=1,
            compression=None,
        ) as socket:

            def exchange(operation: str, data: dict | None = None) -> dict:
                nonlocal phase, trace_bytes
                phase = operation
                request = {"type": operation}
                if data is not None:
                    request["data"] = data
                request_json = json.dumps(request, allow_nan=False)
                send(socket, request_json)
                raw = socket.recv(timeout=remaining())
                if not isinstance(raw, str):
                    raise ValueError("binary response is not the JSON protocol")
                trace_bytes += len(raw.encode("utf-8")) + len(
                    request_json.encode("utf-8")
                )
                if trace_bytes > MAX_TRACE_BYTES:
                    raise ValueError("trace exceeds total size bound")
                exchanges.append(
                    WireExchange(
                        operation=operation,
                        request_json=request_json,
                        response_json=raw,
                    )
                )
                response = json.loads(raw)
                expected = "state" if operation == "state" else "observation"
                if (
                    not isinstance(response, dict)
                    or response.get("type") != expected
                    or not isinstance(response.get("data"), dict)
                ):
                    raise ValueError("unexpected response envelope")
                return response["data"]

            reset = dict(plan.reset.options)
            reset.update(seed=plan.reset.seed, episode_id=plan.reset.episode_id)
            observation = exchange("reset", reset)
            exchange("state")
            for action in plan.actions:
                if observation.get("done") is True:
                    break
                observation = exchange("step", action)
                exchange("state")
            completed = True
            send(socket, json.dumps({"type": "close"}))
        return RuntimeEvidence(
            exchanges=tuple(exchanges), observation_schema_json=schema_json
        )
    except Exception as exc:
        if completed:
            return RuntimeEvidence(
                exchanges=tuple(exchanges), observation_schema_json=schema_json
            )
        # Exception text may include submitted payloads or URL credentials.
        return RuntimeEvidence(
            exchanges=tuple(exchanges),
            observation_schema_json=schema_json,
            failure_phase=phase,
            failure_reason=f"{phase} failed ({type(exc).__name__})",
        )
