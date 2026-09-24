"""Bounded raw protocol collection in one OpenEnv orchestration session."""

import json
import socket
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import httpx
from websockets.sync.client import connect

from .contracts import RuntimeEvidence, RuntimePlan, WireExchange
from .discovery import collect_task_evidence

MAX_MESSAGE_BYTES = 1024 * 1024
MAX_TRACE_BYTES = 8 * 1024 * 1024


def _abort_transport(connection):
    # The send thread may hold websockets' protocol lock. Shut down the raw
    # transport directly so sendall and its concurrent receiver can both exit.
    try:
        connection.socket.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # The peer or another cleanup path may already have closed it.
    try:
        connection.socket.close()
    except OSError:
        pass  # A concurrent close must not replace the original operation error.


def _bounded_call(connection, operation, timeout_s):
    if timeout_s <= 0:
        raise TimeoutError("episode deadline exceeded")
    expired = threading.Event()
    deadline = time.monotonic() + timeout_s

    def abort():
        expired.set()
        _abort_transport(connection)

    watchdog = threading.Timer(timeout_s, abort)
    watchdog.daemon = True
    watchdog.start()
    try:
        operation()
    except KeyboardInterrupt:
        _abort_transport(connection)
        raise
    except Exception:
        if expired.is_set():
            raise TimeoutError("transport deadline exceeded") from None
        raise
    finally:
        watchdog.cancel()
        watchdog.join()
    if expired.is_set() or time.monotonic() >= deadline:
        _abort_transport(connection)
        raise TimeoutError("transport deadline exceeded")


class RuntimeCollectionInterrupted(KeyboardInterrupt):
    """Cancellation carrying the completed, immutable episode prefix."""

    def __init__(self, evidence: RuntimeEvidence):
        super().__init__("runtime collection interrupted")
        self.evidence = evidence


def collect_runtime_evidence(
    base_url: str,
    plan: RuntimePlan,
    *,
    episode_timeout_s: float,
    request_timeout_s: float = 5.0,
    validation_token: str | None = None,
    collect_tools: bool = False,
    task_env_name: str | None = None,
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
        validation_token (`str`, *optional*):
            Run-scoped telemetry authorization; never retained in evidence.
        collect_tools (`bool`, *optional*, defaults to `False`):
            Discover tools on the measured WebSocket before reset.
        task_env_name (`str`, *optional*):
            Sample this environment's task metadata through its HTTP task API.

    Returns:
        [`~openenv.validation.runtime.contracts.RuntimeEvidence`]: raw evidence.
    """
    deadline = time.monotonic() + episode_timeout_s
    exchanges = []
    schema_json = None
    phase = "schema"
    trace_bytes = 0
    telemetry_json = None
    telemetry_error = None
    tools_json = tools_error = tasks_json = tasks_error = None

    def remaining() -> float:
        value = min(request_timeout_s, deadline - time.monotonic())
        if value <= 0:
            raise TimeoutError("episode deadline exceeded")
        return value

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
            schema_payload = json.dumps(
                schema["observation"],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if validation_token and validation_token in schema_payload:
                raise ValueError("schema contains validation credentials")
            schema_json = schema_payload

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
        connection = connect(
            ws_url,
            proxy=None,
            open_timeout=remaining(),
            close_timeout=1,
            max_size=MAX_TRACE_BYTES if validation_token else MAX_MESSAGE_BYTES,
            max_queue=1,
            compression=None,
        )
        complete = False
        try:

            def telemetry_request(operation, data, max_bytes=MAX_TRACE_BYTES):
                request = json.dumps({"type": operation, "data": data})
                _bounded_call(connection, lambda: connection.send(request), remaining())
                raw = connection.recv(timeout=remaining())
                if not isinstance(raw, str) or len(raw.encode()) > max_bytes:
                    raise ValueError("invalid telemetry response")
                response = json.loads(raw)
                if not isinstance(response, dict):
                    raise ValueError("invalid telemetry envelope")
                return response

            capability = None
            if validation_token:
                phase = "validation_open"
                try:
                    response = telemetry_request(
                        phase, {"schema_version": 1, "token": validation_token}
                    )
                except (ValueError, RecursionError) as exc:
                    # A consumed malformed reply only invalidates optional telemetry.
                    # Transport failure still aborts this same-session collection.
                    telemetry_error = f"session telemetry failed ({type(exc).__name__})"
                else:
                    data = response.get("data")
                    if (
                        response.get("type") == "validation_open"
                        and isinstance(data, dict)
                        and data.get("schema_version") == 1
                        and isinstance(data.get("capability"), str)
                        and 16 <= len(data["capability"]) <= 256
                    ):
                        capability = data["capability"]
                    else:
                        telemetry_error = (
                            "session telemetry unavailable or authorization refused"
                        )

            def contains_credential(value):
                if isinstance(value, str):
                    return any(
                        secret and secret in value
                        for secret in (validation_token, capability)
                    )
                if isinstance(value, dict):
                    return any(
                        contains_credential(key) or contains_credential(child)
                        for key, child in value.items()
                    )
                if isinstance(value, list):
                    return any(contains_credential(child) for child in value)
                return False

            if collect_tools:
                phase = "tools/list"
                try:
                    response = telemetry_request(
                        "mcp",
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/list",
                            "params": {},
                        },
                        MAX_MESSAGE_BYTES,
                    )
                    if contains_credential(response):
                        raise ValueError("discovery contains validation credentials")
                    rpc = response.get("data")
                    if (
                        response.get("type") != "mcp"
                        or not isinstance(rpc, dict)
                        or rpc.get("jsonrpc") != "2.0"
                        or type(rpc.get("id")) is not int
                        or rpc["id"] != 1
                        or rpc.get("error") is not None
                        or not isinstance(rpc.get("result"), dict)
                        or not isinstance(rpc["result"].get("tools"), list)
                    ):
                        raise ValueError("invalid tool discovery response")
                    discovered = json.dumps(
                        rpc["result"], allow_nan=False, ensure_ascii=False
                    )
                    if len(discovered.encode()) > MAX_MESSAGE_BYTES:
                        raise ValueError("discovery exceeds size bound")
                    tools_json = discovered
                except Exception as exc:
                    tools_error = f"tool discovery failed ({type(exc).__name__})"

            if task_env_name is not None:
                phase = "tasks"
                try:
                    discovered, tasks_error = collect_task_evidence(
                        base_url,
                        task_env_name,
                        deadline=deadline,
                        request_timeout_s=request_timeout_s,
                    )
                    if discovered is not None:
                        if contains_credential(discovered) or contains_credential(
                            json.loads(discovered)
                        ):
                            raise ValueError(
                                "discovery contains validation credentials"
                            )
                        tasks_json = discovered
                except Exception as exc:
                    tasks_error = f"task discovery failed ({type(exc).__name__})"

            def exchange(operation: str, data: dict | None = None) -> dict:
                nonlocal phase, trace_bytes
                phase = operation
                request = {"type": operation}
                if data is not None:
                    request["data"] = data
                request_json = json.dumps(request, allow_nan=False)
                _bounded_call(
                    connection, lambda: connection.send(request_json), remaining()
                )
                raw = connection.recv(timeout=remaining())
                if not isinstance(raw, str):
                    raise ValueError("binary response is not the JSON protocol")
                if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
                    raise ValueError("response exceeds size bound")
                trace_bytes += len(raw.encode("utf-8")) + len(
                    request_json.encode("utf-8")
                )
                if trace_bytes > MAX_TRACE_BYTES:
                    raise ValueError("trace exceeds total size bound")
                # Subjects can echo the authorization into arbitrary JSON fields.
                # Check decoded values as well as raw text (which may be malformed).
                if contains_credential(raw):
                    raise ValueError("response contains validation credentials")
                try:
                    parsed = json.loads(raw)
                except (ValueError, RecursionError):
                    parsed = None
                if contains_credential(parsed):
                    raise ValueError("response contains validation credentials")
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
            if capability:
                phase = "validation_read"
                try:
                    response = telemetry_request(
                        phase, {"schema_version": 1, "capability": capability}
                    )
                    if response.get("type") != "validation" or not isinstance(
                        response.get("data"), dict
                    ):
                        raise ValueError("invalid telemetry envelope")
                    if contains_credential(response["data"]):
                        raise ValueError("telemetry contains validation credentials")
                    snapshot = json.dumps(
                        response["data"],
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    if len(snapshot.encode("utf-8")) > MAX_TRACE_BYTES:
                        raise ValueError("telemetry exceeds size bound")
                    telemetry_json = snapshot
                except Exception as exc:
                    telemetry_error = f"session telemetry failed ({type(exc).__name__})"
            complete = True
        finally:
            # Teardown is best effort and cannot replace an in-episode failure
            # or invalidate an otherwise completely measured episode.
            if complete:
                try:
                    _bounded_call(
                        connection,
                        lambda: connection.send(json.dumps({"type": "close"})),
                        min(remaining(), 1.0),
                    )
                except (Exception, KeyboardInterrupt):
                    pass  # A server may close immediately after its final response.
            try:
                _bounded_call(connection, connection.close, 1.0)
            except (Exception, KeyboardInterrupt):
                _abort_transport(connection)
        return RuntimeEvidence(
            exchanges=tuple(exchanges),
            observation_schema_json=schema_json,
            telemetry_json=telemetry_json,
            telemetry_error=telemetry_error,
            tools_json=tools_json,
            tools_error=tools_error,
            tasks_json=tasks_json,
            tasks_error=tasks_error,
        )
    except KeyboardInterrupt:
        raise RuntimeCollectionInterrupted(
            RuntimeEvidence(
                exchanges=tuple(exchanges),
                observation_schema_json=schema_json,
                failure_phase=phase,
                failure_reason=f"{phase} failed (KeyboardInterrupt)",
                telemetry_json=telemetry_json,
                telemetry_error=telemetry_error,
                tools_json=tools_json,
                tools_error=tools_error,
                tasks_json=tasks_json,
                tasks_error=tasks_error,
            )
        ) from None
    except Exception as exc:
        # Exception text may include submitted payloads or URL credentials.
        return RuntimeEvidence(
            exchanges=tuple(exchanges),
            observation_schema_json=schema_json,
            failure_phase=phase,
            failure_reason=f"{phase} failed ({type(exc).__name__})",
            telemetry_json=telemetry_json,
            telemetry_error=telemetry_error,
            tools_json=tools_json,
            tools_error=tools_error,
            tasks_json=tasks_json,
            tasks_error=tasks_error,
        )
