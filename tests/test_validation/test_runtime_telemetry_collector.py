"""Telemetry cannot expand its byte budget or retain replay credentials."""

import json

import httpx
import pytest
from openenv.validation.runtime import collector
from openenv.validation.runtime.contracts import RuntimePlan

TOKEN = "run-authorization-" + "x" * 32
CAPABILITY = "socket-capability-" + "y" * 32


def collect(monkeypatch, *, snapshot=None, leaked_response=None, escaped=False):
    original = httpx.Client
    monkeypatch.setattr(
        collector.httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"observation": {}})
            ),
            **kwargs,
        ),
    )

    class Connection:
        def send(self, raw):
            self.request = json.loads(raw)

        def recv(self, timeout):
            operation = self.request["type"]
            if operation == "validation_open":
                response = {
                    "type": operation,
                    "data": {"schema_version": 1, "capability": CAPABILITY},
                }
            elif operation == "validation_read":
                response = {
                    "type": "validation",
                    "data": snapshot or {"schema_version": 1},
                }
            elif operation == "state":
                response = {
                    "type": "state",
                    "data": {"episode_id": "test", "step_count": 0},
                }
            else:
                response = {
                    "type": "observation",
                    "data": {
                        "observation": {"message": leaked_response},
                        "done": False,
                        "reward": 0.0,
                    },
                }
            raw = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
            if escaped and leaked_response:
                raw = raw.replace(
                    leaked_response,
                    "".join(f"\\u{ord(c):04x}" for c in leaked_response),
                )
            return raw

        def close(self):
            pass

    monkeypatch.setattr(collector, "connect", lambda *args, **kwargs: Connection())
    plan = RuntimePlan.model_validate(
        {
            "plan_schema_version": "1",
            "reset": {"episode_id": "test", "seed": 1},
            "actions": [{"increment": 1}],
        }
    )
    return collector.collect_runtime_evidence(
        "http://127.0.0.1:8000", plan, episode_timeout_s=2, validation_token=TOKEN
    )


def test_utf8_telemetry_stays_inside_received_budget(monkeypatch):
    monkeypatch.setattr(collector, "MAX_TRACE_BYTES", 2048)
    result = collect(monkeypatch, snapshot={"schema_version": 1, "text": "😀" * 400})
    assert result.telemetry_error is None
    assert len(result.telemetry_json.encode()) <= 2048
    assert "😀" in result.telemetry_json


@pytest.mark.parametrize("credential", [TOKEN, CAPABILITY])
def test_telemetry_cannot_persist_credentials_under_arbitrary_keys(
    monkeypatch, credential
):
    result = collect(monkeypatch, snapshot={"schema_version": 1, "debug": [credential]})
    assert result.telemetry_json is None
    assert result.telemetry_error == "session telemetry failed (ValueError)"
    assert result.failure_reason is None
    assert len(result.exchanges) == 4
    assert credential not in repr(result)


@pytest.mark.parametrize(
    "credential,escaped",
    [(TOKEN, False), (CAPABILITY, False), (TOKEN, True), (CAPABILITY, True)],
)
def test_normal_wire_cannot_persist_plain_or_escaped_credentials(
    monkeypatch, credential, escaped
):
    result = collect(monkeypatch, leaked_response=credential, escaped=escaped)
    assert result.exchanges == ()
    assert result.failure_reason == "reset failed (ValueError)"
    assert credential not in repr(result)
