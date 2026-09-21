"""Hostile HTTP schema responses must stay bounded before JSON/schema grading."""

import json

import httpx
import pytest
from openenv.validation.runtime import collector
from openenv.validation.runtime.contracts import RuntimePlan


class TrackedStream(httpx.SyncByteStream):
    def __init__(self, payload, *, reject_read=False):
        self.payload = payload
        self.reject_read = reject_read
        self.read = False
        self.closed = False

    def __iter__(self):
        self.read = True
        if self.reject_read:
            raise AssertionError("Compressed bytes must never reach the decoder")
        yield self.payload

    def close(self):
        self.closed = True


@pytest.fixture
def plan():
    return RuntimePlan.model_validate(
        {
            "plan_schema_version": "1",
            "reset": {"episode_id": "bounded", "seed": 7},
            "actions": [{"increment": 1}],
        }
    )


def schema_transport(monkeypatch, stream, headers=None):
    requests = []
    original_client = httpx.Client

    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers=headers or {}, stream=stream)

    monkeypatch.setattr(
        collector.httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(respond), **kwargs
        ),
    )

    def no_websocket(*args, **kwargs):
        raise ConnectionError("stop after checking schema retrieval")

    monkeypatch.setattr(collector, "connect", no_websocket)
    return requests


class FakeSocket:
    def __init__(self, *, close_send_error=False, exit_error=False):
        self.socket = self
        self.timeout = None
        self.close_send_error = close_send_error
        self.exit_error = exit_error
        self.send_timeouts = []
        self.responses = iter(
            [
                {
                    "type": "observation",
                    "data": {
                        "observation": {"counter": 0},
                        "reward": None,
                        "done": False,
                    },
                },
                {
                    "type": "state",
                    "data": {"episode_id": "bounded", "step_count": 0},
                },
                {
                    "type": "observation",
                    "data": {
                        "observation": {"counter": 1},
                        "reward": 1,
                        "done": False,
                    },
                },
                {
                    "type": "state",
                    "data": {"episode_id": "bounded", "step_count": 1},
                },
            ]
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.exit_error:
            raise TimeoutError("slow close handshake")

    def gettimeout(self):
        return self.timeout

    def settimeout(self, timeout):
        self.timeout = timeout

    def send(self, message):
        self.send_timeouts.append(self.timeout)
        if self.close_send_error and json.loads(message)["type"] == "close":
            raise TimeoutError("slow protocol close")

    def recv(self, timeout):
        return json.dumps(next(self.responses))


def successful_socket(monkeypatch, socket):
    stream = TrackedStream(json.dumps({"observation": {"type": "object"}}).encode())
    schema_transport(monkeypatch, stream)
    monkeypatch.setattr(collector, "connect", lambda *args, **kwargs: socket)


@pytest.mark.parametrize(
    "encoding", ["gzip", "deflate", "br", "zstd", "gzip, identity", "unknown"]
)
def test_compressed_schema_is_rejected_before_body_read(monkeypatch, plan, encoding):
    stream = TrackedStream(b"untrusted compressed bytes", reject_read=True)
    requests = schema_transport(monkeypatch, stream, {"Content-Encoding": encoding})
    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000", plan, episode_timeout_s=2
    )
    assert requests[0].headers["Accept-Encoding"] == "identity"
    assert not stream.read
    assert stream.closed
    assert evidence.failure_phase == "schema"
    assert evidence.failure_reason == "schema failed (ValueError)"
    assert evidence.observation_schema_json is None
    assert evidence.exchanges == ()


@pytest.mark.parametrize(
    "headers",
    [{}, {"Content-Encoding": "identity"}, {"Content-Encoding": " Identity "}],
)
def test_uncompressed_schema_is_read_and_retained(monkeypatch, plan, headers):
    schema = {"type": "object"}
    stream = TrackedStream(json.dumps({"observation": schema}).encode())
    requests = schema_transport(monkeypatch, stream, headers)
    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000", plan, episode_timeout_s=2
    )
    assert requests[0].headers["Accept-Encoding"] == "identity"
    assert stream.read and stream.closed
    assert json.loads(evidence.observation_schema_json) == schema
    assert evidence.failure_phase == "connect"


def test_uncompressed_schema_still_obeys_total_byte_budget(monkeypatch, plan):
    stream = TrackedStream(b"x" * 65)
    schema_transport(monkeypatch, stream)
    monkeypatch.setattr(collector, "MAX_MESSAGE_BYTES", 64)
    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000", plan, episode_timeout_s=2
    )
    assert stream.read and stream.closed
    assert evidence.failure_phase == "schema"
    assert evidence.failure_reason == "schema failed (ValueError)"
    assert evidence.observation_schema_json is None


@pytest.mark.parametrize("close_send_error,exit_error", [(True, False), (False, True)])
def test_close_errors_do_not_fail_completed_episode(
    monkeypatch, plan, close_send_error, exit_error
):
    socket = FakeSocket(
        close_send_error=close_send_error,
        exit_error=exit_error,
    )
    successful_socket(monkeypatch, socket)

    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000",
        plan,
        episode_timeout_s=2,
        request_timeout_s=0.25,
    )

    assert len(evidence.exchanges) == 4
    assert evidence.failure_reason is None


def test_all_websocket_sends_are_bounded_by_request_timeout(monkeypatch, plan):
    socket = FakeSocket()
    successful_socket(monkeypatch, socket)

    evidence = collector.collect_runtime_evidence(
        "http://127.0.0.1:8000",
        plan,
        episode_timeout_s=2,
        request_timeout_s=0.25,
    )

    assert evidence.failure_reason is None
    assert len(socket.send_timeouts) == 5
    assert all(0 < timeout <= 0.25 for timeout in socket.send_timeouts)
    assert socket.timeout is None
