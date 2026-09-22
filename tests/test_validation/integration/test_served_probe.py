"""Test fixture faults over the production OpenEnv WebSocket endpoint."""

import importlib.util
from pathlib import Path

import pytest
from starlette.testclient import TestClient


FIXTURE = Path(__file__).parents[2] / "fixtures/validation/runtime/served_probe/app.py"
spec = importlib.util.spec_from_file_location("served_probe_app", FIXTURE)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def observation(ws, request):
    ws.send_json(request)
    result = ws.receive_json()
    assert result["type"] == "observation"
    return result["data"]


def test_real_session_retains_counter_and_requested_identity():
    with TestClient(fixture.make_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/schema").json()["observation"]["properties"]["counter"]
        with client.websocket_connect("/ws") as ws:
            initial = observation(
                ws, {"type": "reset", "data": {"episode_id": "test", "seed": 42}}
            )
            assert initial["observation"]["counter"] == 0
            for index in (1, 2):
                result = observation(ws, {"type": "step", "data": {"increment": 1}})
                assert result["observation"]["counter"] == index
                assert result["done"] is (index == 2)
                ws.send_json({"type": "state"})
                assert ws.receive_json()["data"] == {
                    "episode_id": "test",
                    "step_count": index,
                }
        with client.websocket_connect("/ws") as ws:
            result = observation(ws, {"type": "reset", "data": {"episode_id": "fresh"}})
            assert result["observation"]["counter"] == 0


@pytest.mark.parametrize(
    "mode", ["bad_reward", "bad_observation", "missing_done", "bad_state"]
)
def test_fault_is_visible_in_raw_wire_response(mode):
    with TestClient(fixture.make_app(mode)) as client:
        with client.websocket_connect("/ws") as ws:
            result = observation(ws, {"type": "reset", "data": {"episode_id": "test"}})
            if mode == "bad_reward":
                assert result["reward"] is True
            elif mode == "bad_observation":
                assert result["observation"]["counter"] == "not-an-integer"
            elif mode == "missing_done":
                assert "done" not in result
            else:
                ws.send_json({"type": "state"})
                assert ws.receive_json()["data"]["episode_id"] == "wrong-episode"


def test_failed_start_is_explicit():
    with pytest.raises(RuntimeError, match="Deliberate startup failure"):
        fixture.make_app("startup_failure")
