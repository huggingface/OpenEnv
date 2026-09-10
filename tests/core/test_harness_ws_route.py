# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the production /harness WebSocket route and mode wiring."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import create_fastapi_app, HTTPEnvServer
from openenv.core.env_server.types import Observation, ServerMode
from openenv.core.harness import (
    AgenticHarnessAdapter,
    HarnessAction,
    HarnessConfig,
    HarnessEnvironment,
    HarnessEvent,
    HarnessEventType,
)


class FakeAdapter(AgenticHarnessAdapter):
    """Scripted adapter recording lifecycle calls; no real subprocess."""

    created: list["FakeAdapter"] = []

    def __init__(self):
        super().__init__(HarnessConfig(name="fake-harness", command=["fake"]))
        self.calls: list[str] = []
        self.alive = False
        self.fail_on_send = False
        self.hang_on_send = False
        self.omit_turn_complete = False
        FakeAdapter.created.append(self)

    async def start(self, working_directory: str) -> None:
        self.calls.append("start")
        self.alive = True

    async def stop(self) -> None:
        self.calls.append("stop")
        self.alive = False

    async def inject_tools(self, tools, bridge_url: Optional[str] = None) -> None:
        self.calls.append("inject_tools")

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        self.calls.append(f"send:{message}")
        if self.hang_on_send:
            await asyncio.sleep(3600)
        if self.fail_on_send:
            raise RuntimeError("adapter blew up")
        if self.omit_turn_complete:
            yield HarnessEvent(
                type=HarnessEventType.TEXT_OUTPUT, data={"text": "no terminal event"}
            )
            return
        yield HarnessEvent(type=HarnessEventType.TOOL_CALL, data={"tool_name": "shell"})
        yield HarnessEvent(
            type=HarnessEventType.TURN_COMPLETE,
            data={"response": f"handled: {message}", "done": False},
        )

    async def is_alive(self) -> bool:
        return self.alive


def harness_env_factory() -> HarnessEnvironment:
    # mcp=None: no domain tools, so no bridge server is started in tests
    return HarnessEnvironment(adapter=FakeAdapter(), mcp=None)


def make_app(
    mode: ServerMode, env_factory=harness_env_factory, max_concurrent_envs: int = 1
) -> tuple[FastAPI, HTTPEnvServer]:
    # HarnessEnvironment is SUPPORTS_CONCURRENT_SESSIONS=False (one env = one
    # trajectory), so the server only permits max_concurrent_envs=1.
    app = FastAPI()
    server = HTTPEnvServer(
        env_factory,
        HarnessAction,
        Observation,
        max_concurrent_envs=max_concurrent_envs,
    )
    server.register_routes(app, mode=mode)
    return app, server


@pytest.fixture(autouse=True)
def clear_created_adapters():
    FakeAdapter.created = []
    yield
    FakeAdapter.created = []


class TestRouteRegistration:
    def test_registered_in_production_with_harness_factory(self):
        app, _ = make_app(ServerMode.PRODUCTION)
        assert any(getattr(r, "path", None) == "/harness" for r in app.routes)

    def test_absent_in_simulation(self):
        app, _ = make_app(ServerMode.SIMULATION)
        assert not any(getattr(r, "path", None) == "/harness" for r in app.routes)

    def test_absent_for_non_harness_env(self):
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import State

        class PlainEnv(Environment):
            def reset(self, **kwargs):
                return Observation()

            def step(self, action, timeout_s=None, **kwargs):
                return Observation()

            @property
            def state(self):
                return State()

        app, _ = make_app(ServerMode.PRODUCTION, env_factory=PlainEnv)
        assert not any(getattr(r, "path", None) == "/harness" for r in app.routes)

    def test_production_still_excludes_orchestration_routes(self):
        app, _ = make_app(ServerMode.PRODUCTION)
        client = TestClient(app)
        assert client.post("/reset", json={}).status_code in (404, 405)
        assert client.post("/step", json={"action": {}}).status_code in (404, 405)
        assert client.get("/state").status_code in (404, 405)
        assert client.get("/health").status_code == 200

    def test_detection_probe_is_side_effect_free(self):
        make_app(ServerMode.PRODUCTION)
        # The factory-detection probe instantiates one env; it must not
        # start the harness.
        for adapter in FakeAdapter.created:
            assert "start" not in adapter.calls


class TestHarnessWebSocket:
    def test_session_started_then_streamed_turns(self):
        app, server = make_app(ServerMode.PRODUCTION)
        client = TestClient(app)

        with client.websocket_connect("/harness") as websocket:
            started = json.loads(websocket.receive_text())
            assert started["type"] == "session_started"
            assert started["data"]["harness"] == "fake-harness"
            assert started["data"]["session_id"]

            websocket.send_text(json.dumps({"type": "message", "content": "hi"}))
            first = json.loads(websocket.receive_text())
            assert first["type"] == "tool_call"
            second = json.loads(websocket.receive_text())
            assert second["type"] == "turn_complete"
            assert second["data"]["response"] == "handled: hi"

    def test_two_turns_reuse_one_adapter(self):
        app, _ = make_app(ServerMode.PRODUCTION)
        client = TestClient(app)

        with client.websocket_connect("/harness") as websocket:
            websocket.receive_text()  # session_started
            for content in ("one", "two"):
                websocket.send_text(json.dumps({"type": "message", "content": content}))
                websocket.receive_text()  # tool_call
                websocket.receive_text()  # turn_complete

        # One adapter served the whole connection (probe adapters saw no start)
        session_adapters = [a for a in FakeAdapter.created if "start" in a.calls]
        assert len(session_adapters) == 1
        adapter = session_adapters[0]
        assert adapter.calls.count("start") == 1
        assert [c for c in adapter.calls if c.startswith("send:")] == [
            "send:one",
            "send:two",
        ]

    def test_malformed_json_keeps_connection_usable(self):
        app, _ = make_app(ServerMode.PRODUCTION)
        client = TestClient(app)

        with client.websocket_connect("/harness") as websocket:
            websocket.receive_text()  # session_started
            websocket.send_text("this is not json")
            error = json.loads(websocket.receive_text())
            assert error["type"] == "error"
            assert error["data"]["code"] == "INVALID_JSON"

            websocket.send_text(json.dumps({"type": "bogus"}))
            error = json.loads(websocket.receive_text())
            assert error["type"] == "error"
            assert error["data"]["code"] == "VALIDATION_ERROR"

            websocket.send_text(json.dumps({"type": "message", "content": "ok"}))
            assert json.loads(websocket.receive_text())["type"] == "tool_call"

    def test_adapter_crash_streams_error_event_then_closes(self):
        app, server = make_app(ServerMode.PRODUCTION)
        client = TestClient(app)

        with client.websocket_connect("/harness") as websocket:
            websocket.receive_text()  # session_started
            adapter = [a for a in FakeAdapter.created if a.alive][0]
            adapter.fail_on_send = True

            websocket.send_text(json.dumps({"type": "message", "content": "boom"}))
            error_event = json.loads(websocket.receive_text())
            assert error_event["type"] == "error"
            assert error_event["data"]["recoverable"] is False
            assert "adapter blew up" in error_event["data"]["message"]

        assert server.active_sessions == 0

    def test_disconnect_stops_adapter_and_frees_session(self):
        app, server = make_app(ServerMode.PRODUCTION)
        client = TestClient(app)

        with client.websocket_connect("/harness") as websocket:
            websocket.receive_text()  # session_started
            session_adapter = [a for a in FakeAdapter.created if a.alive][0]
            assert server.active_sessions == 1

        assert server.active_sessions == 0
        assert "stop" in session_adapter.calls
        assert session_adapter.alive is False

    def test_capacity_limit_rejects_second_connection(self):
        app, _ = make_app(ServerMode.PRODUCTION, max_concurrent_envs=1)
        client = TestClient(app)

        with client.websocket_connect("/harness") as first:
            first.receive_text()  # session_started
            with client.websocket_connect("/harness") as second:
                error = json.loads(second.receive_text())
                assert error["type"] == "error"
                assert error["data"]["code"] == "CAPACITY_REACHED"


class TestModeWiring:
    def test_create_fastapi_app_mode_production(self):
        app = create_fastapi_app(
            harness_env_factory, HarnessAction, Observation, mode="production"
        )
        client = TestClient(app)
        assert client.post("/reset", json={}).status_code in (404, 405)
        assert any(getattr(r, "path", None) == "/harness" for r in app.routes)

    def test_openenv_mode_env_var_honored(self, monkeypatch):
        monkeypatch.setenv("OPENENV_MODE", "production")
        app = create_fastapi_app(harness_env_factory, HarnessAction, Observation)
        client = TestClient(app)
        assert client.post("/reset", json={}).status_code in (404, 405)

    def test_default_stays_simulation(self, monkeypatch):
        monkeypatch.delenv("OPENENV_MODE", raising=False)
        app = create_fastapi_app(harness_env_factory, HarnessAction, Observation)
        assert not any(getattr(r, "path", None) == "/harness" for r in app.routes)
        route_paths = {getattr(r, "path", None) for r in app.routes}
        assert "/reset" in route_paths

    def test_explicit_mode_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENENV_MODE", "production")
        app = create_fastapi_app(
            harness_env_factory, HarnessAction, Observation, mode="simulation"
        )
        route_paths = {getattr(r, "path", None) for r in app.routes}
        assert "/reset" in route_paths


class TestTurnBoundaries:
    """A production turn must end: on its own, on timeout, or with an error."""

    def test_hung_turn_is_bounded_by_session_timeout(self):
        app, server = make_app(ServerMode.PRODUCTION)
        client = TestClient(app)

        with client.websocket_connect("/harness") as websocket:
            websocket.receive_text()  # session_started
            adapter = [a for a in FakeAdapter.created if a.alive][0]
            adapter.hang_on_send = True
            adapter.config.session_timeout_s = 0.2

            websocket.send_text(json.dumps({"type": "message", "content": "hang"}))
            event = json.loads(websocket.receive_text())
            assert event["type"] == "error"
            assert "exceeded 0.2 seconds" in event["data"]["message"]

        # The session is released rather than pinned at capacity forever.
        assert server.active_sessions == 0

    def test_stream_without_turn_complete_is_reported(self):
        app, server = make_app(ServerMode.PRODUCTION)
        client = TestClient(app)

        with client.websocket_connect("/harness") as websocket:
            websocket.receive_text()  # session_started
            adapter = [a for a in FakeAdapter.created if a.alive][0]
            adapter.omit_turn_complete = True

            websocket.send_text(json.dumps({"type": "message", "content": "hi"}))
            assert json.loads(websocket.receive_text())["type"] == "text_output"
            # A client blocking on the terminal event would otherwise hang.
            event = json.loads(websocket.receive_text())
            assert event["type"] == "error"
            assert "TURN_COMPLETE" in event["data"]["message"]

        assert server.active_sessions == 0
