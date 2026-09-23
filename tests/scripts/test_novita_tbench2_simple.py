"""Regression tests for the Novita Terminal-Bench example."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_example(monkeypatch: pytest.MonkeyPatch):
    tbench2_env = types.ModuleType("tbench2_env")
    tbench2_env.Tbench2Action = object
    tbench2_env.Tbench2Env = object
    monkeypatch.setitem(sys.modules, "tbench2_env", tbench2_env)

    path = Path(__file__).parents[2] / "examples" / "novita_tbench2_simple.py"
    spec = importlib.util.spec_from_file_location("novita_tbench2_simple", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readiness_timeout_stops_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    example = _load_example(monkeypatch)
    events: list[str] = []

    class FakeProvider:
        @staticmethod
        def image_from_dockerfile(path: str) -> str:
            assert path == "envs/tbench2_env/server/Dockerfile"
            return "template:test"

        def start_container(self, *, image: str) -> str:
            assert image == "template:test"
            events.append("start")
            return "https://sandbox.test"

        def wait_for_ready(self, base_url: str, *, timeout_s: float) -> None:
            assert base_url == "https://sandbox.test"
            assert timeout_s == 300
            events.append("wait")
            raise TimeoutError("sandbox did not become ready")

        def stop_container(self) -> None:
            events.append("stop")

    monkeypatch.setattr(example, "NovitaSandboxProvider", FakeProvider)
    monkeypatch.setenv("TB2_TASKS_DIR", "/tmp/tasks")

    with pytest.raises(TimeoutError, match="did not become ready"):
        asyncio.run(example.main())

    assert events == ["start", "wait", "stop"]
