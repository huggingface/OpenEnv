# SPDX-License-Identifier: BSD-3-Clause

"""OpenCodeSessionFactory.create() must tear the sandbox down on any post-provision failure."""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opencode_env.config import OpenCodeConfig  # noqa: E402
from opencode_env.harness import OpenCodeSessionFactory  # noqa: E402


class _FakeSandbox:
    def __init__(self):
        self.sandbox_id = "fake"
        self.killed = False

    def kill(self):
        self.killed = True


class _FakeBackend:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    def create(self, **kwargs):
        return self._sandbox


def _factory(sandbox, **overrides):
    return OpenCodeSessionFactory(
        config=OpenCodeConfig(base_url="http://localhost:8000/v1"),
        sandbox_backend=_FakeBackend(sandbox),
        **overrides,
    )


def test_create_kills_sandbox_when_bootstrap_fails(monkeypatch):
    sandbox = _FakeSandbox()
    factory = _factory(sandbox, create_attempts=1)
    monkeypatch.setattr(
        factory,
        "_bootstrap_sandbox",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        factory.create("write a function")
    assert sandbox.killed is True


def test_create_preserves_root_cause_if_kill_also_fails(monkeypatch):
    class _KillRaises(_FakeSandbox):
        def kill(self):
            raise RuntimeError("kill failed")

    sandbox = _KillRaises()
    factory = _factory(sandbox, create_attempts=1)
    monkeypatch.setattr(
        factory,
        "_bootstrap_sandbox",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    # The original bootstrap failure must surface, not the cleanup error.
    with pytest.raises(RuntimeError, match="boom"):
        factory.create("write a function")


def test_create_retries_then_succeeds(monkeypatch):
    # create() retries a flaky _create_once and returns once it succeeds.
    sandbox = _FakeSandbox()
    factory = _factory(sandbox, create_attempts=3, create_backoff_s=0)
    calls = {"n": 0}
    session = object()

    def _flaky(task, seed=None, episode_id=None, start_agent=True):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient create failure")
        return session

    monkeypatch.setattr(factory, "_create_once", _flaky)
    assert factory.create("write a function") is session
    assert calls["n"] == 3


def test_create_raises_after_exhausting_attempts(monkeypatch):
    # A persistent failure is re-raised after create_attempts tries.
    sandbox = _FakeSandbox()
    factory = _factory(sandbox, create_attempts=3, create_backoff_s=0)
    calls = {"n": 0}

    def _always_fails(task, seed=None, episode_id=None, start_agent=True):
        calls["n"] += 1
        raise RuntimeError("persistent create failure")

    monkeypatch.setattr(factory, "_create_once", _always_fails)
    with pytest.raises(RuntimeError, match="persistent create failure"):
        factory.create("write a function")
    assert calls["n"] == 3
