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
from opencode_env.opencode_runtime import build_install_cmd  # noqa: E402
from opencode_env.sandbox.base import ExecResult  # noqa: E402
from opencode_env.task import OpenCodeTask  # noqa: E402


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


class _ExecScriptedSandbox(_FakeSandbox):
    """exec() returns a fixed failure; echo-ok probes succeed."""

    def __init__(self, failure):
        super().__init__()
        self._failure = failure
        self.exec_calls = []

    def exec(self, cmd, timeout=None):
        self.exec_calls.append(cmd)
        if cmd == "echo ok":
            return ExecResult(exit_code=0, stdout="ok", stderr="")
        return self._failure


def test_exec_with_retry_does_not_retry_fatal_stdout_failure():
    # The opencode installer reports version-resolution failure on stdout with
    # an empty stderr, which the transient heuristic would otherwise retry.
    failure = ExecResult(
        exit_code=1, stdout="Failed to fetch version information", stderr=""
    )
    sandbox = _ExecScriptedSandbox(failure)
    factory = _factory(sandbox)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        factory._exec_with_retry(
            sandbox,
            "install",
            timeout=1,
            attempts=3,
            backoff_s=0,
            fatal_markers=("Failed to fetch version information",),
        )
    assert len(sandbox.exec_calls) == 1


def test_exec_with_retry_still_retries_silent_failures():
    failure = ExecResult(exit_code=137, stdout="", stderr="")
    sandbox = _ExecScriptedSandbox(failure)
    factory = _factory(sandbox)
    with pytest.raises(RuntimeError):
        factory._exec_with_retry(
            sandbox,
            "cmd",
            timeout=1,
            attempts=3,
            backoff_s=0,
            fatal_markers=("Failed to fetch version information",),
        )
    assert len(sandbox.exec_calls) == 3


def test_bootstrap_install_rate_limit_raises_actionable_error(monkeypatch):
    failure = ExecResult(
        exit_code=1, stdout="Failed to fetch version information", stderr=""
    )
    sandbox = _ExecScriptedSandbox(failure)
    factory = _factory(sandbox)
    monkeypatch.setattr(factory, "_opencode_already_installed", lambda s: False)
    with pytest.raises(RuntimeError, match="Pin opencode_version"):
        factory._bootstrap_sandbox(
            sandbox, OpenCodeTask(instruction="write a function")
        )
    # one echo-ok probe + exactly one install attempt, no retries
    assert len([c for c in sandbox.exec_calls if c != "echo ok"]) == 1


class TestBuildInstallCmdVersionPin:
    """The pin must reach the installer, which reads args/VERSION in the bash
    side of the ``curl | bash`` pipe — an env prefix on curl never gets there."""

    def test_pinned_version_is_passed_as_installer_argument(self):
        cmd = build_install_cmd(
            OpenCodeConfig(base_url="http://proxy:8000/v1", opencode_version="1.0.180")
        )
        assert "| bash -s -- --version 1.0.180" in cmd
        assert "OPENCODE_VERSION" not in cmd

    def test_latest_omits_version_argument(self):
        cmd = build_install_cmd(OpenCodeConfig(base_url="http://proxy:8000/v1"))
        assert "| bash &&" in cmd
        assert "--version 1" not in cmd
