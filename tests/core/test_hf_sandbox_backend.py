# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the HF sandbox backend.

These tests mock ``hf-sandbox`` so they run without network or HF credentials.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
import types
from dataclasses import dataclass, field

import pytest


@dataclass
class _FakeSandbox:
    job_id: str
    files: dict[str, str] = field(default_factory=dict)
    marker_files: dict[str, str] = field(default_factory=dict)
    bg_jobs: dict[int, dict] = field(default_factory=dict)
    next_pid: int = 1000
    terminated: bool = False

    def exec(
        self,
        *cmd: str,
        workdir: str | None = None,
        stdin: str | None = None,
        timeout: int = 600,
    ) -> subprocess.CompletedProcess:
        del workdir, stdin, timeout
        if len(cmd) < 3:
            return subprocess.CompletedProcess(cmd, 1, "", "invalid command")
        script = cmd[2]

        if "ok_cmd" in script:
            return subprocess.CompletedProcess(cmd, 0, "ok\n", "")
        if "fail_cmd" in script:
            return subprocess.CompletedProcess(cmd, 42, "", "failed")
        if "timeout_cmd" in script:
            return subprocess.CompletedProcess(cmd, -1, "", "timeout")

        if "mkdir -p" in script:
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if "test -e " in script:
            match = re.search(r"test -e '([^']+)'", script)
            assert match is not None
            path = match.group(1)
            exists = path in self.files or path in self.marker_files
            return subprocess.CompletedProcess(cmd, 0 if exists else 1, "", "")

        if "cat '/tmp/.openenv_bg_" in script:
            match = re.search(r"cat '([^']+)'", script)
            assert match is not None
            marker = match.group(1)
            if marker in self.marker_files:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    f"{self.marker_files[marker]}\n",
                    "",
                )
            return subprocess.CompletedProcess(cmd, 1, "", "missing")

        if script.strip().startswith("kill -0 "):
            pid = int(script.strip().split()[2])
            alive = self.bg_jobs.get(pid, {}).get("alive", False)
            return subprocess.CompletedProcess(cmd, 0 if alive else 1, "", "")

        if script.strip().startswith("kill -9 "):
            pid = int(script.strip().split()[2])
            if pid in self.bg_jobs:
                self.bg_jobs[pid]["alive"] = False
                marker = self.bg_jobs[pid]["marker"]
                self.marker_files[marker] = "137"
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if "echo $!" in script:
            marker_match = re.search(r"(/tmp/\.openenv_bg_[A-Za-z0-9]+\.exit)", script)
            assert marker_match is not None
            marker = marker_match.group(1)
            pid = self.next_pid
            self.next_pid += 1
            long_running = "sleep 300" in script
            self.bg_jobs[pid] = {
                "marker": marker,
                "alive": long_running,
            }
            if not long_running:
                self.marker_files[marker] = "0"
            return subprocess.CompletedProcess(cmd, 0, f"{pid}\n", "")

        return subprocess.CompletedProcess(cmd, 0, "", "")

    def write_file(
        self,
        path: str,
        content: str | bytes | bytearray | memoryview,
    ) -> None:
        if isinstance(content, str):
            normalized = content
        else:
            normalized = bytes(content).decode("utf-8", "replace")
        self.files[path] = normalized

    def read_file(self, path: str, text: bool = True) -> str | bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path] if text else self.files[path].encode()

    def terminate(self) -> None:
        self.terminated = True


class _FakeSandboxAPI:
    calls: list[dict] = []

    @classmethod
    def create(
        cls,
        image: str,
        flavor: str,
        timeout: str,
        forward_hf_token: bool,
    ) -> _FakeSandbox:
        cls.calls.append(
            {
                "image": image,
                "flavor": flavor,
                "timeout": timeout,
                "forward_hf_token": forward_hf_token,
            }
        )
        return _FakeSandbox(job_id="job-123")


def _install_fake_hf_sandbox(monkeypatch) -> None:
    fake_module = types.ModuleType("hf_sandbox")
    setattr(fake_module, "Sandbox", _FakeSandboxAPI)
    monkeypatch.setitem(sys.modules, "hf_sandbox", fake_module)


@pytest.fixture(autouse=True)
def _reset_fake_hf_calls() -> None:
    _FakeSandboxAPI.calls.clear()


class TestHFSandboxBackend:
    def test_exported_from_package(self, monkeypatch):
        _install_fake_hf_sandbox(monkeypatch)

        import openenv.core.harness.sandbox as sandbox_pkg

        importlib.reload(sandbox_pkg)
        assert hasattr(sandbox_pkg, "HFSandboxBackend")
        assert hasattr(sandbox_pkg, "HFSandboxHandle")
        assert hasattr(sandbox_pkg, "HFBgJob")

    def test_create_exec_write_read_exists_bg_and_kill(self, monkeypatch):
        import openenv.core.harness.sandbox.hf_backend as hf_backend

        _install_fake_hf_sandbox(monkeypatch)
        importlib.reload(hf_backend)

        monkeypatch.setattr(hf_backend, "Sandbox", _FakeSandboxAPI)

        backend = hf_backend.HFSandboxBackend(
            image="python:3.12",
            flavor="cpu-basic",
            forward_hf_token=True,
        )
        sandbox = backend.create(timeout_s=120, envs={"GLOBAL_ENV": "on"})

        assert sandbox.sandbox_id == "job-123"
        assert _FakeSandboxAPI.calls[-1]["timeout"] == "2m"

        ok = sandbox.exec("ok_cmd")
        assert ok.exit_code == 0

        failed = sandbox.exec("fail_cmd")
        assert failed.exit_code == 42

        timed = sandbox.exec("timeout_cmd")
        assert timed.exit_code == -1

        sandbox.write_text("/tmp/hello.txt", "hello")
        assert sandbox.exists("/tmp/hello.txt")
        assert sandbox.read_text("/tmp/hello.txt") == "hello"

        short_job = sandbox.start_bg("echo done > /tmp/bg.txt")
        assert short_job.wait(timeout=2) == 0

        long_job = sandbox.start_bg("sleep 300")
        with pytest.raises(TimeoutError):
            long_job.wait(timeout=0.1)
        long_job.kill()
        assert isinstance(long_job.wait(timeout=2), int)

        sandbox.kill()
        raw = getattr(sandbox, "raw", None)
        assert raw is not None
        assert raw.terminated is True

    def test_factory_creates_hf_backend(self, monkeypatch):
        _install_fake_hf_sandbox(monkeypatch)

        import openenv.core.harness.sandbox as sandbox_pkg
        import openenv.core.harness.sandbox.hf_backend as hf_backend

        importlib.reload(hf_backend)
        importlib.reload(sandbox_pkg)

        monkeypatch.setattr(hf_backend, "Sandbox", _FakeSandboxAPI)
        backend = sandbox_pkg.create_sandbox_backend("hf", image="python:3.12")
        assert isinstance(backend, hf_backend.HFSandboxBackend)
