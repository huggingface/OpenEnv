# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face Sandbox implementation of :class:`SandboxBackend`.

Wraps `hf-sandbox` (https://github.com/huggingface/hf-sandbox) so OpenEnv
harnesses can use it through the same protocol.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import PurePosixPath
from threading import Event
from typing import Any

from hf_sandbox import Sandbox
from openenv.core.harness.sandbox._util import shell_quote
from openenv.core.harness.sandbox.base import BgJob, ExecResult, SandboxHandle

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class HFSandboxError(RuntimeError):
    """Base class for HF sandbox backend errors."""


class HFSandboxCreateError(HFSandboxError):
    """Raised when backend cannot create a sandbox."""


class HFBgJob:
    """Background process handle for :class:`HFSandboxHandle`."""

    def __init__(
        self,
        sandbox: "HFSandboxHandle",
        *,
        pid: int,
        marker_path: str,
        poll_interval_s: float = 0.5,
    ) -> None:
        self._sandbox = sandbox
        self._pid = pid
        self._marker_path = marker_path
        self._poll_interval_s = poll_interval_s
        self._done = Event()
        self._exit_code: int | None = None

    @property
    def pid(self) -> int:
        return self._pid

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            if self._done.is_set():
                return self._exit_code if self._exit_code is not None else 0
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(
                    f"Background command (pid={self._pid}) "
                    f"did not exit within {timeout}s"
                )

            marker = self._sandbox.exec(
                f"cat {shell_quote(self._marker_path)}",
                timeout=10,
            )
            if marker.exit_code == 0 and marker.stdout.strip():
                self._exit_code = _parse_exit_code(marker.stdout.strip(), default=0)
                self._done.set()
                return self._exit_code

            alive = self._sandbox.exec(f"kill -0 {self._pid}", timeout=10)
            if alive.exit_code != 0:
                self._exit_code = 1
                self._done.set()
                return self._exit_code

            time.sleep(self._poll_interval_s)

    def kill(self) -> None:
        if self._done.is_set():
            return
        try:
            self._sandbox.exec(f"kill -9 {self._pid}", timeout=5)
        except Exception:
            pass
        self._exit_code = 137
        self._done.set()


class HFSandboxHandle:
    """Wraps a live ``hf_sandbox.Sandbox`` to satisfy :class:`SandboxHandle`."""

    def __init__(
        self,
        sandbox: Any,
        *,
        default_envs: dict[str, str] | None = None,
    ) -> None:
        self._sbx = sandbox
        self._default_envs = dict(default_envs or {})
        self._bg_jobs: list[HFBgJob] = []

    @property
    def sandbox_id(self) -> str:
        return str(getattr(self._sbx, "job_id", "hf-sandbox"))

    @property
    def raw(self) -> Any:
        """Escape hatch for callers that need the underlying SDK object."""
        return self._sbx

    def exec(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> ExecResult:
        merged_envs = dict(self._default_envs)
        merged_envs.update(envs or {})
        shell_cmd = _with_env_prefix(cmd, merged_envs)
        timeout_s = _normalize_exec_timeout(timeout)
        try:
            result = self._sbx.exec(
                "bash",
                "-lc",
                shell_cmd,
                workdir=cwd,
                timeout=timeout_s,
            )
            return ExecResult(
                exit_code=int(getattr(result, "returncode", 1)),
                stdout=str(getattr(result, "stdout", "") or ""),
                stderr=str(getattr(result, "stderr", "") or ""),
            )
        except Exception as exc:
            return ExecResult(exit_code=-1, stdout="", stderr=str(exc))

    def start_bg(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> BgJob:
        marker_path = f"/tmp/.openenv_bg_{uuid.uuid4().hex[:12]}.exit"
        wrapped = f"{cmd}; rc=$?; echo $rc > {shell_quote(marker_path)}"
        launch_cmd = f"nohup bash -lc {shell_quote(wrapped)} >/dev/null 2>&1 & echo $!"

        result = self.exec(launch_cmd, envs=envs, cwd=cwd, timeout=30)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to start background command: {result.stderr or result.stdout}"
            )

        pid = _parse_pid(result.stdout)
        if pid is None:
            raise RuntimeError(
                f"Could not extract PID from start_bg output: {result.stdout!r}"
            )

        job = HFBgJob(self, pid=pid, marker_path=marker_path)
        self._bg_jobs.append(job)
        return job

    def write_text(self, path: str, content: str) -> None:
        parent = str(PurePosixPath(path).parent)
        if parent not in ("", "/"):
            r = self.exec(f"mkdir -p {shell_quote(parent)}", timeout=10)
            if r.exit_code != 0:
                raise RuntimeError(
                    f"Failed to create parent directory {parent!r}: {r.stderr}"
                )
        self._sbx.write_file(path, content)

    def read_text(self, path: str) -> str:
        return str(self._sbx.read_file(path, text=True))

    def exists(self, path: str) -> bool:
        r = self.exec(f"test -e {shell_quote(path)}", timeout=10)
        return r.exit_code == 0

    def kill(self) -> None:
        for job in self._bg_jobs:
            try:
                job.kill()
            except Exception:
                pass
        self._bg_jobs.clear()
        try:
            self._sbx.terminate()
        except Exception:
            pass


class HFSandboxBackend:
    """Creates HF sandboxes for harness rollouts via ``hf-sandbox``."""

    def __init__(
        self,
        *,
        image: str = "python:3.12",
        flavor: str = "cpu-basic",
        timeout: str | None = None,
        forward_hf_token: bool = False,
        create_retries: int = 3,
        create_backoff_s: float = 2.0,
    ) -> None:
        self._image = image
        self._flavor = flavor
        self._timeout = timeout
        self._forward_hf_token = forward_hf_token
        self._create_retries = max(1, int(create_retries))
        self._create_backoff_s = max(0.0, float(create_backoff_s))

    def create(
        self,
        *,
        timeout_s: int = 900,
        envs: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        image: str | None = None,
    ) -> SandboxHandle:
        # `hf-sandbox` does not support metadata at create-time yet.
        del metadata

        timeout = self._timeout or _format_timeout(timeout_s)
        effective_image = image or self._image
        last_error: Exception | None = None

        for attempt in range(self._create_retries):
            try:
                sbx = Sandbox.create(
                    image=effective_image,
                    flavor=self._flavor,
                    timeout=timeout,
                    forward_hf_token=self._forward_hf_token,
                )
                return HFSandboxHandle(sbx, default_envs=envs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 < self._create_retries:
                    time.sleep(self._create_backoff_s * (2**attempt))

        assert last_error is not None
        raise HFSandboxCreateError(
            f"Failed to create HF sandbox after {self._create_retries} attempts "
            f"({type(last_error).__name__})."
        ) from last_error


def _with_env_prefix(cmd: str, envs: dict[str, str]) -> str:
    if not envs:
        return cmd
    parts: list[str] = []
    for key, value in envs.items():
        if not _ENV_KEY_RE.match(key):
            raise ValueError(f"Invalid environment variable name: {key!r}")
        parts.append(f"export {key}={shell_quote(str(value))};")
    return " ".join(parts) + f" {cmd}"


def _normalize_exec_timeout(timeout: float | None) -> int:
    if timeout is None:
        return 24 * 60 * 60
    return max(1, int(timeout))


def _format_timeout(timeout_s: int) -> str:
    timeout_s = max(1, int(timeout_s))
    if timeout_s % 3600 == 0:
        return f"{timeout_s // 3600}h"
    if timeout_s % 60 == 0:
        return f"{timeout_s // 60}m"
    return f"{timeout_s}s"


def _parse_pid(stdout: str) -> int | None:
    for line in reversed(stdout.strip().splitlines()):
        raw = line.strip()
        if raw.isdigit():
            return int(raw)
    return None


def _parse_exit_code(raw: str, *, default: int) -> int:
    try:
        return int(raw.splitlines()[-1].strip())
    except Exception:
        return default


__all__ = [
    "HFBgJob",
    "HFSandboxBackend",
    "HFSandboxCreateError",
    "HFSandboxError",
    "HFSandboxHandle",
]
