# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face implementation of :class:`SandboxBackend` (``huggingface_hub.Sandbox``, >=1.22)."""

from __future__ import annotations

import time
from pathlib import PurePosixPath

import httpx
from huggingface_hub import Sandbox

from .base import BgJob, ExecResult, SandboxHandle


_WAIT_POLL_INTERVAL_S = 0.5
# The sandbox API occasionally drops a single poll; give up only after this many
# consecutive transient failures.
_MAX_TRANSIENT_POLL_ERRORS = 10


class HFBgJob:
    """Satisfies :class:`BgJob`. The SDK has no blocking wait(), so poll ``processes()``."""

    def __init__(self, sandbox: Sandbox, process) -> None:
        self._sandbox = sandbox
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        transient_errors = 0
        while True:
            try:
                procs = self._sandbox.processes()
                transient_errors = 0
            except httpx.TransportError:
                # Transient sandbox-API disconnect (RemoteProtocolError, read
                # timeout, ...): retry on the next poll tick, still honoring the deadline.
                transient_errors += 1
                if transient_errors > _MAX_TRANSIENT_POLL_ERRORS:
                    raise
                procs = None
            if procs is not None:
                proc = next((p for p in procs if p.pid == self.pid), None)
                # Finished processes stay listed (running=False); a vanished pid means
                # the sandbox was torn down mid-run, not a clean exit.
                if proc is None:
                    raise RuntimeError(
                        f"process {self.pid} vanished (sandbox torn down?)"
                    )
                if not proc.running:
                    return int(proc.exit_code) if proc.exit_code is not None else 0
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Background command did not exit within {timeout}s"
                    )
                time.sleep(min(_WAIT_POLL_INTERVAL_S, remaining))
            else:
                time.sleep(_WAIT_POLL_INTERVAL_S)

    def kill(self) -> None:
        try:
            self._process.kill()  # idempotent server-side
        except Exception:
            pass


class HFSandboxHandle:
    """Wraps a live ``huggingface_hub.Sandbox`` to satisfy :class:`SandboxHandle`."""

    def __init__(self, sandbox: Sandbox) -> None:
        self._sbx = sandbox

    @property
    def sandbox_id(self) -> str:
        return self._sbx.id

    @property
    def raw(self) -> Sandbox:
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
        # check=False: surface non-zero exits as a result instead of raising
        result = self._sbx.run(
            ["bash", "-lc", cmd],
            env=envs,
            cwd=cwd,
            timeout=timeout,
            check=False,
        )
        # A timed-out command is killed with exit_code unset; surface it as a
        # non-zero exit instead of a false success. The SDK may return None for
        # stdout/stderr, so coerce to str (ExecResult / downstream expect str).
        if result.timed_out:
            return ExecResult(
                exit_code=124,
                stdout=result.stdout or "",
                stderr=(result.stderr or "") + f"\n[timed out after {timeout}s]",
            )
        return ExecResult(
            exit_code=int(result.exit_code) if result.exit_code is not None else 0,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    def start_bg(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> BgJob:
        process = self._sbx.run(
            ["bash", "-lc", cmd],
            env=envs,
            cwd=cwd,
            background=True,
        )
        return HFBgJob(self._sbx, process)

    def write_text(self, path: str, content: str) -> None:
        parent = str(PurePosixPath(path).parent)
        if parent not in (
            "",
            "/",
            ".",
        ):  # "." == relative path with no parent dir to create
            self._sbx.files.mkdir(parent)
        self._sbx.files.write(path, content)

    def read_text(self, path: str) -> str:
        return self._sbx.files.read_text(path)

    def exists(self, path: str) -> bool:
        return self._sbx.files.exists(path)

    def kill(self) -> None:
        # kill() tears the sandbox down; close() would only drop the client (it keeps idling)
        self._sbx.kill()


class HFSandboxBackend:
    """Creates Hugging Face sandboxes for OpenCode rollouts.

    ``image`` may be a plain base (e.g. ``python:3.12``), in which case opencode
    and the proxy deps are cold-installed per rollout, or a pre-baked image that
    already ships them.
    """

    def __init__(
        self,
        *,
        image: str,
        flavor: str = "cpu-basic",
        forward_hf_token: bool = False,
        sandbox_kwargs: dict | None = None,
    ) -> None:
        self._image = image
        self._flavor = flavor
        self._forward_hf_token = forward_hf_token
        self._sandbox_kwargs = sandbox_kwargs or {}

    def create(
        self,
        *,
        timeout_s: int = 900,
        envs: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> SandboxHandle:
        # metadata: accepted for protocol parity; Sandbox.create has no such param
        sbx = Sandbox.create(
            image=self._image,
            flavor=self._flavor,
            idle_timeout=timeout_s,
            env=envs,
            forward_hf_token=self._forward_hf_token,
            **self._sandbox_kwargs,
        )
        return HFSandboxHandle(sbx)
