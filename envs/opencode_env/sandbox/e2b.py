# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""E2B implementation of :class:`SandboxBackend`."""

from __future__ import annotations

import os
import threading
from pathlib import PurePosixPath

from e2b import Sandbox
from e2b.sandbox_sync.commands.command_handle import CommandHandle

from .base import BgJob, ExecResult, SandboxBackend, SandboxHandle


class E2BBgJob:
    """Wraps an E2B ``CommandHandle`` to satisfy :class:`BgJob`.

    The E2B SDK's ``CommandHandle.wait()`` blocks indefinitely with no native
    timeout. We poll in a worker thread and raise ``TimeoutError`` if the
    process does not exit within the caller-supplied budget.
    """

    def __init__(self, handle: CommandHandle) -> None:
        self._handle = handle
        self._result: "object | None" = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._result = self._handle.wait()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc

    @property
    def pid(self) -> int:
        return self._handle.pid

    def wait(self, timeout: float | None = None) -> int:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError(
                f"Background command did not exit within {timeout}s"
            )
        if self._error is not None:
            # E2B raises CommandExitException on non-zero; treat as exit code.
            code = getattr(self._error, "exit_code", None)
            if code is None:
                raise self._error
            return int(code)
        return int(self._result.exit_code) if self._result is not None else 0

    def kill(self) -> None:
        try:
            self._handle.kill()
        except Exception:
            pass


class E2BSandboxHandle:
    """Wraps a live ``e2b.Sandbox`` to satisfy :class:`SandboxHandle`."""

    def __init__(self, sandbox: Sandbox) -> None:
        self._sbx = sandbox

    @property
    def sandbox_id(self) -> str:
        return self._sbx.sandbox_id

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
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            result = self._sbx.commands.run(
                cmd,
                envs=envs,
                cwd=cwd,
                timeout=timeout,
                background=False,
            )
            return ExecResult(
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except CommandExitException as exc:
            # Non-zero exit codes are expected in many contexts (e.g. polling
            # healthz before the server is up). Surface them as a proper
            # ExecResult instead of an exception.
            return ExecResult(
                exit_code=int(getattr(exc, "exit_code", 1)),
                stdout=str(getattr(exc, "stdout", "") or ""),
                stderr=str(getattr(exc, "stderr", "") or str(exc)),
            )

    def start_bg(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 0,
    ) -> BgJob:
        """Start a background command.

        ``timeout=0`` disables E2B's server-side command deadline (the default
        is 60s, which would otherwise kill long-running agent processes).
        Sandbox lifetime still bounds the job.
        """
        handle = self._sbx.commands.run(
            cmd,
            envs=envs,
            cwd=cwd,
            background=True,
            timeout=timeout,
        )
        return E2BBgJob(handle)

    def write_text(self, path: str, content: str) -> None:
        parent = str(PurePosixPath(path).parent)
        if parent not in ("", "/"):
            self._sbx.files.make_dir(parent)
        self._sbx.files.write(path, content)

    def read_text(self, path: str) -> str:
        return self._sbx.files.read(path)

    def exists(self, path: str) -> bool:
        return self._sbx.files.exists(path)

    def kill(self) -> None:
        self._sbx.kill()


class E2BSandboxBackend:
    """Creates E2B sandboxes for OpenCode rollouts.

    The backend uses the E2B default base template unless ``template`` is
    provided. Resource sizing and other E2B-specific options can be forwarded
    via ``sandbox_kwargs``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        template: str | None = None,
        sandbox_kwargs: dict | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("E2B_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "E2BSandboxBackend requires an api_key or E2B_API_KEY env var."
            )
        self._template = template
        self._sandbox_kwargs = sandbox_kwargs or {}

    def create(
        self,
        *,
        timeout_s: int = 900,
        envs: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> SandboxHandle:
        sbx = Sandbox.create(
            template=self._template,
            timeout=timeout_s,
            envs=envs,
            metadata=metadata,
            api_key=self._api_key,
            **self._sandbox_kwargs,
        )
        return E2BSandboxHandle(sbx)
