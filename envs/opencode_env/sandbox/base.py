# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sandbox backend protocol.

A ``SandboxBackend`` produces ``SandboxHandle`` instances that the harness uses
to stage files, run the OpenCode install, launch the agent as a background
process, and later tear the sandbox down.

Backends can be implemented against any provider (E2B, Docker, Modal, Prime)
as long as they satisfy the Protocols defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ExecResult:
    """Result of a synchronous command inside a sandbox."""

    exit_code: int
    stdout: str
    stderr: str


@runtime_checkable
class BgJob(Protocol):
    """Handle to a background process running inside a sandbox."""

    @property
    def pid(self) -> int: ...

    def wait(self, timeout: float | None = None) -> int:
        """Block until the process exits, returning its exit code.

        Implementations must raise ``TimeoutError`` if ``timeout`` elapses
        before the process exits.
        """

    def kill(self) -> None:
        """Terminate the process."""


@runtime_checkable
class SandboxHandle(Protocol):
    """Opaque handle to one live sandbox."""

    @property
    def sandbox_id(self) -> str: ...

    def exec(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> ExecResult:
        """Run a shell command synchronously and return its result."""

    def start_bg(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> BgJob:
        """Launch a background process and return a handle."""

    def write_text(self, path: str, content: str) -> None:
        """Write text to ``path`` inside the sandbox (parent dirs auto-created)."""

    def read_text(self, path: str) -> str:
        """Read ``path`` as text from the sandbox."""

    def exists(self, path: str) -> bool:
        """Return whether ``path`` exists in the sandbox."""

    def kill(self) -> None:
        """Terminate the sandbox and release resources."""


@runtime_checkable
class SandboxBackend(Protocol):
    """Factory for fresh sandbox instances."""

    def create(
        self,
        *,
        timeout_s: int = 900,
        envs: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> SandboxHandle:
        """Create and return a new, ready-to-use sandbox."""
