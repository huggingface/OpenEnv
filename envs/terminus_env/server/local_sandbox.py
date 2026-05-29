# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small local sandbox backend for cluster smoke training."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ShellResult:
    """Normalized result from a command executed in a local sandbox."""

    stdout: str
    stderr: str
    error: str | None
    success: bool


class LocalSandbox:
    """Runs shell commands in a persistent bubblewrap-backed home directory."""

    def __init__(self, *, root: str | None = None, **_: object):
        if shutil.which("bwrap") is None:
            raise RuntimeError(
                "local sandbox backend requires `bwrap` on the sandbox node"
            )
        self._tmp = tempfile.TemporaryDirectory(prefix="terminus-sandbox-", dir=root)
        self._home = Path(self._tmp.name) / "home" / "user"
        self._tmp_dir = Path(self._tmp.name) / "tmp"
        self._home.mkdir(parents=True, exist_ok=True)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_id = Path(self._tmp.name).name

    def run_shell(self, command: str, timeout_s: int = 120) -> ShellResult:
        process = subprocess.run(
            self._bwrap_command(command),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        success = process.returncode == 0
        return ShellResult(
            stdout=process.stdout or "",
            stderr=process.stderr or "",
            error=None if success else f"exit code {process.returncode}",
            success=success,
        )

    def kill(self) -> None:
        self._tmp.cleanup()

    def _bwrap_command(self, command: str) -> list[str]:
        return [
            "bwrap",
            "--die-with-parent",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/run",
            "--bind",
            str(self._tmp_dir),
            "/tmp",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--ro-bind",
            "/etc",
            "/etc",
            "--dir",
            "/home",
            "--bind",
            str(self._home),
            "/home/user",
            "--chdir",
            "/home/user",
            "--setenv",
            "HOME",
            "/home/user",
            "--setenv",
            "PATH",
            os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "/bin/bash",
            "-lc",
            command,
        ]
