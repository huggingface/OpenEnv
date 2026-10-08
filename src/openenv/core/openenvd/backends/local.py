# SPDX-License-Identifier: BSD-3-Clause
"""The local backend: host subprocesses, no isolation, honestly labelled."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import stat
import sys
import uuid
from pathlib import Path

from ..isolation import IsolationError
from .base import EnforcementBackend

_TERMINATION_TIMEOUT = 5


class LocalBackend(EnforcementBackend):
    """
    Runs each sandbox as a temporary directory and host subprocesses.

    Useful for developing an environment and its graders without a gateway. It
    provides no [`~openenv.core.openenvd.policy.Guarantee`]s: the workload runs as
    the daemon's user, can read the daemon's files, and can reach its surfaces.
    An environment that requires any guarantee refuses to start under it.

    Args:
        config ([`~openenv.core.openenvd.policy.OpenEnvDConfig`]):
            The environment's validated `openenvd:` block.
    """

    name = "local"
    guarantees = frozenset()

    @property
    def python(self) -> str:
        return sys.executable

    async def probe(self) -> None:
        """Nothing to check: the local backend promises nothing."""

    def sandbox(self, timeout_s: float) -> "LocalSandbox":
        return LocalSandbox(timeout_s=timeout_s)

    def workload_env(self) -> dict[str, str]:
        env = super().workload_env()
        env["PATH"] = str(Path(self.python).parent) + os.pathsep + os.defpath
        return env


class LocalSandbox:
    """A private directory holding a copy of the seed; processes run on the host."""

    def __init__(self, *, timeout_s: float = 300):
        self.timeout_s = timeout_s
        self.id: str | None = None
        self._root: Path | None = None
        self._processes: list[asyncio.subprocess.Process] = []

    @property
    def workspace(self) -> Path:
        self._require_started()
        return self._root / "workspace"

    def _require_started(self) -> None:
        if self.id is None or self._root is None:
            raise IsolationError("local sandbox is not started")

    async def start(self, seed: Path, directory: Path) -> None:
        if self.id is not None:
            raise IsolationError("local sandbox has already been started")
        self.id = "local-" + uuid.uuid4().hex
        self._root = directory / self.id
        self._root.mkdir(mode=0o700)
        shutil.copytree(seed, self.workspace, symlinks=True)

    async def spawn(
        self, argv: list[str], env: dict[str, str]
    ) -> asyncio.subprocess.Process:
        self._require_started()
        local_env = dict(env)
        local_env["HOME"] = str(self._root)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.workspace,
            env=local_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=16 * 1024 * 1024,
        )
        self._processes.append(process)
        return process

    async def download(self, destination: Path) -> None:
        self._require_started()
        if (
            destination.is_symlink()
            or not destination.is_dir()
            or any(destination.iterdir())
        ):
            raise IsolationError("downloads require an empty staging directory")
        await asyncio.to_thread(
            shutil.copytree, self.workspace, destination, dirs_exist_ok=True
        )

    async def close(self) -> None:
        for process in self._processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), _TERMINATION_TIMEOUT)
            except asyncio.TimeoutError:
                raise IsolationError("local process termination was not confirmed")
        self._processes.clear()
        if self._root is not None:
            for root, directories, _ in os.walk(self._root, followlinks=False):
                for name in directories:
                    path = Path(root) / name
                    if not path.is_symlink():
                        path.chmod(stat.S_IMODE(path.stat().st_mode) | 0o700)
            shutil.rmtree(self._root)
            self._root = None
        self.id = None
