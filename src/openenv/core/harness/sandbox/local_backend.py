# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local subprocess implementation of :class:`SandboxBackend`.

Each sandbox gets an isolated temp root on the host filesystem plus an
optional per-sandbox virtualenv. Commands execute directly on the trainer
host, which makes this backend suitable for rootless cluster environments
where Docker/HF Jobs are unavailable.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from openenv.core.harness.sandbox.base import BgJob, ExecResult, SandboxHandle

_CANONICAL_HOME = "/home/user"
_CANONICAL_WORKDIR = "/testbed"


class LocalBgJob:
    """Handle to a background subprocess launched in a local sandbox."""

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self._proc = proc

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    def wait(self, timeout: float | None = None) -> int:
        try:
            return int(self._proc.wait(timeout=timeout))
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Background command (pid={self._proc.pid}) did not exit within {timeout}s"
            ) from exc

    def kill(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            os.killpg(self._proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            self._proc.terminate()
            return

        deadline = time.monotonic() + 5.0
        while self._proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except Exception:
                self._proc.kill()


class LocalSandboxHandle:
    """Host-backed sandbox handle with an isolated temp root."""

    supports_images = False

    def __init__(
        self,
        *,
        root_dir: str,
        home_dir: str,
        workdir: str,
        tmp_dir: str,
        default_envs: dict[str, str] | None = None,
        preserve_root: bool = False,
    ) -> None:
        self._root_dir = root_dir
        self._home_dir = home_dir
        self._workdir = workdir
        self._tmp_dir = tmp_dir
        self._default_envs = dict(default_envs or {})
        self._preserve_root = preserve_root
        self._bg_jobs: list[LocalBgJob] = []
        self._sandbox_id = f"local-{uuid.uuid4().hex[:12]}"

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    @property
    def sandbox_home(self) -> str:
        return self._home_dir

    @property
    def workdir(self) -> str:
        return self._workdir

    @property
    def tmp_dir(self) -> str:
        return self._tmp_dir

    def exec(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> ExecResult:
        run_env = self._build_env(envs)
        resolved_cwd = self._resolve_cwd(cwd)
        try:
            result = subprocess.run(
                ["bash", "-lc", cmd],
                cwd=resolved_cwd,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(
                exit_code=int(result.returncode),
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
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
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=self._resolve_cwd(cwd),
            env=self._build_env(envs),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            preexec_fn=os.setsid,
        )
        job = LocalBgJob(proc)
        self._bg_jobs.append(job)
        return job

    def write_text(self, path: str, content: str) -> None:
        resolved = Path(self._resolve_path(path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)

    def read_text(self, path: str) -> str:
        return Path(self._resolve_path(path)).read_text()

    def exists(self, path: str) -> bool:
        return Path(self._resolve_path(path)).exists()

    def kill(self) -> None:
        for job in self._bg_jobs:
            try:
                job.kill()
            except Exception:
                pass
        self._bg_jobs.clear()
        if not self._preserve_root:
            shutil.rmtree(self._root_dir, ignore_errors=True)

    def _resolve_cwd(self, cwd: str | None) -> str:
        candidate = self._resolve_path(cwd) if cwd else self._workdir
        Path(candidate).mkdir(parents=True, exist_ok=True)
        return candidate

    def _resolve_path(self, path: str | None) -> str:
        if not path:
            return self._workdir
        if path == _CANONICAL_HOME or path.startswith(f"{_CANONICAL_HOME}/"):
            suffix = path[len(_CANONICAL_HOME) :].lstrip("/")
            return str(Path(self._home_dir) / suffix) if suffix else self._home_dir
        if path == _CANONICAL_WORKDIR or path.startswith(f"{_CANONICAL_WORKDIR}/"):
            suffix = path[len(_CANONICAL_WORKDIR) :].lstrip("/")
            return str(Path(self._workdir) / suffix) if suffix else self._workdir
        return path

    def _build_env(self, envs: dict[str, str] | None) -> dict[str, str]:
        merged = os.environ.copy()
        merged.update(self._default_envs)
        merged.update(envs or {})
        merged.setdefault("HOME", self._home_dir)
        merged.setdefault("TMPDIR", self._tmp_dir)
        merged.setdefault("PIP_CACHE_DIR", str(Path(self._home_dir) / ".cache" / "pip"))
        merged.setdefault("XDG_CACHE_HOME", str(Path(self._home_dir) / ".cache"))
        merged.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        return merged


class LocalSandboxBackend:
    """Create host-local sandboxes rooted in unique temp directories."""

    supports_images = False

    def __init__(
        self,
        *,
        root_dir: str | None = None,
        create_virtualenv: bool = True,
        python_executable: str | None = None,
        preserve_root: bool = False,
    ) -> None:
        self._root_dir = root_dir
        self._create_virtualenv = create_virtualenv
        self._python_executable = python_executable or sys.executable
        self._preserve_root = preserve_root

    def create(
        self,
        *,
        timeout_s: int = 900,
        envs: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        image: str | None = None,
    ) -> SandboxHandle:
        del timeout_s, metadata, image

        if self._root_dir:
            Path(self._root_dir).mkdir(parents=True, exist_ok=True)
        root = tempfile.mkdtemp(prefix="openenv_local_", dir=self._root_dir)
        home = str(Path(root) / "home")
        workdir = str(Path(root) / "testbed")
        tmp_dir = str(Path(root) / "tmp")
        Path(home).mkdir(parents=True, exist_ok=True)
        Path(workdir).mkdir(parents=True, exist_ok=True)
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        default_envs = dict(envs or {})
        default_envs.setdefault("HOME", home)
        default_envs.setdefault("TMPDIR", tmp_dir)
        default_envs.setdefault(
            "PIP_CACHE_DIR",
            os.environ.get("PIP_CACHE_DIR", str(Path(home) / ".cache" / "pip")),
        )
        default_envs.setdefault(
            "XDG_CACHE_HOME",
            os.environ.get("XDG_CACHE_HOME", str(Path(home) / ".cache")),
        )
        default_envs.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

        venv_dir = Path(home) / "venv"
        if self._create_virtualenv:
            subprocess.run(
                [self._python_executable, "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            path_prefix = str(venv_dir / "bin")
            default_envs["VIRTUAL_ENV"] = str(venv_dir)
            default_envs["PATH"] = (
                f"{path_prefix}:{os.environ.get('PATH', '')}".rstrip(":")
            )

        return LocalSandboxHandle(
            root_dir=root,
            home_dir=home,
            workdir=workdir,
            tmp_dir=tmp_dir,
            default_envs=default_envs,
            preserve_root=self._preserve_root,
        )


__all__ = [
    "LocalBgJob",
    "LocalSandboxBackend",
    "LocalSandboxHandle",
]
