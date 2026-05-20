# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Docker implementation of :class:`SandboxBackend`.

Runs each sandbox as a ``docker run -d`` container on the local machine.
Commands execute via ``docker exec``, files transfer via ``docker exec``
with stdin piping. Suitable for CI, local dev, and environments without
KVM or cloud sandbox credentials.

Usage::

    from openenv.core.harness.sandbox.docker_backend import DockerSandboxBackend

    backend = DockerSandboxBackend(image="ubuntu:22.04")
    sandbox = backend.create()
    result = sandbox.exec("echo hello")
    print(result.stdout)  # "hello"
    sandbox.kill()
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
import uuid
from pathlib import PurePosixPath

from openenv.core.harness.sandbox._util import shell_quote
from openenv.core.harness.sandbox.base import BgJob, ExecResult

_log = logging.getLogger(__name__)


class DockerBgJob:
    """Handle to a background process running inside a Docker container.

    Launches the command via ``docker exec -d`` and tracks the wrapper
    shell PID. Completion is detected by polling whether the PID is still
    alive inside the container.
    """

    def __init__(
        self,
        container_id: str,
        pid: int,
        poll_thread: threading.Thread | None = None,
    ) -> None:
        self._container_id = container_id
        self._pid = pid
        self._exit_code: int | None = None
        self._done = threading.Event()
        self._poll_thread = poll_thread

    @property
    def pid(self) -> int:
        return self._pid

    def wait(self, timeout: float | None = None) -> int:
        if not self._done.wait(timeout=timeout):
            raise TimeoutError(
                f"Background command (pid={self._pid}) did not exit within {timeout}s"
            )
        return self._exit_code if self._exit_code is not None else 0

    def kill(self) -> None:
        try:
            subprocess.run(
                ["docker", "exec", self._container_id, "kill", "-9", str(self._pid)],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
        self._done.set()


class DockerSandboxHandle:
    """Wraps a running Docker container to satisfy :class:`SandboxHandle`."""

    def __init__(self, container_id: str, *, user: str | None = None) -> None:
        self._container_id = container_id
        self._user = user
        self._bg_jobs: list[DockerBgJob] = []

    @property
    def sandbox_id(self) -> str:
        return self._container_id[:12]

    def exec(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> ExecResult:
        docker_cmd = self._build_exec_cmd(envs=envs, cwd=cwd)
        docker_cmd.extend(["bash", "-c", cmd])
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                exit_code=-1, stdout="", stderr=f"Command timed out after {timeout}s"
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
        marker = f"/tmp/.bg_{uuid.uuid4().hex}"
        wrapped = f"bash -c {shell_quote(cmd + f'; echo $? > {marker}')} &\necho $!"
        docker_cmd = self._build_exec_cmd(envs=envs, cwd=cwd)
        docker_cmd.extend(["bash", "-c", wrapped])
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start background command: {result.stderr}")
        # Extract PID from the last numeric-only line (commands may print
        # banners like "Database migration complete." before the PID).
        pid_line = None
        for line in reversed(result.stdout.strip().splitlines()):
            if line.strip().isdigit():
                pid_line = line.strip()
                break
        if pid_line is None:
            raise RuntimeError(
                f"Could not extract PID from start_bg output: {result.stdout!r}"
            )
        pid = int(pid_line)

        job = DockerBgJob(self._container_id, pid)
        poll_thread = threading.Thread(
            target=self._poll_bg_job,
            args=(job, marker),
            daemon=True,
        )
        job._poll_thread = poll_thread
        self._bg_jobs.append(job)
        poll_thread.start()
        return job

    def write_text(self, path: str, content: str) -> None:
        parent = str(PurePosixPath(path).parent)
        if parent not in ("", "/"):
            mkdir_result = subprocess.run(
                ["docker", "exec", self._container_id, "mkdir", "-p", parent],
                capture_output=True,
                timeout=10,
            )
            if mkdir_result.returncode != 0:
                raise RuntimeError(
                    f"Failed to create directory {parent!r} in container "
                    f"{self._container_id}: {mkdir_result.stderr.decode(errors='replace')}"
                )
        write_result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self._container_id,
                "bash",
                "-c",
                f"cat > {shell_quote(path)}",
            ],
            input=content.encode(),
            capture_output=True,
            timeout=30,
        )
        if write_result.returncode != 0:
            raise RuntimeError(
                f"Failed to write file {path!r} in container "
                f"{self._container_id}: {write_result.stderr.decode(errors='replace')}"
            )

    def read_text(self, path: str) -> str:
        result = subprocess.run(
            ["docker", "exec", self._container_id, "cat", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise FileNotFoundError(f"No such file in container: {path}")
        return result.stdout

    def exists(self, path: str) -> bool:
        result = subprocess.run(
            ["docker", "exec", self._container_id, "test", "-e", path],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0

    def kill(self) -> None:
        for job in self._bg_jobs:
            try:
                job.kill()
            except Exception:
                pass
        self._bg_jobs.clear()
        try:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
                timeout=15,
            )
        except Exception:
            pass

    def _build_exec_cmd(
        self,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> list[str]:
        cmd = ["docker", "exec"]
        if self._user:
            cmd.extend(["-u", self._user])
        if cwd:
            cmd.extend(["-w", cwd])
        for k, v in (envs or {}).items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.append(self._container_id)
        return cmd

    def _poll_bg_job(self, job: DockerBgJob, marker: str) -> None:
        consecutive_failures = 0
        while not job._done.is_set():
            try:
                result = subprocess.run(
                    ["docker", "exec", self._container_id, "cat", marker],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    job._exit_code = int(result.stdout.strip())
                    job._done.set()
                    return
                if "No such container" in (result.stderr or ""):
                    job._exit_code = 1
                    job._done.set()
                    return
            except Exception:
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            # Also check if PID is gone (crash without writing marker).
            try:
                check = subprocess.run(
                    ["docker", "exec", self._container_id, "kill", "-0", str(job._pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if check.returncode != 0:
                    job._exit_code = 1
                    job._done.set()
                    return
                if "No such container" in (check.stderr or ""):
                    job._exit_code = 1
                    job._done.set()
                    return
            except Exception:
                consecutive_failures += 1

            if consecutive_failures >= 10:
                job._exit_code = 1
                job._done.set()
                return

            time.sleep(0.5)


class DockerSandboxBackend:
    """Creates Docker container sandboxes.

    Each :meth:`create` call spawns a fresh ``docker run -d`` container
    that stays alive until :meth:`SandboxHandle.kill` is called or the
    container's ``timeout_s`` sleep expires.
    """

    def __init__(
        self,
        *,
        image: str = "ubuntu:22.04",
        docker_args: list[str] | None = None,
        user: str | None = None,
    ) -> None:
        self._image = image
        self._docker_args = list(docker_args or [])
        self._user = user

        # Linux Docker Engine does not auto-resolve host.docker.internal
        # unless we explicitly map it.
        if "host.docker.internal:host-gateway" not in self._docker_args:
            self._docker_args.extend(
                ["--add-host", "host.docker.internal:host-gateway"]
            )

        try:
            subprocess.run(
                ["docker", "version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise RuntimeError(
                "DockerSandboxBackend requires a running Docker daemon."
            ) from exc

    def create(
        self,
        *,
        timeout_s: int = 900,
        envs: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        image: str | None = None,
    ) -> DockerSandboxHandle:
        cmd = [
            "docker",
            "run",
            "-d",
            "--label",
            "openenv.sandbox=true",
        ]
        if metadata:
            for k, v in metadata.items():
                cmd.extend(["--label", f"openenv.{k}={v}"])
        for k, v in (envs or {}).items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.extend(self._docker_args)
        effective_image = image or self._image
        cmd.extend([effective_image, "sleep", str(timeout_s)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create Docker sandbox: {result.stderr.strip()}"
            )
        container_id = result.stdout.strip()
        _log.info(
            "Docker sandbox created: %s (image=%s)",
            container_id[:12],
            effective_image,
        )
        return DockerSandboxHandle(container_id, user=self._user)
