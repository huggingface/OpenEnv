# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TB2 environment server implementation (Spaces-compatible local mode)."""

from __future__ import annotations

import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from openenv.core.env_server.interfaces import Environment


# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from tbench2_env.models import Tbench2Action, Tbench2Observation, Tbench2State
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from models import Tbench2Action, Tbench2Observation, Tbench2State

_CAMEL_IMPORT_ERROR: Exception | None = None


def _require_terminal_toolkit() -> Any:
    global _CAMEL_IMPORT_ERROR
    if _CAMEL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "camel-ai (TerminalToolkit) is required for TB2. Install from PyPI or from the CAMEL repo."
        ) from _CAMEL_IMPORT_ERROR

    try:
        from camel.toolkits import TerminalToolkit
    except Exception as exc:  # pragma: no cover
        _CAMEL_IMPORT_ERROR = exc
        raise RuntimeError(
            "camel-ai (TerminalToolkit) is required for TB2. Install from PyPI or from the CAMEL repo."
        ) from exc

    return TerminalToolkit


def _download_tb2_repo(cache_dir: Path) -> Path:
    repo_url = os.getenv(
        "TB2_REPO_URL",
        "https://github.com/laude-institute/terminal-bench-2/archive/refs/heads/main.zip",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "terminal-bench-2.zip"

    if not archive_path.exists():
        urllib.request.urlretrieve(repo_url, archive_path)

    with zipfile.ZipFile(archive_path) as zf:
        root = zf.namelist()[0].split("/")[0]
        extract_dir = cache_dir / root
        if not extract_dir.exists():
            zf.extractall(cache_dir)

    return extract_dir


def _read_instruction(task_dir: Path) -> str:
    instruction_path = task_dir / "instruction.md"
    if instruction_path.exists():
        return instruction_path.read_text(encoding="utf-8")
    return ""


def _read_timeout(task_dir: Path, fallback: float) -> float:
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return fallback
    try:
        data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    verifier = data.get("verifier", {})
    return float(verifier.get("timeout_sec", fallback))


class Tbench2Environment(Environment[Tbench2Action, Tbench2Observation, Tbench2State]):
    """OpenEnv wrapper around Terminal-Bench 2 tasks (local execution)."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        tasks_dir: str | None = None,
        output_dir: str | None = None,
        command_timeout_s: float = 20.0,
        safe_mode: bool = False,
        default_task_id: str | None = None,
    ) -> None:
        super().__init__()
        self.tasks_dir = tasks_dir or os.getenv("TB2_TASKS_DIR", "")
        self.output_dir = Path(
            output_dir or os.getenv("TB2_OUTPUT_DIR", "/tmp/tbench2_env_runs")
        )
        self.command_timeout_s = command_timeout_s
        self.safe_mode = safe_mode
        self.default_task_id = default_task_id or os.getenv(
            "TB2_DEFAULT_TASK_ID", "headless-terminal"
        )

        self._state = Tbench2State()
        self._task_dir: Path | None = None
        self._terminal_toolkit = None
        self._instruction = ""

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Tbench2Observation:
        del seed

        TerminalToolkit = _require_terminal_toolkit()

        task_id = (
            kwargs.get("task_id") or kwargs.get("task_name") or self.default_task_id
        )
        task_path = kwargs.get("task_path") or kwargs.get("path")

        task_dir = self._resolve_task_path(task_id, task_path)
        resolved_task_id = task_id or task_dir.name

        self._instruction = _read_instruction(task_dir)
        self._task_dir = task_dir

        trial_name = f"{resolved_task_id}.{episode_id or uuid4().hex}"
        session_logs_dir = (
            self.output_dir / trial_name / "terminal_toolkit_session_logs"
        )
        session_logs_dir.mkdir(parents=True, exist_ok=True)

        self._terminal_toolkit = TerminalToolkit(
            timeout=self.command_timeout_s,
            working_directory=str(task_dir),
            use_docker_backend=False,
            session_logs_dir=session_logs_dir,
            safe_mode=self.safe_mode,
            install_dependencies=["pytest"],
        )

        self._state = Tbench2State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=resolved_task_id,
            task_path=str(task_dir),
            terminal_ready=True,
        )

        return Tbench2Observation(
            instruction=self._instruction,
            output="",
            success=True,
            error="",
            task_id=resolved_task_id,
            task_path=str(task_dir),
            session_id=None,
            action_type="reset",
            info={},
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: Tbench2Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Tbench2Observation:
        del timeout_s, kwargs

        if not isinstance(action, Tbench2Action):
            raise TypeError(f"Expected Tbench2Action, got {type(action)}")

        if self._terminal_toolkit is None or self._task_dir is None:
            raise RuntimeError("TB2 environment not initialized. Call reset() first.")

        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        self._state.last_command = action.command

        output = ""
        error = ""
        success = True
        reward = None
        done = False
        info: dict[str, Any] = {}
        session_id = action.session_id or "tb2-session"

        try:
            if action.action_type == "exec":
                output = self._terminal_toolkit.shell_exec(
                    command=action.command,
                    block=action.block,
                    id=session_id,
                )
            elif action.action_type == "write":
                self._ensure_session_id(action.session_id, action.action_type)
                output = self._terminal_toolkit.shell_write_to_process(
                    id=action.session_id,
                    command=action.command,
                )
            elif action.action_type == "view":
                self._ensure_session_id(action.session_id, action.action_type)
                output = self._terminal_toolkit.shell_view(id=action.session_id)
            elif action.action_type == "wait":
                self._ensure_session_id(action.session_id, action.action_type)
                wait_seconds = action.wait_seconds or 0.0
                output = self._terminal_toolkit.shell_wait(
                    id=action.session_id,
                    wait_seconds=wait_seconds,
                )
            elif action.action_type == "kill":
                self._ensure_session_id(action.session_id, action.action_type)
                self._terminal_toolkit.shell_kill_process(id=action.session_id)
                output = f"Killed session {action.session_id}"
            elif action.action_type == "write_file":
                self._terminal_toolkit.shell_write_content_to_file(
                    content=action.content,
                    file_path=action.file_path,
                )
                output = f"Wrote content to {action.file_path}"
            elif action.action_type == "evaluate":
                output, reward, info = self._evaluate_task()
                done = True
            elif action.action_type == "close":
                self.close()
                output = "Closed TB2 environment."
                done = True
            else:
                raise ValueError(f"Unsupported action_type: {action.action_type}")
        except Exception as exc:  # pragma: no cover
            success = False
            error = str(exc)

        self._state.last_output = output
        self._state.session_id = session_id or ""

        return Tbench2Observation(
            instruction=self._instruction,
            output=output,
            success=success,
            error=error,
            task_id=self._state.task_id,
            task_path=self._state.task_path,
            session_id=session_id or "",
            action_type=action.action_type,
            info=info,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> Tbench2State:
        return self._state

    def close(self) -> None:
        self._terminal_toolkit = None
        self._task_dir = None
        self._instruction = ""

    def _resolve_task_path(self, task_id: str | None, task_path: str | None) -> Path:
        if task_path:
            resolved = Path(task_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Task path not found: {resolved}")
            return resolved

        if not task_id:
            raise ValueError("Provide task_id or task_path to reset TB2 environment.")

        if not self.tasks_dir:
            cache_dir = Path(
                os.getenv("TB2_CACHE_DIR", str(self.output_dir / "repo_cache"))
            )
            repo_dir = _download_tb2_repo(cache_dir)
            resolved = repo_dir / task_id
        else:
            resolved = Path(self.tasks_dir).expanduser().resolve() / task_id

        if not resolved.exists():
            raise FileNotFoundError(f"Task path not found: {resolved}")
        return resolved

    def _ensure_session_id(self, session_id: str | None, action_type: str) -> None:
        if not session_id:
            raise ValueError(f"session_id is required for action_type='{action_type}'")

    def _evaluate_task(self) -> tuple[str, float, dict[str, Any]]:
        if self._task_dir is None:
            raise RuntimeError("TB2 environment not initialized. Call reset() first.")
        if self._terminal_toolkit is None:
            raise RuntimeError("Terminal toolkit not initialized.")

        _read_timeout(self._task_dir, fallback=900.0)  # Validate timeout config
        tests_dir = self._task_dir / "tests"
        cmd = f"cd {self._task_dir} && python -m pytest -q {tests_dir} -rA; echo __TB2_EXIT_CODE__:$?"
        output = self._terminal_toolkit.shell_exec(
            id="tb2-tests",
            command=cmd,
            block=True,
        )

        exit_code = 1
        marker = "__TB2_EXIT_CODE__"
        for line in output.splitlines()[::-1]:
            if marker in line:
                try:
                    exit_code = int(line.split(":", 1)[1].strip())
                except Exception:
                    exit_code = 1
                break

        reward = 1.0 if exit_code == 0 else 0.0
        info = {"tests_passed": exit_code == 0, "exit_code": exit_code}
        return output, reward, info


class Tbench2DockerEnvironment(
    Environment[Tbench2Action, Tbench2Observation, Tbench2State]
):
    """OpenEnv wrapper around Terminal-Bench 2 tasks with Docker isolation.

    This environment runs each task in its own Docker container, reading
    the image specification from task.toml's [environment] section.

    Requires:
    - Docker socket mounted (/var/run/docker.sock)
    - Sufficient disk space for container images
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        tasks_dir: str | None = None,
        output_dir: str | None = None,
        command_timeout_s: float = 300.0,
        safe_mode: bool = True,
        default_task_id: str | None = None,
    ) -> None:
        super().__init__()
        self.tasks_dir = tasks_dir or os.getenv("TB2_TASKS_DIR", "")
        self.output_dir = Path(
            output_dir or os.getenv("TB2_OUTPUT_DIR", "/tmp/tbench2_env_runs")
        )
        self.command_timeout_s = command_timeout_s
        self.safe_mode = safe_mode
        self.default_task_id = default_task_id or os.getenv(
            "TB2_DEFAULT_TASK_ID", "headless-terminal"
        )

        self._state = Tbench2State()
        self._task_dir: Path | None = None
        self._docker_client = None
        self._container = None
        self._instruction = ""
        self._task_image = ""
        self._task_config: dict[str, Any] = {}

    def _get_docker_client(self) -> Any:
        """Lazy initialization of Docker client."""
        if self._docker_client is None:
            try:
                import docker

                self._docker_client = docker.from_env()
            except Exception as exc:
                raise RuntimeError(
                    f"Docker client not available. Ensure Docker socket is mounted. Error: {exc}"
                ) from exc
        return self._docker_client

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Tbench2Observation:
        del seed

        task_id = (
            kwargs.get("task_id") or kwargs.get("task_name") or self.default_task_id
        )
        task_path = kwargs.get("task_path") or kwargs.get("path")

        task_dir = self._resolve_task_path(task_id, task_path)
        resolved_task_id = task_id or task_dir.name

        # Read task configuration including Docker image
        task_toml_path = task_dir / "task.toml"
        if task_toml_path.exists():
            self._task_config = tomllib.loads(
                task_toml_path.read_text(encoding="utf-8")
            )
            self._task_image = self._task_config.get("environment", {}).get(
                "docker_image", ""
            )
        else:
            self._task_image = ""
            self._task_config = {}

        self._instruction = _read_instruction(task_dir)
        self._task_dir = task_dir

        # Create trial directory for logs
        trial_name = f"{resolved_task_id}.{episode_id or uuid4().hex}"
        trial_dir = self.output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Start Docker container if image is specified
        if self._task_image:
            self._start_container(task_dir, trial_dir)
        else:
            # Fallback to local mode if no image specified
            self._state = Tbench2State(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
                task_id=resolved_task_id,
                task_path=str(task_dir),
                terminal_ready=not self._task_image,  # Ready if no container needed
            )

        return Tbench2Observation(
            instruction=self._instruction,
            output="",
            success=True,
            error="",
            task_id=resolved_task_id,
            task_path=str(task_dir),
            session_id=None,
            action_type="reset",
            info={"docker_image": self._task_image} if self._task_image else {},
            reward=0.0,
            done=False,
        )

    def _start_container(self, task_dir: Path, trial_dir: Path) -> None:
        """Start a Docker container for the task.

        Uses file copying instead of bind mounts to support Docker-in-Docker
        scenarios where the server runs inside a container. Bind mounts reference
        host paths, which don't exist when the server is containerized.
        """
        docker = self._get_docker_client()

        try:
            # Pull image if needed
            try:
                docker.images.get(self._task_image)
            except Exception:
                logging.info(f"Pulling image {self._task_image}...")
                docker.images.pull(self._task_image)

            # Start container WITHOUT bind mounts (for DinD compatibility)
            self._container = docker.containers.run(
                image=self._task_image,
                command="sleep infinity",
                detach=True,
                network_mode="host",
                working_dir="/task",
                remove=False,
            )

            # Copy task files into container using tar archive
            # This works in Docker-in-Docker because we read files from our
            # filesystem and stream them to the container via the Docker API
            self._copy_dir_to_container(task_dir, "/task")

            self._state = Tbench2State(
                episode_id=str(uuid4()),
                step_count=0,
                task_id=task_dir.name,
                task_path=str(task_dir),
                terminal_ready=True,
            )

        except Exception as exc:
            raise RuntimeError(f"Failed to start container: {exc}") from exc

    def _copy_dir_to_container(self, src_dir: Path, dest_path: str) -> None:
        """Copy a directory into the container using tar archive.

        This method streams files via the Docker API, avoiding bind mount
        issues in Docker-in-Docker scenarios.
        """
        import io
        import tarfile

        if self._container is None:
            raise RuntimeError("Container not started")

        # Create tar archive in memory
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for item in src_dir.rglob("*"):
                arcname = str(item.relative_to(src_dir))
                tar.add(str(item), arcname=arcname)

        tar_stream.seek(0)

        # Copy to container
        self._container.put_archive(dest_path, tar_stream.getvalue())

    def _exec_in_container(
        self, command: str, workdir: str = "/task"
    ) -> tuple[int, str]:
        """Execute a command inside the container."""
        if self._container is None:
            raise RuntimeError("Container not started. Call reset() first.")

        exit_code, output = self._container.exec_run(
            cmd=f"bash -c 'cd {workdir} && {command}'",
            workdir="/task",
            stdout=True,
            stderr=True,
        )
        return exit_code, output.decode("utf-8", errors="replace")

    def step(
        self,
        action: Tbench2Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Tbench2Observation:
        del timeout_s, kwargs

        if not isinstance(action, Tbench2Action):
            raise TypeError(f"Expected Tbench2Action, got {type(action)}")

        if self._task_dir is None:
            raise RuntimeError("TB2 environment not initialized. Call reset() first.")

        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        self._state.last_command = action.command

        output = ""
        error = ""
        success = True
        reward = None
        done = False
        info: dict[str, Any] = {}
        session_id = action.session_id or "tb2-session"

        try:
            if action.action_type == "exec":
                if self._container:
                    exit_code, output = self._exec_in_container(action.command)
                    success = exit_code == 0
                else:
                    # Fallback to local execution
                    import subprocess

                    result = subprocess.run(
                        action.command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=self.command_timeout_s,
                    )
                    output = result.stdout + result.stderr
                    success = result.returncode == 0

            elif action.action_type == "write_file":
                if self._container:
                    # Write to container
                    exit_code, _ = self._exec_in_container(
                        f"cat > {action.file_path} << 'EOF'\n{action.content}\nEOF"
                    )
                    success = exit_code == 0
                    output = f"Wrote to {action.file_path}"
                else:
                    # Local write
                    Path(action.file_path).write_text(action.content)
                    output = f"Wrote to {action.file_path}"

            elif action.action_type == "evaluate":
                if self._container:
                    output, reward, info = self._evaluate_docker()
                else:
                    output, reward, info = self._evaluate_local()
                done = True

            elif action.action_type == "close":
                self.close()
                output = "Closed TB2 environment."
                done = True

            else:
                raise ValueError(
                    f"Unsupported action_type in Docker mode: {action.action_type}"
                )

        except Exception as exc:
            success = False
            error = str(exc)

        self._state.last_output = output
        self._state.session_id = session_id or ""

        return Tbench2Observation(
            instruction=self._instruction,
            output=output,
            success=success,
            error=error,
            task_id=self._state.task_id,
            task_path=self._state.task_path,
            session_id=session_id or "",
            action_type=action.action_type,
            info=info,
            reward=reward,
            done=done,
        )

    def _evaluate_docker(self) -> tuple[str, float, dict[str, Any]]:
        """Evaluate task inside Docker container."""
        if self._container is None:
            raise RuntimeError("Container not started.")
        assert self._task_dir is not None, "Task directory not set"

        # Run pytest in the container's /task directory
        # Use exit code marker for consistency with local mode
        cmd = "cd /task && python -m pytest -q tests/ -rA; echo __TB2_EXIT_CODE__:$?"

        exit_code, output = self._container.exec_run(
            cmd=f"bash -c '{cmd}'",
            workdir="/task",
            stdout=True,
            stderr=True,
        )
        output_str = output.decode("utf-8", errors="replace")

        # Parse exit code from marker (same logic as local mode)
        ec = 1
        marker = "__TB2_EXIT_CODE__"
        for line in output_str.splitlines()[::-1]:
            if marker in line:
                try:
                    ec = int(line.split(":", 1)[1].strip())
                except Exception:
                    ec = 1
                break

        reward = 1.0 if ec == 0 else 0.0
        info = {"tests_passed": ec == 0, "exit_code": ec}
        return output_str, reward, info

    def _evaluate_local(self) -> tuple[str, float, dict[str, Any]]:
        """Evaluate task locally (fallback)."""
        if self._task_dir is None:
            raise RuntimeError("Task not initialized.")

        tests_dir = self._task_dir / "tests"
        cmd = f"cd {self._task_dir} && python -m pytest -q {tests_dir} -rA; echo __TB2_EXIT_CODE__:$?"

        import subprocess

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=900.0,
        )
        output = result.stdout + result.stderr
        exit_code = result.returncode

        reward = 1.0 if exit_code == 0 else 0.0
        info = {"tests_passed": exit_code == 0, "exit_code": exit_code}
        return output, reward, info

    @property
    def state(self) -> Tbench2State:
        return self._state

    def close(self) -> None:
        if self._container:
            try:
                self._container.stop(timeout=10)
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None
        self._task_dir = None
        self._instruction = ""

    def _resolve_task_path(self, task_id: str | None, task_path: str | None) -> Path:
        if task_path:
            resolved = Path(task_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Task path not found: {resolved}")
            return resolved

        if not task_id:
            raise ValueError("Provide task_id or task_path to reset TB2 environment.")

        if not self.tasks_dir:
            cache_dir = Path(
                os.getenv("TB2_CACHE_DIR", str(self.output_dir / "repo_cache"))
            )
            repo_dir = _download_tb2_repo(cache_dir)
            resolved = repo_dir / task_id
        else:
            resolved = Path(self.tasks_dir).expanduser().resolve() / task_id

        if not resolved.exists():
            raise FileNotFoundError(f"Task path not found: {resolved}")
        return resolved
