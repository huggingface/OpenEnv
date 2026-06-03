"""Providers for launching ASGI applications via ``uv run``."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Dict, Optional

import requests

from .providers import RuntimeProvider


def _check_uv_installed() -> None:
    try:
        subprocess.check_output(["uv", "--version"])
    except FileNotFoundError as exc:
        raise RuntimeError(
            "`uv` executable not found. Install uv from https://docs.astral.sh and ensure it is on PATH."
        ) from exc


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _create_uv_command(
    *,
    host: str,
    port: int,
    reload: bool,
    workers: int,
    app: str,
    project_path: str,
) -> list[str]:
    command: list[str] = ["uv", "run", "--isolated", "--project", project_path]

    command.append("--")
    command.extend(
        [
            "uvicorn",
            app,
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            str(workers),
        ]
    )

    if reload:
        command.append("--reload")

    return command


def _poll_health(health_url: str, timeout_s: float) -> None:
    """Poll a health endpoint until it returns HTTP 200 or times out."""

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            timeout = max(0.0001, min(deadline - time.time(), 2.0))
            response = requests.get(health_url, timeout=timeout)
            if response.status_code == 200:
                return
        except requests.RequestException:
            continue

        time.sleep(0.5)

    raise TimeoutError(f"Server did not become ready within {timeout_s:.1f} seconds")


class UVProvider(RuntimeProvider):
    """
    RuntimeProvider implementation backed by ``uv run``.

    Args:
        project_path: Local path to a uv project (passed to ``uv run --project``)
        app: ASGI application path for uvicorn (defaults to ``server.app:app``)
        host: Host interface to bind to (defaults to ``0.0.0.0``)
        reload: Whether to enable uvicorn's reload mode
        env_vars: Environment variables to pass through to the spawned process
        context_timeout_s: How long to wait for the environment to become ready

    Examples:

        ```python
        provider = UVProvider(project_path="/path/to/env")
        base_url = provider.start()
        print(base_url)  # http://localhost:8000
        # Use the environment via base_url
        provider.stop()
        ```
    """

    def __init__(
        self,
        *,
        project_path: str,
        app: str = "server.app:app",
        host: str = "0.0.0.0",
        reload: bool = False,
        env_vars: Optional[Dict[str, str]] = None,
        context_timeout_s: float = 60.0,
    ):
        """Initialize the UVProvider."""
        self.project_path = os.path.abspath(project_path)
        self.app = app
        self.host = host
        self.reload = reload
        self.env_vars = env_vars
        self.context_timeout_s = context_timeout_s
        _check_uv_installed()
        self._process = None
        self._base_url = None

    def start(
        self,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        workers: int = 1,
        **_: Dict[str, str],
    ) -> str:
        """
        Start the environment via `uv run`.

        Args:
            port: The port to bind the environment to
            env_vars: Environment variables to pass to the environment
            workers: The number of workers to use

        Returns:
            The base URL of the environment

        Raises:
            RuntimeError: If the environment is already running
        """
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("UVProvider is already running")

        bind_port = port or _find_free_port()

        command = _create_uv_command(
            host=self.host,
            port=bind_port,
            reload=self.reload,
            workers=workers,
            app=self.app,
            project_path=self.project_path,
        )

        env = os.environ.copy()

        if self.env_vars:
            env.update(self.env_vars)
        if env_vars:
            env.update(env_vars)

        try:
            self._process = subprocess.Popen(command, env=env)
        except OSError as exc:
            raise RuntimeError(f"Failed to launch `uv run`: {exc}") from exc

        client_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        self._base_url = f"http://{client_host}:{bind_port}"
        return self._base_url

    def wait_for_ready(self, timeout_s: float = 60.0) -> None:
        """
        Wait for the environment to become ready.

        Args:
            timeout_s: The timeout to wait for the environment to become ready

        Raises:
            RuntimeError: If the environment is not running
            TimeoutError: If the environment does not become ready within the timeout
        """
        if self._process and self._process.poll() is not None:
            code = self._process.returncode
            raise RuntimeError(f"uv process exited prematurely with code {code}")

        _poll_health(f"{self._base_url}/health", timeout_s=timeout_s)

    def stop(self) -> None:
        """
        Stop the environment.

        Raises:
            RuntimeError: If the environment is not running
        """
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)

        self._process = None
        self._base_url = None

    @property
    def base_url(self) -> str:
        """
        The base URL of the environment.

        Returns:
            The base URL of the environment

        Raises:
            RuntimeError: If the environment is not running
        """
        if self._base_url is None:
            raise RuntimeError("UVProvider has not been started")
        return self._base_url
