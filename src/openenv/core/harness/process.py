# SPDX-License-Identifier: BSD-3-Clause

"""Loop-agnostic subprocess helper for CLI harnesses (RFC 005)."""

from __future__ import annotations

import asyncio
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from typing import Callable, IO, Optional

from .adapter import HarnessNotRunningError, HarnessStartupError

_STDERR_TAIL_LINES = 50


class HarnessProcess:
    """
    Manage a long-lived CLI harness subprocess with line-based stdio.

    Built on `subprocess.Popen` with daemon reader threads rather than asyncio
    subprocess transports so the same instance works across event loops: the
    sync facades in [`~openenv.core.harness.environment.HarnessEnvironment`]
    spin a fresh loop per call, while the HTTP server keeps one long-lived
    loop. All blocking pipe I/O is offloaded via `asyncio.to_thread`.

    Args:
        command (`list[str]`):
            Command line to execute.
        cwd (`str`):
            Working directory for the process.
        env_vars (`dict[str, str]`, *optional*):
            Extra environment variables, merged over `os.environ`.
        startup_timeout_s (`float`, *optional*, defaults to `30.0`):
            Maximum time for the process to become ready in `start()`.
        terminate_grace_s (`float`, *optional*, defaults to `5.0`):
            Time to wait after SIGTERM before escalating to SIGKILL.

    Examples:

    ```python
    process = HarnessProcess(["openclaw", "run"], cwd="/workspace")
    await process.start(ready_check=lambda line: line.startswith("ready"))
    await process.write_line("hello")
    reply = await process.read_line(timeout_s=10.0)
    await process.stop()
    ```
    """

    def __init__(
        self,
        command: list[str],
        cwd: str,
        env_vars: Optional[dict[str, str]] = None,
        startup_timeout_s: float = 30.0,
        terminate_grace_s: float = 5.0,
    ):
        self.command = list(command)
        self.cwd = cwd
        self.env_vars = dict(env_vars or {})
        self.startup_timeout_s = startup_timeout_s
        self.terminate_grace_s = terminate_grace_s

        self._proc: Optional[subprocess.Popen] = None
        self._stdout_queue: queue.Queue = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._reader_threads: list[threading.Thread] = []

    def is_running(self) -> bool:
        """Whether the subprocess is alive."""
        return self._proc is not None and self._proc.poll() is None

    def drain_stderr(self) -> str:
        """Return the most recent stderr output for diagnostics."""
        return "".join(self._stderr_tail)

    async def start(self, ready_check: Optional[Callable[[str], bool]] = None) -> None:
        """
        Start the subprocess and optionally wait for a readiness line.

        Args:
            ready_check (`Callable[[str], bool]`, *optional*):
                Predicate applied to each stdout line. `start()` returns once
                a line matches. When `None`, `start()` returns as soon as the
                process is spawned.

        Raises:
            [`~openenv.core.harness.adapter.HarnessStartupError`]:
                If the process cannot be spawned, exits before becoming
                ready, or does not become ready within `startup_timeout_s`.
        """
        if self.is_running():
            raise HarnessStartupError("harness process is already running")

        try:
            self._proc = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env={**os.environ, **self.env_vars},
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # Explicit UTF-8: text=True alone decodes with the locale
                # encoding, which is often ASCII in a container, and harness
                # output is routinely non-ASCII. errors="replace" keeps a
                # stray byte from killing the reader thread mid-turn.
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            self._proc = None
            raise HarnessStartupError(
                f"failed to spawn harness process {self.command!r}: {exc}"
            ) from exc

        self._stdout_queue = queue.Queue()
        self._stderr_tail = deque(maxlen=_STDERR_TAIL_LINES)
        self._reader_threads = [
            self._spawn_reader(self._proc.stdout, self._on_stdout_line),
            self._spawn_reader(self._proc.stderr, self._stderr_tail.append),
        ]

        if ready_check is None:
            return

        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await self.stop()
                raise HarnessStartupError(
                    f"harness did not become ready within {self.startup_timeout_s}s; "
                    f"stderr tail:\n{self.drain_stderr()}"
                )
            line = await self.read_line(timeout_s=min(remaining, 0.2))
            if line is None:
                if not self.is_running():
                    exit_code = self._proc.poll()
                    await self.stop()
                    raise HarnessStartupError(
                        f"harness exited with code {exit_code} during startup; "
                        f"stderr tail:\n{self.drain_stderr()}"
                    )
                continue
            if ready_check(line):
                return

    async def stop(self) -> None:
        """
        Stop the subprocess, escalating SIGTERM to SIGKILL after the grace period.

        Idempotent: safe to call when the process was never started or has
        already exited.
        """
        proc = self._proc
        if proc is None:
            return
        await asyncio.to_thread(self._stop_blocking, proc)

    def _stop_blocking(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            self._signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=self.terminate_grace_s)
            except subprocess.TimeoutExpired:
                self._signal_group(proc, signal.SIGKILL)
                proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for thread in self._reader_threads:
            thread.join(timeout=2.0)
        self._reader_threads = []

    @staticmethod
    def _signal_group(proc: subprocess.Popen, sig: signal.Signals) -> None:
        """Signal the process group (start_new_session=True) with a fallback."""
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

    async def write_line(self, text: str) -> None:
        """
        Write one line to the subprocess's stdin.

        Args:
            text (`str`):
                Line content; a trailing newline is appended if missing.

        Raises:
            [`~openenv.core.harness.adapter.HarnessNotRunningError`]:
                If the process is not running or stdin is closed.
        """
        if not self.is_running() or self._proc.stdin is None:
            raise HarnessNotRunningError(
                "cannot write to harness process: it is not running"
            )
        if not text.endswith("\n"):
            text += "\n"

        def _write() -> None:
            try:
                self._proc.stdin.write(text)
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                raise HarnessNotRunningError(
                    f"harness process stdin is closed: {exc}"
                ) from exc

        await asyncio.to_thread(_write)

    async def read_line(self, timeout_s: Optional[float] = None) -> Optional[str]:
        """
        Read one stdout line from the subprocess.

        Args:
            timeout_s (`float`, *optional*):
                Maximum time to wait. `None` blocks until a line or EOF.

        Returns:
            `str` line without its trailing newline, or `None` on EOF (the
            process exited or closed stdout) or timeout.
        """

        def _read() -> Optional[str]:
            try:
                item = self._stdout_queue.get(timeout=timeout_s)
            except queue.Empty:
                return None
            return item

        return await asyncio.to_thread(_read)

    def _on_stdout_line(self, line: str) -> None:
        self._stdout_queue.put(line.rstrip("\n"))

    def _spawn_reader(
        self, stream: Optional[IO[str]], sink: Callable[[str], None]
    ) -> threading.Thread:
        def _pump() -> None:
            if stream is None:
                return
            try:
                for line in stream:
                    sink(line)
            except (ValueError, OSError):
                # Raised when the pipe is closed during shutdown. Decoding
                # cannot raise here: the stream is opened with
                # errors="replace".
                pass

        thread = threading.Thread(target=_pump, daemon=True)
        thread.start()
        return thread


__all__ = ["HarnessProcess"]
