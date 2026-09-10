# SPDX-License-Identifier: BSD-3-Clause

"""Subprocess lifecycle tests for HarnessProcess (real processes)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from openenv.core.harness import (
    HarnessNotRunningError,
    HarnessProcess,
    HarnessStartupError,
)

# Subprocess behaviors live in a real, lintable module (see its docstring
# for the available modes) instead of inline code strings.
SCRIPTED_HARNESS = Path(__file__).parent / "scripted_harness.py"


def make_process(mode: str, **kwargs) -> HarnessProcess:
    defaults = {"startup_timeout_s": 10.0, "terminate_grace_s": 1.0}
    defaults.update(kwargs)
    return HarnessProcess(
        [sys.executable, "-u", str(SCRIPTED_HARNESS), mode], cwd=".", **defaults
    )


def ready(line: str) -> bool:
    return line == "ready"


class TestLifecycle:
    async def test_start_echo_read_stop(self):
        process = make_process("echo")
        assert process.is_running() is False
        await process.start(ready_check=ready)
        assert process.is_running() is True

        await process.write_line("hello")
        assert await process.read_line(timeout_s=10.0) == "echo:hello"

        await process.stop()
        assert process.is_running() is False

    async def test_stop_is_idempotent(self):
        process = make_process("echo")
        await process.start(ready_check=ready)
        await process.stop()
        await process.stop()
        assert process.is_running() is False

    async def test_stop_before_start_is_noop(self):
        process = make_process("echo")
        await process.stop()
        assert process.is_running() is False

    async def test_double_start_rejected(self):
        process = make_process("echo")
        await process.start(ready_check=ready)
        try:
            with pytest.raises(HarnessStartupError, match="already running"):
                await process.start(ready_check=ready)
        finally:
            await process.stop()


class TestStartupFailures:
    async def test_startup_timeout_kills_process(self):
        process = make_process("slow-start", startup_timeout_s=0.5)
        with pytest.raises(HarnessStartupError, match="did not become ready"):
            await process.start(ready_check=ready)
        assert process.is_running() is False

    async def test_immediate_exit_reports_code_and_stderr(self):
        process = make_process("exit-now")
        with pytest.raises(HarnessStartupError) as exc_info:
            await process.start(ready_check=ready)
        assert "exited with code 3" in str(exc_info.value)
        assert "dying" in str(exc_info.value)

    async def test_unspawnable_command(self):
        process = HarnessProcess(["/nonexistent/definitely-not-a-binary"], cwd=".")
        with pytest.raises(HarnessStartupError, match="failed to spawn"):
            await process.start()


class TestCrashDetection:
    async def test_crash_mid_session(self):
        process = make_process("crash-after-echo")
        await process.start(ready_check=ready)
        await process.write_line("one")
        assert await process.read_line(timeout_s=10.0) == "echo:one"

        # Process exits after the first echo; wait for it to die
        deadline = time.monotonic() + 10.0
        while process.is_running() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert process.is_running() is False

        with pytest.raises(HarnessNotRunningError):
            await process.write_line("two")
        await process.stop()

    async def test_read_line_returns_none_on_timeout(self):
        process = make_process("echo")
        await process.start(ready_check=ready)
        try:
            assert await process.read_line(timeout_s=0.1) is None
        finally:
            await process.stop()


class TestTerminateEscalation:
    async def test_sigterm_ignorer_is_killed_within_grace(self):
        process = make_process("ignore-sigterm", terminate_grace_s=0.5)
        await process.start(ready_check=ready)

        started = time.monotonic()
        await process.stop()
        elapsed = time.monotonic() - started

        assert process.is_running() is False
        assert elapsed < 5.0  # grace (0.5s) + margin, well under the 60s sleep


class TestEncoding:
    async def test_non_ascii_output_survives_the_reader(self):
        # text=True alone decodes with the locale encoding (often ASCII in a
        # container), and UnicodeDecodeError is a ValueError, which the reader
        # thread used to swallow -- killing stdout pumping for the whole turn.
        process = make_process("unicode-echo")
        await process.start(ready_check=ready)
        try:
            await process.write_line("hello")
            line = await process.read_line(timeout_s=10.0)
            assert line == "echo:✓ hello 世界 \U0001f600"

            # The pump is still alive for subsequent turns.
            await process.write_line("again")
            assert await process.read_line(timeout_s=10.0) is not None
        finally:
            await process.stop()
