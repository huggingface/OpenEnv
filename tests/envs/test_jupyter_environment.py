# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import io
import subprocess
from types import SimpleNamespace

import pytest
from jupyter_env.models import JupyterState
from jupyter_env.server.e2b_sandbox import CellResult, E2BSandbox
from jupyter_env.server.jupyter_environment import JupyterEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


class FakeSandbox:
    sandbox_id = "fake-sandbox"

    def __init__(self) -> None:
        self.killed = False
        self.shell_commands: list[str] = []
        self.process_commands: list[str] = []

    def run_code(self, code: str) -> CellResult:
        return CellResult(
            stdout=f"ran: {code}",
            stderr="",
            error=None,
            error_name=None,
            text_results=[],
            images=[],
            execution_count=1,
            success=True,
        )

    def run_shell(self, command: str) -> CellResult:
        self.shell_commands.append(command)
        return self._respond(command)

    def run_command(self, command: str, timeout_s: float = 120) -> CellResult:
        self.process_commands.append(command)
        return self._respond(command)

    def _respond(self, command: str) -> CellResult:
        if "exit 1" in command:
            return CellResult(
                stdout="",
                stderr="failed",
                error="failed",
                error_name="CommandError",
                text_results=[],
                images=[],
                execution_count=1,
                success=False,
            )
        if command.startswith("cat /home/user/logs/verifier/reward.txt"):
            return CellResult(
                stdout="",
                stderr="",
                error=None,
                error_name=None,
                text_results=[],
                images=[],
                execution_count=1,
                success=True,
            )
        return CellResult(
            stdout=f"shell: {command}",
            stderr="",
            error=None,
            error_name=None,
            text_results=[],
            images=[],
            execution_count=1,
            success=True,
        )

    def kill(self) -> None:
        self.killed = True


def _extract_text(result) -> str:
    if hasattr(result, "content") and result.content:
        return result.content[0].text
    if hasattr(result, "data"):
        return str(result.data)
    return str(result)


def test_lists_notebook_tools_without_reset():
    env = JupyterEnvironment()

    obs = env.step(ListToolsAction())

    tool_names = {tool.name for tool in obs.tools}
    assert "add_and_execute_code_cell" in tool_names
    assert "edit_and_execute_current_cell" in tool_names
    assert "execute_shell_command" in tool_names
    assert "get_notebook_state" in tool_names
    assert "final_answer" in tool_names


def test_reset_without_e2b_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    env = JupyterEnvironment()

    obs = env.reset()

    assert obs.done is True
    assert obs.metadata["status"] == "error"
    assert "E2B_API_KEY" in obs.metadata["error"]


def test_code_tool_updates_notebook_state():
    env = JupyterEnvironment()
    env._sandbox = FakeSandbox()
    env._state = JupyterState(episode_id="episode-1", sandbox_id="fake-sandbox")

    obs = env.step(
        CallToolAction(
            tool_name="add_and_execute_code_cell",
            arguments={"code": "print('hello')"},
        )
    )

    assert obs.error is None
    assert "ran: print('hello')" in _extract_text(obs.result)
    assert env.state.step_count == 1
    assert len(env.state.cells) == 1
    assert env.state.cells[0].code == "print('hello')"


def test_shell_tool_updates_notebook_state():
    env = JupyterEnvironment()
    env._sandbox = FakeSandbox()
    env._state = JupyterState(episode_id="episode-1", sandbox_id="fake-sandbox")

    obs = env.step(
        CallToolAction(
            tool_name="execute_shell_command",
            arguments={"command": "pwd"},
        )
    )

    assert obs.error is None
    assert "shell: pwd" in _extract_text(obs.result)
    assert env.state.step_count == 1
    assert env.state.cells[0].cell_type == "shell"


def test_reset_runs_setup_and_stores_verify_commands(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "fake-key")
    env = JupyterEnvironment()
    fake_sandbox = FakeSandbox()
    monkeypatch.setattr(
        "jupyter_env.server.jupyter_environment.E2BSandbox",
        lambda api_key: fake_sandbox,
    )

    obs = env.reset(setup=["echo setup"], verify=["test -f answer.py"])

    assert obs.done is False
    assert fake_sandbox.shell_commands == [
        "mkdir -p /home/user/logs/verifier",
        "echo setup",
    ]
    assert env.state.setup_results[0].command == "echo setup"
    assert env.state.setup_results[0].success is True
    assert env.state.verify_commands == ["test -f answer.py"]
    assert obs.metadata["verify_commands"] == ["test -f answer.py"]


def test_reset_fails_when_setup_command_fails(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "fake-key")
    env = JupyterEnvironment()
    monkeypatch.setattr(
        "jupyter_env.server.jupyter_environment.E2BSandbox",
        lambda api_key: FakeSandbox(),
    )

    obs = env.reset(setup=["exit 1"], verify=["test -f answer.py"])

    assert obs.done is True
    assert obs.metadata["status"] == "error"
    assert obs.metadata["setup_results"][0]["success"] is False


def test_final_answer_runs_verify_commands():
    env = JupyterEnvironment()
    fake_sandbox = FakeSandbox()
    env._sandbox = fake_sandbox
    env._state = JupyterState(
        episode_id="episode-1",
        sandbox_id="fake-sandbox",
        verify_commands=["test -f answer.py", "exit 1"],
    )

    obs = env.step(
        CallToolAction(
            tool_name="final_answer",
            arguments={"answer": "done"},
        )
    )

    assert obs.error is None
    assert "Verification: 1/2 passed; reward=0.5" in _extract_text(obs.result)
    assert env.state.submitted_answer == "done"
    assert env.state.last_reward == 0.5
    assert [result.command for result in env.state.verify_results] == [
        "test -f answer.py",
        "exit 1",
    ]
    assert "exit 1" in fake_sandbox.process_commands
    assert "exit 1" not in fake_sandbox.shell_commands


class _Kernel:
    """A kernel whose namespace persists between cells, as E2B's run_code does."""

    def __init__(self) -> None:
        self.namespace: dict = {}
        self.cells: list[str] = []

    def run_code(self, code: str, **kwargs):
        self.cells.append(code)
        stdout, stderr, error = io.StringIO(), io.StringIO(), None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, self.namespace)
        except BaseException as exc:  # a kernel reports SystemExit, it doesn't exit
            error = SimpleNamespace(
                name=type(exc).__name__, value=str(exc), traceback=""
            )
        return SimpleNamespace(
            error=error,
            logs=SimpleNamespace(
                stdout=[stdout.getvalue()] if stdout.getvalue() else [],
                stderr=[stderr.getvalue()] if stderr.getvalue() else [],
            ),
            results=[],
            execution_count=len(self.cells),
        )


class _CommandExit(Exception):
    """Stands in for `e2b.CommandExitException`, raised on a non-zero exit."""

    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        super().__init__(f"Command exited with code {exit_code}")
        self.exit_code, self.stdout, self.stderr = exit_code, stdout, stderr


class _Processes:
    """E2B's process API: each command is a fresh process with a real exit status.

    `/home/user` is mapped to a temporary directory so the commands can run here.
    """

    def __init__(self, home) -> None:
        self.home = str(home)
        self.calls: list[dict] = []
        # Captured now, before any notebook cell gets a chance to rebind it.
        self._run = subprocess.run

    def run(self, cmd, background=None, user=None, cwd=None, timeout=None, **kwargs):
        self.calls.append(
            {
                "cmd": cmd,
                "background": background,
                "user": user,
                "cwd": cwd,
                "timeout": timeout,
            }
        )
        done = self._run(
            ["/bin/bash", "-c", cmd.replace("/home/user", self.home)],
            cwd=(cwd or "").replace("/home/user", self.home) or None,
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
        return _Handle(done=done)


class _Handle:
    """What `commands.run(background=True)` returns: wait for it, or kill it."""

    def __init__(self, done=None, wait_error=None) -> None:
        self._done, self._wait_error = done, wait_error
        self.killed = False

    def wait(self):
        if self._wait_error is not None:
            raise self._wait_error
        if self._done.returncode:
            raise _CommandExit(
                self._done.returncode, self._done.stdout, self._done.stderr
            )
        return SimpleNamespace(
            stdout=self._done.stdout, stderr=self._done.stderr, exit_code=0
        )

    def kill(self) -> bool:
        self.killed = True
        return True


def _environment_on_real_sandbox(home, verify_commands):
    kernel, processes = _Kernel(), _Processes(home)
    sandbox = E2BSandbox.__new__(E2BSandbox)
    sandbox._sbx = SimpleNamespace(run_code=kernel.run_code, commands=processes)
    sandbox.sandbox_id = "fake-sandbox"
    env = JupyterEnvironment()
    env._sandbox = sandbox
    env._state = JupyterState(
        episode_id="episode-1",
        sandbox_id="fake-sandbox",
        verify_commands=verify_commands,
    )
    return env, kernel, processes


def _run_cell(env, code):
    env.step(
        CallToolAction(tool_name="add_and_execute_code_cell", arguments={"code": code})
    )


def _submit(env):
    env.step(CallToolAction(tool_name="final_answer", arguments={"answer": "done"}))


# Reproduction from #1210: a cell that rebinds subprocess.run used to decide the
# outcome of every verify command, and of the reward-file read.
_REBIND_SUBPROCESS = """\
import subprocess

class _Fake:
    returncode = 0
    stdout = "1.0"
    stderr = ""

subprocess.run = lambda *a, **k: _Fake()
"""


def test_verification_ignores_a_rebound_subprocess(monkeypatch, tmp_path):
    # Restores the real function after the cell below replaces it.
    monkeypatch.setattr(subprocess, "run", subprocess.run)
    env, kernel, processes = _environment_on_real_sandbox(
        tmp_path, ["exit 1", "exit 1"]
    )

    _run_cell(env, _REBIND_SUBPROCESS)
    _submit(env)

    assert [result.success for result in env.state.verify_results] == [False, False]
    assert env.state.last_reward == 0.0
    assert not any("exit 1" in cell for cell in kernel.cells)


def test_reward_file_is_read_outside_the_kernel(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", subprocess.run)
    env, _, _ = _environment_on_real_sandbox(
        tmp_path,
        ["echo 0.8 > /home/user/logs/verifier/reward.txt", "exit 1"],
    )

    _run_cell(env, _REBIND_SUBPROCESS)
    _submit(env)

    # The rebound function would have reported "1.0" for the read.
    assert env.state.last_reward == 0.8


def test_verification_does_not_inherit_the_notebook_directory(monkeypatch, tmp_path):
    home, elsewhere = tmp_path / "home", tmp_path / "elsewhere"
    home.mkdir()
    elsewhere.mkdir()
    # Restores the working directory after the cell changes it.
    monkeypatch.chdir(tmp_path)
    env, _, processes = _environment_on_real_sandbox(home, ["pwd"])

    _run_cell(env, f"import os\nos.chdir({str(elsewhere)!r})\n")
    _submit(env)

    assert [result.output for result in env.state.verify_results] == [str(home)]
    assert {
        (call["background"], call["user"], call["cwd"], call["timeout"])
        for call in processes.calls
    } == {(True, "root", "/home/user", 120)}


def _sandbox_with_handle(handle=None, start_error=None):
    sandbox = E2BSandbox.__new__(E2BSandbox)

    def run(cmd, **kwargs):
        if start_error is not None:
            raise start_error
        return handle

    sandbox._sbx = SimpleNamespace(commands=SimpleNamespace(run=run))
    return sandbox


def test_run_command_reports_a_non_zero_exit():
    handle = _Handle(wait_error=_CommandExit(3, "partial output", "boom"))

    result = _sandbox_with_handle(handle).run_command("false")

    assert not result.success
    assert result.error == "exit code 3"
    assert (result.stdout, result.stderr) == ("partial output", "boom")
    # The command exited, so there is nothing to stop.
    assert not handle.killed


def test_run_command_kills_a_command_it_stops_waiting_for():
    handle = _Handle(wait_error=TimeoutError("context deadline exceeded"))

    result = _sandbox_with_handle(handle).run_command("sleep 999")

    assert not result.success
    assert result.error == "TimeoutError: context deadline exceeded"
    assert handle.killed


def test_run_command_reports_a_command_that_could_not_start():
    sandbox = _sandbox_with_handle(start_error=ConnectionError("sandbox unreachable"))

    result = sandbox.run_command("true")

    assert not result.success
    assert result.error == "ConnectionError: sandbox unreachable"


def test_run_command_matches_the_sdk_exception():
    e2b = pytest.importorskip("e2b")
    exc = e2b.CommandExitException(stderr="boom", stdout="", exit_code=2, error=None)
    handle = _Handle(wait_error=exc)

    result = _sandbox_with_handle(handle).run_command("false")

    assert not result.success
    assert result.error == "exit code 2"
    assert result.stderr == "boom"
    assert not handle.killed
