"""SWE tool environment for TRL AsyncGRPO / GRPO training.

Implements the ``environment_factory`` protocol expected by
``AsyncGRPOTrainer`` and ``GRPOTrainer``:

- ``reset(**row)`` — create a sandbox from the SWE-Gym per-task image.
- ``bash(command)`` — execute a shell command in ``/testbed``.
- ``answer()`` — run SWE-Gym grading and return resolved/not.

TRL discovers public methods as tools, drives the model's generation
loop, handles tokenization, logprobs, token IDs, weight sync, and
sample assembly.  We only provide the environment.

Usage with ``AsyncGRPOTrainer``::

    from mini_swe_env.async_grpo import SWEToolEnv, swe_reward

    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-1.7B",
        reward_funcs=swe_reward,
        train_dataset=dataset,
        environment_factory=SWEToolEnv.factory(backend),
    )
    trainer.train()
"""

from __future__ import annotations

import json
import logging
import shlex
from typing import Any, Callable

from ..grading import grade_from_case_results
from ..models import SWEGymTask, SWETask

_log = logging.getLogger(__name__)

TESTBED = "/testbed"
VERIFY_TIMEOUT_S = 300


class SWEToolEnv:
    """TRL-compatible SWE environment with ``bash`` and ``answer`` tools.

    One instance is created per ``max_inflight_tasks`` slot.  TRL calls
    ``reset(**row)`` before each generation, then the model calls
    ``bash(command=...)`` and ``answer()`` as tool calls.
    """

    def __init__(self, sandbox_backend: Any) -> None:
        self._backend = sandbox_backend
        self._sandbox: Any | None = None
        self._task: SWETask | None = None
        self._gym_task: SWEGymTask | None = None
        self.reward: float = 0.0
        self.resolved: bool = False
        self._answer_called: bool = False

    # ── TRL lifecycle ──────────────────────────────────────────────

    def reset(self, **kwargs: Any) -> str | None:
        """Create a fresh sandbox for the task described in *kwargs*.

        Called by TRL before each generation.  Receives all columns from
        the dataset row as keyword arguments.
        """
        self._cleanup()
        self.reward = 0.0
        self.resolved = False
        self._answer_called = False

        task_json = kwargs.get("task_json", "")
        if not task_json:
            return None

        raw = json.loads(task_json)
        self._gym_task = SWEGymTask(**raw)
        self._task = self._gym_task.to_swe_task()

        self._sandbox = self._backend.create(
            timeout_s=self._task.timeout_s,
            image=self._task.sandbox_image,
        )

        return None

    # ── Tools (discovered by TRL via inspect.getmembers) ──────────

    def bash(self, command: str) -> str:
        """Execute a shell command in the repository at /testbed.

        Args:
            command: Shell command to run.

        Returns:
            Combined stdout and stderr output.
        """
        if self._sandbox is None:
            return "Error: no sandbox — call reset() first."
        result = self._sandbox.exec(command, cwd=TESTBED, timeout=VERIFY_TIMEOUT_S)
        stdout = (result.stdout or "")[-8000:]
        stderr = (result.stderr or "")[-4000:]
        if stderr:
            return f"{stdout}\nSTDERR:\n{stderr}"
        return stdout

    def answer(self) -> str:
        """Submit your solution for grading.

        Runs the SWE-Gym test suite (FAIL_TO_PASS / PASS_TO_PASS) and
        returns whether the issue is resolved.  Can only be called once.

        Returns:
            A string indicating whether the issue was resolved.
        """
        if self._answer_called:
            return "Error: answer() already called."
        self._answer_called = True

        if self._sandbox is None or self._gym_task is None:
            self.reward = 0.0
            self.resolved = False
            return "Error: no sandbox or task — call reset() first."

        try:
            reward, resolved = _grade_submission(
                self._sandbox, self._task, self._gym_task,
            )
            self.reward = reward
            self.resolved = resolved
            return f"Resolved: {str(resolved).lower()}"
        except Exception as exc:
            _log.exception("SWE grading failed for %s", self._gym_task.instance_id)
            self.reward = 0.0
            self.resolved = False
            return f"Grading error: {type(exc).__name__}: {exc}"

    # ── Cleanup ────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
            except Exception:
                pass
            self._sandbox = None
        self._task = None
        self._gym_task = None

    def __del__(self) -> None:
        self._cleanup()

    # ── Factory helper ─────────────────────────────────────────────

    @staticmethod
    def factory(sandbox_backend: Any) -> Callable[[], "SWEToolEnv"]:
        """Return an ``environment_factory`` callable for TRL."""
        def _create() -> SWEToolEnv:
            return SWEToolEnv(sandbox_backend)
        return _create


# ── Reward function ────────────────────────────────────────────────────


def swe_reward(
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    """Reward function for SWE training.

    Scans each completion for an ``answer`` tool result message and
    parses ``"Resolved: true"`` / ``"Resolved: false"``.

    Works with both ``GRPOTrainer`` (receives ``environments=``) and
    ``AsyncGRPOTrainer`` (does not — parses completion messages).
    """
    # If GRPOTrainer passes environments directly, use them.
    environments = kwargs.get("environments")
    if environments is not None:
        return [float(getattr(env, "reward", 0.0)) for env in environments]

    # AsyncGRPOTrainer path: parse completion messages.
    rewards: list[float] = []
    for completion in completions:
        reward = 0.0
        if isinstance(completion, list):
            for msg in completion:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "tool" and msg.get("name") == "answer":
                    content = str(msg.get("content", ""))
                    if "resolved: true" in content.lower():
                        reward = 1.0
                    break
        rewards.append(reward)
    return rewards


# ── Grading (extracted from harness.py) ────────────────────────────────


def _grade_submission(
    sandbox: Any,
    task: SWETask,
    gym_task: SWEGymTask,
) -> tuple[float, bool]:
    """Run SWE-Gym grading: revert test files, apply test_patch, run tests."""
    touched_files = _extract_paths_from_test_patch(gym_task.test_patch)

    # Revert test files to base_commit state before applying test_patch.
    _revert_test_files(sandbox, base_commit=task.base_commit, paths=touched_files)

    # Apply the test patch.
    patch_path = "/tmp/.openenv_swe_test_patch.diff"
    sandbox.write_text(patch_path, gym_task.test_patch)
    result = sandbox.exec(
        f"git apply --whitespace=nowarn {shlex.quote(patch_path)}",
        cwd=TESTBED,
        timeout=30,
    )
    if result.exit_code != 0:
        raise RuntimeError(
            f"failed to apply test_patch: {(result.stderr or result.stdout or '').strip()}"
        )

    # Run each test case.
    cases: list[str] = []
    seen: set[str] = set()
    for case in [*gym_task.FAIL_TO_PASS, *gym_task.PASS_TO_PASS]:
        if case not in seen:
            seen.add(case)
            cases.append(case)

    case_results: dict[str, bool] = {}
    for case in cases:
        cmd = f"python -m pytest -q --maxfail=1 {shlex.quote(case)}"
        run = sandbox.exec(cmd, cwd=TESTBED, timeout=VERIFY_TIMEOUT_S)
        case_results[case] = run.exit_code == 0

    grade = grade_from_case_results(gym_task, case_results)

    # Best-effort cleanup.
    try:
        _revert_test_files(sandbox, base_commit=task.base_commit, paths=touched_files)
    except Exception:
        pass

    return float(grade.reward), bool(grade.resolved)


def _extract_paths_from_test_patch(test_patch: str) -> list[str]:
    paths: list[str] = []
    for line in (test_patch or "").splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            if path and path != "/dev/null":
                paths.append(path)
    return sorted(set(paths))


def _revert_test_files(
    sandbox: Any,
    *,
    base_commit: str,
    paths: list[str],
) -> None:
    for path in paths:
        has_file = sandbox.exec(
            f"git cat-file -e {shlex.quote(f'{base_commit}:{path}')}",
            cwd=TESTBED,
            timeout=10,
        )
        if has_file.exit_code == 0:
            cmd = f"git checkout --quiet {shlex.quote(base_commit)} -- {shlex.quote(path)}"
        else:
            cmd = f"rm -f -- {shlex.quote(path)}"
        result = sandbox.exec(cmd, cwd=TESTBED, timeout=20)
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to revert {path}: {(result.stderr or result.stdout or '').strip()}"
            )


__all__ = [
    "SWEToolEnv",
    "swe_reward",
]
