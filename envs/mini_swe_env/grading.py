# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Binary grading for SWE-Gym tasks via ``swebench.harness.grading``.

Wraps the upstream ``get_eval_report()`` function to produce a binary
reward: **1.0** if the instance is fully resolved (all FAIL_TO_PASS tests
now pass, all PASS_TO_PASS tests still pass), **0.0** otherwise.

Two entry points:

1. :func:`grade_from_log` — grade from a test log file (typical
   in-sandbox flow: run eval script → parse log → grade).

2. :func:`grade_from_test_output` — grade from raw test stdout/stderr
   (useful when the test runner output is captured directly).

Both require a :class:`SWEGymTask` for the ground-truth test lists.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from .models import SWEGymTask

_log = logging.getLogger(__name__)


class GradingError(RuntimeError):
    """Raised when grading fails due to infrastructure issues."""


# ── Public API ─────────────────────────────────────────────────────────────


def grade_from_log(
    task: SWEGymTask,
    log_path: str,
    *,
    model_name: str = "agent",
    model_patch: str | None = None,
) -> GradeResult:
    """Grade a submission using a test log file.

    Args:
        task: The SWE-Gym task (provides FAIL_TO_PASS, PASS_TO_PASS, etc.).
        log_path: Path to the evaluation log file (output of the
            ``swebench`` eval script).
        model_name: Identifier for the model producing the patch.
        model_patch: The git diff of the agent's changes.  If ``None``,
            a placeholder is used (``get_eval_report`` only checks
            whether the patch field is non-None, the content is not
            re-applied).

    Returns:
        A :class:`GradeResult` with binary ``reward`` and metadata.
    """
    if not os.path.isfile(log_path):
        raise GradingError(f"Log file not found: {log_path}")

    test_spec = _task_to_test_spec(task)
    prediction = {
        "instance_id": task.instance_id,
        "model_name_or_path": model_name,
        "model_patch": model_patch if model_patch is not None else "placeholder",
    }

    from swebench.harness.grading import get_eval_report

    report = get_eval_report(
        test_spec=test_spec,
        prediction=prediction,
        test_log_path=log_path,
        include_tests_status=True,
    )

    return _report_to_grade_result(report, task.instance_id)


def grade_from_test_output(
    task: SWEGymTask,
    test_output: str,
    *,
    model_name: str = "agent",
    model_patch: str | None = None,
) -> GradeResult:
    """Grade a submission from raw test output text.

    Writes the output to a temporary file and delegates to
    :func:`grade_from_log`.

    Args:
        task: The SWE-Gym task.
        test_output: Combined stdout/stderr from running the eval script.
        model_name: Identifier for the model.
        model_patch: The git diff (or placeholder).

    Returns:
        A :class:`GradeResult`.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, prefix="swe_grade_"
    ) as f:
        f.write(test_output)
        tmp_path = f.name

    try:
        return grade_from_log(
            task,
            tmp_path,
            model_name=model_name,
            model_patch=model_patch,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def make_eval_script(task: SWEGymTask) -> str:
    """Generate the swebench evaluation shell script for a task.

    This script is intended to be deployed into the sandbox and run
    after the agent applies its patch.  The script:

    1. Applies the ``test_patch`` (new/changed tests from the task).
    2. Runs the test command for the repo/version.
    3. Reverts the ``test_patch`` so the working tree is clean.

    The output is bounded by ``>>>>> Start Test Output`` / ``>>>>> End
    Test Output`` markers that ``swebench.harness.grading.get_logs_eval``
    expects.

    Returns:
        Shell script content as a string.
    """
    test_spec = _task_to_test_spec(task)

    return test_spec.eval_script


# ── GradeResult ────────────────────────────────────────────────────────────


class GradeResult:
    """Result of grading a SWE-Gym submission.

    Attributes:
        reward: Binary reward (1.0 if resolved, 0.0 otherwise).
        resolved: Whether the instance was fully resolved.
        patch_applied: Whether the agent's patch was successfully applied.
        tests_status: Detailed per-test status (if available).
        instance_id: The task's instance id.
    """

    __slots__ = (
        "reward",
        "resolved",
        "patch_applied",
        "tests_status",
        "instance_id",
    )

    def __init__(
        self,
        *,
        reward: float,
        resolved: bool,
        patch_applied: bool,
        tests_status: dict[str, Any] | None = None,
        instance_id: str = "",
    ):
        self.reward = reward
        self.resolved = resolved
        self.patch_applied = patch_applied
        self.tests_status = tests_status
        self.instance_id = instance_id

    def __repr__(self) -> str:
        return (
            f"GradeResult(reward={self.reward}, resolved={self.resolved}, "
            f"patch_applied={self.patch_applied}, instance_id={self.instance_id!r})"
        )


# ── Internal helpers ───────────────────────────────────────────────────────


def _task_to_test_spec(task: SWEGymTask) -> Any:
    """Convert a :class:`SWEGymTask` to a ``swebench`` ``TestSpec``."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    # Build the instance dict that make_test_spec expects.
    instance: dict[str, Any] = {
        "instance_id": task.instance_id,
        "repo": task.repo,
        "base_commit": task.base_commit,
        "version": task.version,
        "patch": task.patch,
        "test_patch": task.test_patch,
        "problem_statement": task.problem_statement,
        "hints_text": task.hints_text,
        "FAIL_TO_PASS": list(task.FAIL_TO_PASS),
        "PASS_TO_PASS": list(task.PASS_TO_PASS),
    }

    return make_test_spec(instance)


def _report_to_grade_result(
    report: dict[str, Any],
    instance_id: str,
) -> GradeResult:
    """Convert a ``get_eval_report`` output dict to a :class:`GradeResult`."""
    if instance_id not in report:
        # Report didn't contain this instance — treat as failure.
        _log.warning("Instance %s not in eval report", instance_id)
        return GradeResult(
            reward=0.0,
            resolved=False,
            patch_applied=False,
            instance_id=instance_id,
        )

    entry = report[instance_id]
    resolved = bool(entry.get("resolved", False))
    patch_applied = bool(entry.get("patch_successfully_applied", False))
    tests_status = entry.get("tests_status")

    return GradeResult(
        reward=1.0 if resolved else 0.0,
        resolved=resolved,
        patch_applied=patch_applied,
        tests_status=tests_status,
        instance_id=instance_id,
    )


__all__ = [
    "GradeResult",
    "GradingError",
    "grade_from_log",
    "grade_from_test_output",
    "make_eval_script",
]
