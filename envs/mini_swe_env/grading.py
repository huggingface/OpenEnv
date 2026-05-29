# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grading helpers for SWE-Gym tasks.

Supports both the canonical binary SWE-Gym reward and an optional
case-fraction shaping mode for RL experiments:

- ``binary``: reward = 1.0 iff every FAIL_TO_PASS test passes AND every
  PASS_TO_PASS test still passes, else 0.0.
- ``case_fraction``: reward = passed_cases / total_cases, while
  ``resolved`` still requires every FAIL_TO_PASS and PASS_TO_PASS case to
  pass.
"""

from __future__ import annotations

from typing import Any

from .models import SWEGymTask


class GradingError(RuntimeError):
    """Raised when grading input is invalid."""


class GradeResult:
    """Result of grading a SWE-Gym submission."""

    __slots__ = (
        "reward",
        "case_fraction",
        "resolved",
        "patch_applied",
        "tests_status",
        "instance_id",
    )

    def __init__(
        self,
        *,
        reward: float,
        case_fraction: float,
        resolved: bool,
        patch_applied: bool,
        tests_status: dict[str, Any] | None = None,
        instance_id: str = "",
    ):
        self.reward = reward
        self.case_fraction = case_fraction
        self.resolved = resolved
        self.patch_applied = patch_applied
        self.tests_status = tests_status
        self.instance_id = instance_id

    def __repr__(self) -> str:
        return (
            f"GradeResult(reward={self.reward}, case_fraction={self.case_fraction}, "
            f"resolved={self.resolved}, "
            f"patch_applied={self.patch_applied}, instance_id={self.instance_id!r})"
        )


def grade_from_case_results(
    task: SWEGymTask,
    case_results: dict[str, bool],
    *,
    patch_applied: bool = True,
    reward_mode: str = "binary",
) -> GradeResult:
    """Grade directly from per-test-case outcomes.

    Args:
        task: SWE-Gym task with FAIL_TO_PASS and PASS_TO_PASS lists.
        case_results: Mapping ``test_case -> passed``.
        patch_applied: Whether test patch was successfully applied.
        reward_mode: ``"binary"`` or ``"case_fraction"``.
    """

    if not isinstance(case_results, dict):
        raise GradingError("case_results must be a dict[str, bool]")
    if reward_mode not in {"binary", "case_fraction"}:
        raise GradingError(
            f"Unknown reward_mode {reward_mode!r}; expected 'binary' or 'case_fraction'"
        )

    def _passed(case: str) -> bool:
        return bool(case_results.get(case, False))

    all_cases = list(dict.fromkeys([*task.FAIL_TO_PASS, *task.PASS_TO_PASS]))
    passed_cases = sum(1 for case in all_cases if _passed(case))
    total_cases = len(all_cases)
    case_fraction = (passed_cases / total_cases) if total_cases else 0.0

    resolved = all(_passed(case) for case in task.FAIL_TO_PASS) and all(
        _passed(case) for case in task.PASS_TO_PASS
    )

    tests_status = {
        "FAIL_TO_PASS": {
            case: "PASSED" if _passed(case) else "FAILED"
            for case in task.FAIL_TO_PASS
        },
        "PASS_TO_PASS": {
            case: "PASSED" if _passed(case) else "FAILED"
            for case in task.PASS_TO_PASS
        },
    }

    reward = 1.0 if resolved else 0.0
    if reward_mode == "case_fraction":
        reward = case_fraction

    return GradeResult(
        reward=reward,
        case_fraction=case_fraction,
        resolved=resolved,
        patch_applied=patch_applied,
        tests_status=tests_status,
        instance_id=task.instance_id,
    )


__all__ = [
    "GradeResult",
    "GradingError",
    "grade_from_case_results",
]
