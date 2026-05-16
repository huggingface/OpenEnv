# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Binary grading for SWE-Gym tasks.

This module is intentionally SWE-Gym-native and does not depend on
external repo/version parser maps.

The grading contract matches SWE-Gym semantics:

- reward = 1.0 iff every FAIL_TO_PASS test passes AND every PASS_TO_PASS
  test still passes.
- reward = 0.0 otherwise.
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


def grade_from_case_results(
    task: SWEGymTask,
    case_results: dict[str, bool],
    *,
    patch_applied: bool = True,
) -> GradeResult:
    """Grade directly from per-test-case outcomes.

    Args:
        task: SWE-Gym task with FAIL_TO_PASS and PASS_TO_PASS lists.
        case_results: Mapping ``test_case -> passed``.
        patch_applied: Whether test patch was successfully applied.
    """

    if not isinstance(case_results, dict):
        raise GradingError("case_results must be a dict[str, bool]")

    def _passed(case: str) -> bool:
        return bool(case_results.get(case, False))

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

    return GradeResult(
        reward=1.0 if resolved else 0.0,
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
