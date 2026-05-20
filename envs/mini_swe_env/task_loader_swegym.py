# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE-Gym task loader.

Loads tasks from the HuggingFace ``SWE-Gym/SWE-Gym`` and
``SWE-Gym/SWE-Gym-Lite`` datasets and converts them to
:class:`SWEGymTask` instances.

Usage::

    from mini_swe_env.task_loader_swegym import load_swegym_tasks

    # Full dataset (2,438 tasks)
    tasks = load_swegym_tasks("full")

    # Lite subset (230 tasks)
    tasks = load_swegym_tasks("lite")
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from .models import DEFAULT_TIMEOUT_S, SWEGymTask, SWETask

_log = logging.getLogger(__name__)

# HuggingFace dataset identifiers.
_HF_DATASETS: dict[str, str] = {
    "full": "SWE-Gym/SWE-Gym",
    "lite": "SWE-Gym/SWE-Gym-Lite",
}

# All SWE-Gym datasets use the "train" split.
_HF_SPLIT = "train"


class SWEGymLoadError(ValueError):
    """Raised when a SWE-Gym dataset cannot be loaded or a row is invalid."""


# ── Public API ─────────────────────────────────────────────────────────────


def load_swegym_tasks(
    variant: str = "lite",
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> list[SWEGymTask]:
    """Load SWE-Gym tasks from HuggingFace.

    Args:
        variant: ``"lite"`` (230 tasks) or ``"full"`` (2,438 tasks).
        timeout_s: Per-task timeout override.

    Returns:
        List of validated :class:`SWEGymTask` instances.

    Raises:
        SWEGymLoadError: If the variant is unknown or the dataset fails
            to load.
    """
    dataset_name = _HF_DATASETS.get(variant)
    if dataset_name is None:
        raise SWEGymLoadError(
            f"Unknown variant {variant!r}; choose from {sorted(_HF_DATASETS)}"
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SWEGymLoadError(
            "The 'datasets' package is required to load SWE-Gym tasks.  "
            "Install it with: pip install datasets"
        ) from exc

    _log.info("Loading SWE-Gym dataset %s (split=%s) ...", dataset_name, _HF_SPLIT)

    try:
        ds = load_dataset(dataset_name, split=_HF_SPLIT)
    except Exception as exc:
        raise SWEGymLoadError(
            f"Failed to load HuggingFace dataset {dataset_name!r}: {exc}"
        ) from exc

    tasks: list[SWEGymTask] = []
    for idx, row in enumerate(ds):
        try:
            task = _row_to_swegym_task(row, timeout_s=timeout_s)
            validate_swegym_task(task)
            tasks.append(task)
        except Exception as exc:
            _log.warning(
                "Skipping row %d (%s): %s",
                idx,
                row.get("instance_id", "?"),
                exc,
            )

    _log.info("Loaded %d SWE-Gym tasks from %s", len(tasks), dataset_name)
    return tasks


def load_swegym_tasks_from_dicts(
    rows: Sequence[dict[str, Any]],
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> list[SWEGymTask]:
    """Convert raw dicts (e.g. from JSONL) to :class:`SWEGymTask` instances.

    Args:
        rows: Iterable of dicts matching the SWE-Gym HF schema.
        timeout_s: Per-task timeout override.

    Returns:
        List of validated :class:`SWEGymTask` instances.
    """
    tasks: list[SWEGymTask] = []
    for row in rows:
        task = _row_to_swegym_task(row, timeout_s=timeout_s)
        validate_swegym_task(task)
        tasks.append(task)
    return tasks


def swegym_task_to_swe_task(task: SWEGymTask) -> SWETask:
    """Convert a :class:`SWEGymTask` to the internal :class:`SWETask`.

    Convenience wrapper around ``task.to_swe_task()``.
    """
    return task.to_swe_task()


def validate_swegym_task(task: SWEGymTask) -> None:
    """Validate a :class:`SWEGymTask` and raise on errors."""
    errors: list[str] = []

    for field_name in ("instance_id", "repo", "base_commit", "problem_statement"):
        value = getattr(task, field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field_name} must be a non-empty string")

    if not isinstance(task.version, str):
        errors.append("version must be a string")

    if not isinstance(task.patch, str) or not task.patch.strip():
        errors.append("patch (ground truth) must be a non-empty string")

    if not isinstance(task.test_patch, str) or not task.test_patch.strip():
        errors.append("test_patch must be a non-empty string")

    if not isinstance(task.FAIL_TO_PASS, list) or not task.FAIL_TO_PASS:
        errors.append("FAIL_TO_PASS must be a non-empty list")

    if not isinstance(task.PASS_TO_PASS, list):
        errors.append("PASS_TO_PASS must be a list")

    if not isinstance(task.timeout_s, int) or task.timeout_s <= 0:
        errors.append("timeout_s must be a positive int")

    if errors:
        raise SWEGymLoadError(
            f"Invalid SWEGymTask {task.instance_id!r}: {'; '.join(errors)}"
        )


def get_instance_image(instance_id: str) -> str:
    """Derive the per-task Docker image name from an instance id.

    SWE-Gym convention:
      ``xingyaoww/sweb.eval.x86_64.<sanitised_instance_id>:latest``
    where ``__`` in the instance id is replaced with ``_1776_``.
    """
    sanitised = instance_id.lower().replace("__", "_s_")
    return f"xingyaoww/sweb.eval.x86_64.{sanitised}:latest"


# ── Internal helpers ───────────────────────────────────────────────────────


def _coerce_string_list(value: Any) -> list[str]:
    """Coerce a value to ``list[str]``.

    Handles: ``list``, JSON-encoded string, ``None``.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        return [str(parsed)]
    return [str(value)]


def _row_to_swegym_task(
    row: dict[str, Any],
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> SWEGymTask:
    """Convert one HF dataset row to an :class:`SWEGymTask`."""
    instance_id = row.get("instance_id", "")
    if not instance_id:
        raise SWEGymLoadError("Row missing required field 'instance_id'")

    return SWEGymTask(
        instance_id=str(instance_id),
        repo=str(row.get("repo", "")),
        base_commit=str(row.get("base_commit", "")),
        problem_statement=str(row.get("problem_statement", "")),
        version=str(row.get("version", "")),
        patch=str(row.get("patch", "")),
        test_patch=str(row.get("test_patch", "")),
        FAIL_TO_PASS=_coerce_string_list(row.get("FAIL_TO_PASS")),
        PASS_TO_PASS=_coerce_string_list(row.get("PASS_TO_PASS")),
        hints_text=str(row.get("hints_text", "") or ""),
        created_at=str(row.get("created_at", "") or ""),
        timeout_s=timeout_s,
    )


__all__ = [
    "SWEGymLoadError",
    "get_instance_image",
    "load_swegym_tasks",
    "load_swegym_tasks_from_dicts",
    "swegym_task_to_swe_task",
    "validate_swegym_task",
]
