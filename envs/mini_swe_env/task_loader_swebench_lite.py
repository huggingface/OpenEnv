# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task adapter utilities for SWE-bench Lite.

Includes:
  - normalize dataset rows into ``SWETask``
  - deterministic mini-subset selection and train/eval split
  - explicit schema validation and skip reasons
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

SOURCE_NAME = "swebench_lite"
DEFAULT_TIMEOUT_S = 1800


class SWETaskValidationError(ValueError):
    """Raised when a task fails schema validation."""


class SWEBenchLiteAdapterError(ValueError):
    """Raised when a SWE-bench Lite row cannot be adapted."""


@dataclass(frozen=True)
class SWETask:
    task_id: str
    source: str
    instance_id: str
    repo: str
    base_commit: str
    instruction: str
    setup: list[str]
    verify: list[str]
    timeout_s: int = DEFAULT_TIMEOUT_S
    sandbox_image: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSONL-friendly dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class AdaptationSkip:
    row_index: int
    reason: str
    instance_id: str | None = None


def validate_swe_task(task: SWETask) -> None:
    """Validate a ``SWETask`` and raise explicit schema errors."""
    errors: list[str] = []

    for field_name in (
        "task_id",
        "source",
        "instance_id",
        "repo",
        "base_commit",
        "instruction",
    ):
        value = getattr(task, field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field_name} must be a non-empty string")

    if not isinstance(task.setup, list):
        errors.append("setup must be a list[str]")
    else:
        for idx, command in enumerate(task.setup):
            if not isinstance(command, str) or not command.strip():
                errors.append(f"setup[{idx}] must be a non-empty string")

    if not isinstance(task.verify, list):
        errors.append("verify must be a list[str]")
    elif not task.verify:
        errors.append("verify must contain at least one command")
    else:
        for idx, command in enumerate(task.verify):
            if not isinstance(command, str) or not command.strip():
                errors.append(f"verify[{idx}] must be a non-empty string")

    if not isinstance(task.timeout_s, int) or task.timeout_s <= 0:
        errors.append("timeout_s must be a positive int")

    if task.sandbox_image is not None and (
        not isinstance(task.sandbox_image, str) or not task.sandbox_image.strip()
    ):
        errors.append("sandbox_image must be None or a non-empty string")

    if not isinstance(task.metadata, dict):
        errors.append("metadata must be a dict")

    if errors:
        raise SWETaskValidationError("; ".join(errors))


def coerce_swe_task(value: SWETask | dict[str, Any]) -> SWETask:
    """Coerce an input mapping into a validated ``SWETask``."""
    if isinstance(value, SWETask):
        validate_swe_task(value)
        return value
    if not isinstance(value, dict):
        raise SWETaskValidationError(
            f"Expected SWETask or dict, got {type(value).__name__}"
        )
    task = SWETask(**value)
    validate_swe_task(task)
    return task


def adapt_swebench_lite_row(row: dict[str, Any]) -> SWETask:
    """Convert one SWE-bench Lite row into the internal ``SWETask`` shape."""
    if not isinstance(row, dict):
        raise SWEBenchLiteAdapterError(f"row must be a dict, got {type(row).__name__}")

    instance_id = _pick_str(row, "instance_id", required=True)
    repo = _pick_str(row, "repo", "repository", required=True)
    base_commit = _pick_str(
        row,
        "base_commit",
        "commit",
        "base_sha",
        "environment_setup_commit",
        required=True,
    )
    instruction = _pick_str(
        row,
        "instruction",
        "problem_statement",
        "prompt",
        required=True,
    )

    setup = _pick_commands(
        row,
        "setup",
        "setup_commands",
        "setup_script",
        "setup_scripts",
    )
    verify = _pick_commands(
        row,
        "verify",
        "verify_commands",
        "evaluation_commands",
        "test_commands",
    )

    if not verify:
        verify = _derive_verify_from_test_lists(row)

    timeout_s = _coerce_positive_int(row.get("timeout_s"), default=DEFAULT_TIMEOUT_S)
    sandbox_image = _pick_optional_str(row, "sandbox_image", "image")

    task_id = _pick_optional_str(row, "task_id") or f"{SOURCE_NAME}::{instance_id}"

    fail_to_pass = _parse_json_string_list(
        row.get("FAIL_TO_PASS", row.get("fail_to_pass"))
    )
    pass_to_pass = _parse_json_string_list(
        row.get("PASS_TO_PASS", row.get("pass_to_pass"))
    )
    hints_text = _pick_optional_str(row, "hints_text")

    metadata: dict[str, Any] = {
        "dataset": SOURCE_NAME,
        "created_at": _pick_optional_str(row, "created_at"),
        "version": _pick_optional_str(row, "version"),
        "fail_to_pass_count": len(fail_to_pass),
        "pass_to_pass_count": len(pass_to_pass),
    }
    if hints_text:
        # Keep metadata compact for JSONL task files.
        metadata["hints_preview"] = hints_text[:280]
    metadata = {k: v for k, v in metadata.items() if v not in (None, "")}

    task = SWETask(
        task_id=task_id,
        source=SOURCE_NAME,
        instance_id=instance_id,
        repo=repo,
        base_commit=base_commit,
        instruction=instruction,
        setup=setup,
        verify=verify,
        timeout_s=timeout_s,
        sandbox_image=sandbox_image,
        metadata=metadata,
    )

    try:
        validate_swe_task(task)
    except SWETaskValidationError as exc:
        raise SWEBenchLiteAdapterError(
            f"row {instance_id!r} failed schema validation: {exc}"
        ) from exc

    return task


def adapt_swebench_lite_rows(
    rows: Iterable[dict[str, Any]], *, strict: bool = False
) -> tuple[list[SWETask], list[AdaptationSkip]]:
    """Adapt an iterable of rows.

    When ``strict=False`` invalid rows are skipped and reported in ``AdaptationSkip``.
    When ``strict=True`` the first adapter error is raised.
    """
    tasks: list[SWETask] = []
    skipped: list[AdaptationSkip] = []

    for index, row in enumerate(rows, start=1):
        try:
            task = adapt_swebench_lite_row(row)
            tasks.append(task)
        except SWEBenchLiteAdapterError as exc:
            if strict:
                raise
            instance_id = None
            if isinstance(row, dict):
                maybe_id = row.get("instance_id")
                instance_id = str(maybe_id) if maybe_id else None
            skipped.append(
                AdaptationSkip(
                    row_index=index, reason=str(exc), instance_id=instance_id
                )
            )

    return tasks, skipped


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON rows from ``path``."""
    file_path = Path(path)
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(file_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SWEBenchLiteAdapterError(
                f"invalid JSON on line {line_no}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise SWEBenchLiteAdapterError(
                f"line {line_no} must decode to an object, got {type(payload).__name__}"
            )
        rows.append(payload)
    return rows


def write_tasks_jsonl(path: str | Path, tasks: Sequence[SWETask]) -> None:
    """Write validated tasks to JSONL."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for task in tasks:
        validate_swe_task(task)
        lines.append(json.dumps(task.to_dict(), sort_keys=True))
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def deterministic_train_eval_split(
    tasks: Sequence[SWETask],
    *,
    subset_size: int = 20,
    train_size: int = 16,
    seed: int = 17,
) -> tuple[list[SWETask], list[SWETask]]:
    """Create a deterministic subset and split without relying on RNG internals."""
    if subset_size <= 0:
        raise ValueError("subset_size must be > 0")
    if train_size <= 0:
        raise ValueError("train_size must be > 0")
    if train_size >= subset_size:
        raise ValueError("train_size must be < subset_size")

    deduped: dict[str, SWETask] = {}
    for task in tasks:
        validate_swe_task(task)
        deduped[task.task_id] = task

    unique_tasks = list(deduped.values())
    if len(unique_tasks) < subset_size:
        raise ValueError(
            f"need at least {subset_size} valid unique tasks, got {len(unique_tasks)}"
        )

    ranked = sorted(
        unique_tasks,
        key=lambda task: (_stable_seeded_rank(task.task_id, seed), task.task_id),
    )
    selected = ranked[:subset_size]
    return selected[:train_size], selected[train_size:]


def load_task_file(path: str | Path) -> list[SWETask]:
    """Load and validate existing task JSONL file."""
    return [coerce_swe_task(row) for row in read_jsonl_rows(path)]


def _stable_seeded_rank(task_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()
    return digest


def _pick_str(row: dict[str, Any], *keys: str, required: bool = False) -> str:
    value = _pick_optional_str(row, *keys)
    if value is not None:
        return value
    if required:
        key_list = ", ".join(keys)
        raise SWEBenchLiteAdapterError(f"missing required field(s): {key_list}")
    return ""


def _pick_optional_str(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _coerce_positive_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception as exc:
        raise SWEBenchLiteAdapterError(
            f"timeout_s must be an int, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise SWEBenchLiteAdapterError("timeout_s must be > 0")
    return parsed


def _pick_commands(row: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        if key in row:
            return _coerce_commands(row.get(key), field_name=key)
    return []


def _coerce_commands(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            else:
                return _coerce_commands(parsed, field_name=field_name)
        return [line.strip() for line in stripped.splitlines() if line.strip()]

    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        commands: list[str] = []
        for item in value:
            if item is None:
                continue
            if not isinstance(item, str):
                item = str(item)
            normalized = item.strip()
            if not normalized:
                continue
            commands.append(normalized)
        return commands

    raise SWEBenchLiteAdapterError(
        f"{field_name} must be a string or list of strings, got {type(value).__name__}"
    )


def _derive_verify_from_test_lists(row: dict[str, Any]) -> list[str]:
    candidates = _parse_json_string_list(
        row.get("FAIL_TO_PASS", row.get("fail_to_pass"))
    )
    if not candidates:
        candidates = _parse_json_string_list(
            row.get("PASS_TO_PASS", row.get("pass_to_pass"))
        )

    commands: list[str] = []
    for item in candidates:
        normalized = str(item).strip()
        if not normalized:
            continue
        if _looks_like_shell_command(normalized):
            commands.append(normalized)
            continue

        converted_unittest_name = _convert_unittest_style_test_name(normalized)
        if converted_unittest_name is not None:
            commands.append(
                f"python -m pytest -q {shlex.quote(converted_unittest_name)}"
            )
            continue

        if _looks_like_pytest_nodeid(normalized):
            commands.append(f"python -m pytest -q {shlex.quote(normalized)}")
        else:
            commands.append(f"python -m pytest -q -k {shlex.quote(normalized)}")
    return commands


def _parse_json_string_list(value: Any) -> list[str]:
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


def _looks_like_shell_command(value: str) -> bool:
    starts = (
        "pytest ",
        "python ",
        "python3 ",
        "tox ",
        "nox ",
        "bash ",
        "./",
        "sh ",
    )
    if value.startswith(starts):
        return True
    return any(token in value for token in (" && ", ";", "|", " > ", " < "))


def _looks_like_pytest_nodeid(value: str) -> bool:
    return "::" in value or value.endswith(".py") or "/" in value


def _convert_unittest_style_test_name(value: str) -> str | None:
    # Example: "test_name (pkg.subpkg.module.TestClass)"
    match = re.match(r"^([^\s(]+)\s+\(([^()]+)\)$", value)
    if not match:
        return None

    test_name = match.group(1).strip()
    location = match.group(2).strip()
    if not test_name or not location:
        return None

    parts = [part for part in location.split(".") if part]
    if len(parts) < 2:
        return None

    class_name = parts[-1]
    module_path = "/".join(parts[:-1]) + ".py"
    return f"{module_path}::{class_name}::{test_name}"


__all__ = [
    "AdaptationSkip",
    "DEFAULT_TIMEOUT_S",
    "SOURCE_NAME",
    "SWEBenchLiteAdapterError",
    "SWETask",
    "SWETaskValidationError",
    "adapt_swebench_lite_row",
    "adapt_swebench_lite_rows",
    "coerce_swe_task",
    "deterministic_train_eval_split",
    "load_task_file",
    "read_jsonl_rows",
    "validate_swe_task",
    "write_tasks_jsonl",
]
