# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the mini_swe_env.

Contains:
  - ``SWEGymTask`` — frozen dataclass mirroring the SWE-Gym HF dataset
    schema (``instance_id``, ``repo``, ``patch``, ``test_patch``, etc.).
  - ``SWETask`` — internal task shape shared across environment, harness,
    and client layers.  ``SWEGymTask.to_swe_task()`` converts between them.
  - ``SWERolloutResult`` / ``SWECommandResult`` / ``SWEState`` — Pydantic
    models for the server's MCP tool.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from openenv.core.env_server.types import State
from pydantic import BaseModel, Field


DEFAULT_TIMEOUT_S = 1800
SWEGYM_SOURCE = "swegym"


# ── SWEGymTask (matches HF dataset schema) ────────────────────────────────


class SWETaskValidationError(ValueError):
    """Raised when a task fails schema validation."""


@dataclass(frozen=True)
class SWEGymTask:
    """One SWE-Gym task, mirroring the HuggingFace dataset columns.

    Fields correspond 1:1 to the ``SWE-Gym/SWE-Gym`` dataset:
      ``instance_id``, ``repo``, ``base_commit``, ``problem_statement``,
      ``version``, ``patch`` (ground-truth), ``test_patch``,
      ``FAIL_TO_PASS``, ``PASS_TO_PASS``, ``hints_text``, ``created_at``.

    The ``timeout_s`` field is not in the dataset; it is set by the loader.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    version: str
    patch: str
    test_patch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    hints_text: str = ""
    created_at: str = ""
    timeout_s: int = DEFAULT_TIMEOUT_S

    # ── Derived helpers ────────────────────────────────────────────────

    @property
    def instance_image(self) -> str:
        """Docker image name for this task (SWE-Gym convention).

        Uses the ``xingyaoww/`` namespace with ``sweb.eval.x86_64.`` prefix.
        Doubles underscores are replaced per SWE-bench convention.
        """
        sanitised = self.instance_id.lower().replace("__", "_1776_")
        return f"xingyaoww/sweb.eval.x86_64.{sanitised}:latest"

    def to_swe_task(self) -> SWETask:
        """Convert to the internal ``SWETask`` used by the harness.

        The ``verify`` list is left empty because Phase 3 uses
        ``swebench.harness.grading`` for reward, not shell commands.
        The ``instruction`` is the ``problem_statement``.
        """
        return SWETask(
            task_id=f"{SWEGYM_SOURCE}::{self.instance_id}",
            source=SWEGYM_SOURCE,
            instance_id=self.instance_id,
            repo=self.repo,
            base_commit=self.base_commit,
            instruction=self.problem_statement,
            setup=[],
            verify=[],  # grading via swebench, not shell commands
            timeout_s=self.timeout_s,
            sandbox_image=self.instance_image,
            metadata={
                "version": self.version,
                "patch": self.patch,
                "test_patch": self.test_patch,
                "FAIL_TO_PASS": list(self.FAIL_TO_PASS),
                "PASS_TO_PASS": list(self.PASS_TO_PASS),
                "hints_text": self.hints_text,
                "created_at": self.created_at,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)


# ── SWETask (internal task shape) ──────────────────────────────────────────


@dataclass(frozen=True)
class SWETask:
    """One SWE task (repo + commit + instruction + verify commands).

    This is the internal task shape shared across the environment,
    harness, and client layers.  It is backend-agnostic: both
    SWE-Gym and any future task sources produce ``SWETask`` instances.
    """

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


# ── Pydantic models (server payloads) ─────────────────────────────────────


class SWECommandResult(BaseModel):
    """Outcome of one shell command in setup or verify."""

    cmd: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


class SWERolloutResult(BaseModel):
    """Full payload returned from one ``run_swe_rollout`` invocation.

    The trainer (or any client) decodes this from the MCP tool result JSON
    and feeds ``reward`` into GRPO.
    """

    # Identifiers
    task_id: str = ""
    instance_id: str = ""
    sandbox_id: str = ""

    # Scalars — binary reward (1.0 resolved, 0.0 not resolved)
    reward: float | None = None
    resolved: bool | None = None
    agent_exit_code: int | None = None
    wall_s: float = 0.0

    # Per-step results
    setup_results: list[SWECommandResult] = Field(default_factory=list)
    verify_results: list[SWECommandResult] = Field(default_factory=list)

    # Filesystem the agent produced (path -> contents, truncated)
    files: dict[str, str] = Field(default_factory=dict)
    files_extra: list[str] = Field(default_factory=list)

    # Diagnostic tails
    agent_log_tail: str = ""

    # Error surfacing
    error: str | None = None


class SWEState(State):
    """Per-session environment state across calls to one SWEEnvironment instance.

    Each HTTP session gets its own env (``SUPPORTS_CONCURRENT_SESSIONS=True``
    on the server class), so this state is per-session.
    """

    rollouts_completed: int = 0
    last_reward: float | None = None
    last_task_id: str | None = None
    last_instance_id: str | None = None
    last_sandbox_id: str | None = None
