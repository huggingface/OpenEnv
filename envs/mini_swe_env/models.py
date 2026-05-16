# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the mini_swe_env.

Contains:
  - ``SWETask`` — frozen dataclass representing one SWE task (repo, commit,
    instruction, etc.).  Shared by the environment server, harness, and
    client layers.
  - ``SWERolloutResult`` / ``SWECommandResult`` / ``SWEState`` — Pydantic
    models for the server's MCP tool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from openenv.core.env_server.types import State
from pydantic import BaseModel, Field


DEFAULT_TIMEOUT_S = 1800


class SWETaskValidationError(ValueError):
    """Raised when a task fails schema validation."""


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

    # Scalars
    reward: float | None = None
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
