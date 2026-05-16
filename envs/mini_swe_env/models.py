# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic models for the mini_swe_env server.

The server exposes a single MCP tool ``run_swe_rollout`` that takes an
SWE task, runs one agent rollout inside a sandbox, and returns a
JSON-serialized :class:`SWERolloutResult`.
"""

from __future__ import annotations

from openenv.core.env_server.types import State
from pydantic import BaseModel, Field


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
