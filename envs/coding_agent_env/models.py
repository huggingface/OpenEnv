# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic models for the deployed coding_agent_env HTTP server.

The server exposes a single MCP tool ``run_rollout`` that takes a Task
(instruction + setup commands + verify commands) plus an LLM endpoint
config, runs one coding-agent rollout end-to-end inside an E2B sandbox, and
returns a :class:`RolloutResult` JSON.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import State
from pydantic import BaseModel, Field


class RolloutTurn(BaseModel):
    """One intercepted LLM turn shape (trainer-owned in interception_gate mode)."""

    turn: int
    finish_reason: str | None = None
    completion_tokens: list[str] = Field(default_factory=list)
    completion_token_ids: list[int] = Field(default_factory=list)
    per_token_logps: list[float] = Field(default_factory=list)
    latency_s: float = 0.0
    timestamp: float = 0.0
    upstream_status: int | None = None
    upstream_error: dict[str, Any] | None = None


class CommandResult(BaseModel):
    """Outcome of one bash command in setup/verify.

    When ``exit_code`` is ``None``, the command ran during sandbox bootstrap
    and its individual exit code was not captured (bootstrap succeeds or fails
    atomically).
    """

    cmd: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


class RolloutResult(BaseModel):
    """Full payload returned from one ``run_rollout`` invocation."""

    # Identifiers
    task_id: str = ""
    sandbox_id: str = ""

    # Scalars
    reward: float | None = None
    agent_exit_code: int | None = None
    wall_s: float = 0.0
    mode: str = "black_box"

    # Per-step results
    setup_results: list[CommandResult] = Field(default_factory=list)
    verify_results: list[CommandResult] = Field(default_factory=list)

    # Per-turn LLM trajectory placeholder. Capture is trainer-owned in
    # interception_gate mode; environment currently leaves this empty.
    proxy_turns: list[RolloutTurn] = Field(default_factory=list)

    # Filesystem the agent produced (path -> contents, truncated)
    files: dict[str, str] = Field(default_factory=dict)
    files_extra: list[str] = Field(default_factory=list)

    # Diagnostic tails
    agent_log_tail: str = ""
    proxy_log_tail: str = ""

    # Error surfacing
    error: str | None = None


class CodingAgentState(State):
    """Per-session env state across calls to one CodingAgentEnvironment instance.

    Each HTTP session gets its own env (``SUPPORTS_CONCURRENT_SESSIONS=True``
    on the server class), so this state is per-session.
    """

    rollouts_completed: int = 0
    last_reward: float | None = None
    last_task_id: str | None = None
    last_sandbox_id: str | None = None
