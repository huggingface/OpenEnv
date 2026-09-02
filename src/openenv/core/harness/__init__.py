# SPDX-License-Identifier: BSD-3-Clause

"""Harness helpers for training and evaluation.

The trainer-side rollout API now lives in ``openenv.core.harness.rollout``:
a harness drives an entire episode in one ``run_white_box``/``run_black_box``
call against a resource session. It is re-exported here unchanged, so
``from openenv.core.harness import ...`` keeps working exactly as before.

Splitting the package this way makes room for the RFC 005 turn-based agentic
harness layer to land alongside it in sibling modules, rather than growing a
single monolithic ``__init__``.
"""

from .rollout import (  # noqa: F401  (_resolve_env_reward: private back-compat re-export)
    _resolve_env_reward,
    build_harness_rollout_func,
    CLIHarnessAdapter,
    HarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    MCPHarnessAdapter,
    Message,
    ModelStep,
    ModelStepResult,
    RESERVED_TOOL_NAMES,
    ResourceSession,
    ResourceSessionFactory,
    RolloutEvent,
    SessionMCPBridge,
    StepEnvSessionAdapter,
    ToolResult,
    ToolTraceEntry,
    VerifyResult,
)

__all__ = [
    "CLIHarnessAdapter",
    "HarnessAdapter",
    "HarnessRolloutResult",
    "HarnessRunLimits",
    "MCPHarnessAdapter",
    "Message",
    "ModelStep",
    "ModelStepResult",
    "RESERVED_TOOL_NAMES",
    "ResourceSession",
    "ResourceSessionFactory",
    "RolloutEvent",
    "SessionMCPBridge",
    "StepEnvSessionAdapter",
    "ToolResult",
    "ToolTraceEntry",
    "VerifyResult",
    "build_harness_rollout_func",
]
