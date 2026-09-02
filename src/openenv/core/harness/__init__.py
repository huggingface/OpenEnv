# SPDX-License-Identifier: BSD-3-Clause

"""Harness helpers for training, evaluation, and wrapping external agents.

This package hosts two complementary layers:

1. **Trainer-side rollout API** (``openenv.core.harness.rollout``): a harness
   drives an entire episode in one call (``run_white_box``/``run_black_box``)
   against a resource session. Used by ``openenv collect`` and the training
   tutorials.
2. **Turn-based agentic harness API** (RFC 005): the types describing an
   external harness such as OpenClaw or Claude Code, where each ``step()`` is
   one conversational turn. See
   [`~openenv.core.harness.adapter.AgenticHarnessAdapter`].

Both layers are importable from ``openenv.core.harness``.
"""

from .adapter import (
    AgenticHarnessAdapter,
    HarnessError,
    HarnessNotRunningError,
    HarnessStartupError,
    HarnessTurnTimeoutError,
)
from .config import HarnessConfig, HarnessTransport
from .events import (
    events_to_metadata,
    HarnessClientMessage,
    HarnessEvent,
    HarnessEventType,
    HarnessResponse,
)
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
from .tools import resolve_tool_conflicts

__all__ = [
    # Trainer-side rollout API (openenv.core.harness.rollout)
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
    # Turn-based agentic harness API (RFC 005)
    "AgenticHarnessAdapter",
    "HarnessClientMessage",
    "HarnessConfig",
    "HarnessError",
    "HarnessEvent",
    "HarnessEventType",
    "HarnessNotRunningError",
    "HarnessResponse",
    "HarnessStartupError",
    "HarnessTransport",
    "HarnessTurnTimeoutError",
    "events_to_metadata",
    "resolve_tool_conflicts",
]
