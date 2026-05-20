# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coding-agent environment for OpenEnv.

Two layers in this package:

1. **Harness primitive** -- :class:`CodingAgentSessionFactory` /
   :class:`CodingAgentSession` / :class:`CodingAgentConfig` /
   :class:`E2BSandboxBackend`. Built on the generic
   :class:`CLIAgentDriver` from ``openenv.core.harness.agents``.

2. **Deployable env** -- :class:`CodingAgentEnv` (MCP client) talks to the
   FastAPI server at ``server/app.py`` over HTTP. Use this when the
   sandbox + agent live behind an HTTP boundary (e.g. an HF Space).
   See ``client.py`` and ``server/``.
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv.core.harness.sandbox import SandboxBackend, SandboxHandle

from .client import CodingAgentEnv
from .config import CodingAgentConfig, Provider
from .harness import CodingAgentSession, CodingAgentSessionFactory
from .models import CommandResult, CodingAgentState, RolloutResult
from .task import CodingAgentTask

try:
    from openenv.core.harness.sandbox import E2BSandboxBackend
except ImportError:  # e2b not installed
    E2BSandboxBackend = None  # type: ignore[assignment,misc]

__all__ = [
    # Deployed-env client
    "CodingAgentEnv",
    "CallToolAction",
    "ListToolsAction",
    # HTTP API models
    "CommandResult",
    "CodingAgentState",
    "RolloutResult",
    # Harness primitive
    "CodingAgentConfig",
    "CodingAgentSession",
    "CodingAgentSessionFactory",
    "CodingAgentTask",
    "Provider",
    # Sandbox backend
    "E2BSandboxBackend",
    "SandboxBackend",
    "SandboxHandle",
]
