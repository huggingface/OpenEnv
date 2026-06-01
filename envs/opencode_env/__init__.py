# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenCode environment for OpenEnv.

Two layers in this package:

1. **Harness primitive** -- :class:`OpenCodeSessionFactory` /
   :class:`OpenCodeSession` / :class:`OpenCodeConfig` /
   :class:`E2BSandboxBackend`. Built on the generic
   :class:`CLIAgentDriver` from ``openenv.core.harness.agents``.

2. **Deployable env** -- :class:`OpenCodeEnv` (MCP client) talks to the
   FastAPI server at ``server/app.py`` over HTTP. Use this when the
   sandbox + OpenCode live behind an HTTP boundary (e.g. an HF Space).
   See ``client.py`` and ``server/``.
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv.core.harness.sandbox import SandboxBackend, SandboxHandle

from .client import OpenCodeEnv
from .config import OpenCodeConfig, Provider
from .harness import OpenCodeSession, OpenCodeSessionFactory
from .models import CommandResult, OpenCodeState, RolloutResult, RolloutTurn
from .task import OpenCodeTask

try:
    from openenv.core.harness.sandbox import E2BSandboxBackend
except ImportError:  # e2b not installed
    E2BSandboxBackend = None  # type: ignore[assignment,misc]

__all__ = [
    # Deployed-env client
    "OpenCodeEnv",
    "CallToolAction",
    "ListToolsAction",
    # HTTP API models
    "CommandResult",
    "OpenCodeState",
    "RolloutResult",
    "RolloutTurn",
    # Harness primitive
    "OpenCodeConfig",
    "OpenCodeSession",
    "OpenCodeSessionFactory",
    "OpenCodeTask",
    "Provider",
    # Sandbox backend
    "E2BSandboxBackend",
    "SandboxBackend",
    "SandboxHandle",
]
