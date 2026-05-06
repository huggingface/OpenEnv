# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenCode environment for OpenEnv.

Two layers in this package:

1. **Harness primitive** — :class:`OpenCodeSessionFactory` /
   :class:`OpenCodeSession` / :class:`OpenCodeConfig` /
   :class:`E2BSandboxBackend`. Used in-process to drive one rollout
   inside an E2B sandbox. See ``harness.py``.

2. **Deployable env** — :class:`OpenCodeEnv` (MCP client) talks to the
   FastAPI server at ``server/app.py`` over HTTP. Use this when the
   sandbox + agent live behind an HTTP boundary (e.g. an HF Space).
   See ``client.py`` and ``server/``.
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import OpenCodeEnv
from .config import OpenCodeConfig, Provider
from .harness import OpenCodeSession, OpenCodeSessionFactory
from .models import (
    CommandResult,
    OpenCodeState,
    RolloutResult,
    RolloutTurn,
)
from .sandbox import E2BSandboxBackend, SandboxBackend, SandboxHandle
from .task import OpenCodeTask

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
