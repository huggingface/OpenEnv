# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pi environment for OpenEnv.

Two layers in this package:

1. **Harness primitive** — :class:`PiSessionFactory` / :class:`PiSession` /
   :class:`PiConfig` / :class:`PiTask`. Used in-process to drive one rollout
   inside a sandbox. See ``harness.py``.

2. **Deployable env** — :class:`PiEnv` (MCP client) talks to the FastAPI
   server at ``server/app.py`` over HTTP. Use this when the sandbox + agent
   live behind an HTTP boundary (e.g. an HF Space). See ``client.py`` and
   ``server/``.

The sandbox backend and interception proxy are shared with ``opencode_env``
(to be consolidated into ``openenv.core``).
"""

import warnings

from opencode_env.sandbox import (
    HFSandboxBackend,
    SandboxBackend,
    SandboxHandle,
)

from .client import PiEnv
from .config import PiConfig
from .harness import PiSession, PiSessionFactory
from .models import (
    CommandResult,
    PiState,
    RolloutResult,
    RolloutTurn,
)
from .task import PiTask

warnings.warn(
    "pi_env is deprecated and will be removed in OpenEnv 0.9.0. "
    "Run pi through harbor_env instead: "
    "`HarborSessionFactory(server_url, harness='pi')` against "
    "`openenv harbor serve`. "
    "See https://huggingface.co/docs/openenv/environments/harbor",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    # Deployed-env client
    "PiEnv",
    # HTTP API models
    "CommandResult",
    "PiState",
    "RolloutResult",
    "RolloutTurn",
    # Harness primitive
    "PiConfig",
    "PiSession",
    "PiSessionFactory",
    "PiTask",
    # Sandbox backend (shared with opencode_env)
    "HFSandboxBackend",
    "SandboxBackend",
    "SandboxHandle",
]
