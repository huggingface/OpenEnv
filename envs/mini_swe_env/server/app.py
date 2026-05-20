# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Mini SWE Environment.

Exposes the SWEEnvironment over HTTP and WebSocket endpoints,
compatible with MCPToolClient.

Usage:
    # Development:
    PYTHONPATH=src:envs uvicorn mini_swe_env.server.app:app --reload --port 8000

    # Production:
    uvicorn mini_swe_env.server.app:app --host 0.0.0.0 --port 8000
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .swe_environment import SWEEnvironment
except ImportError:  # pragma: no cover
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.swe_environment import SWEEnvironment  # type: ignore


max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "4"))

app = create_app(
    SWEEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="mini_swe_env",
    max_concurrent_envs=max_concurrent,
)


def main() -> None:
    """Entry point for ``uv run --project . server`` and direct invocation."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
