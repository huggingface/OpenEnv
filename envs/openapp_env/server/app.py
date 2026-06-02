# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the OpenApp Environment.

This module creates an HTTP server that exposes the OpenAppEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server
"""

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app

    from ..models import OpenAppAction, OpenAppObservation
    from .openapp_environment import OpenAppEnvironment
except ImportError:
    try:
        from openapp_env.models import OpenAppAction, OpenAppObservation
        from openapp_env.server.openapp_environment import OpenAppEnvironment

        # Standalone imports when openenv is available.
        from openenv.core.env_server.http_server import create_app
    except ImportError:
        from openapp_env.models import OpenAppAction, OpenAppObservation
        from openapp_env.server.openapp_environment import OpenAppEnvironment

        # Backward-compatible standalone imports when only openenv_core is available.
        from openenv_core.env_server.http_server import create_app

# Create the app with web interface and README integration
# Pass the class (factory) instead of an instance for WebSocket session support
# Each client gets its own environment instance. The environment reads
# OPENAPPS_URL from environment variables in __init__.
app = create_app(
    OpenAppEnvironment, OpenAppAction, OpenAppObservation, env_name="openapp_env"
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.openapp_env.server.app
        openenv serve openapp_env

    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
