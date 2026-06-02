# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app for the opencode_env MCP server.

Mirrors the standard OpenEnv pattern (echo_env / repl_env / jupyter_agent)
plus the custom Gradio UI mounted at ``/web`` per the
``customizing-web-ui`` doc.

Usage::

    # Local dev:
    E2B_API_KEY=... uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Docker:
    docker run -p 8000:8000 -e E2B_API_KEY=... opencode-env

    # HF Space: deploys via the root ``Dockerfile``.

The ``ENABLE_WEB_INTERFACE`` env var is set to ``true`` automatically so
the UI is always reachable at ``/web``. Set it to ``false`` to disable.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_env_file() -> None:
    """Lightweight ``.env`` loader (no python-dotenv dep).

    Loads ``../.env`` (env dir's ``.env``) into ``os.environ`` for local
    development convenience. Existing process env vars take precedence so
    HF Space secrets always win.
    """
    candidate = Path(__file__).resolve().parents[1] / ".env"
    if not candidate.exists():
        return
    for raw in candidate.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env_file()


try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import (
        CallToolAction,
        CallToolObservation,
    )

    from .gradio_ui import opencode_gradio_builder
    from .opencode_environment import OpenCodeEnvironment
except ImportError:  # pragma: no cover
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import (
        CallToolAction,
        CallToolObservation,
    )
    from server.gradio_ui import opencode_gradio_builder  # type: ignore
    from server.opencode_environment import OpenCodeEnvironment  # type: ignore


# Always expose the Gradio UI at /web. Set ENABLE_WEB_INTERFACE=false to
# disable (e.g., on HF Spaces where you want the API only).
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")


def _custom_gradio_builder(
    web_manager,
    action_fields,
    metadata,
    is_chat_env,
    title,
    quick_start_md,
):
    """Hand off to ``server.gradio_ui.opencode_gradio_builder``."""
    return opencode_gradio_builder(
        web_manager,
        action_fields,
        metadata,
        is_chat_env,
        title or "opencode_env",
        quick_start_md,
    )


app = create_app(
    OpenCodeEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="opencode_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
    gradio_builder=_custom_gradio_builder,
)


def main() -> None:
    """Entrypoint for ``uv run --project . server`` and direct invocation."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
