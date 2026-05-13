# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app for coding_tools_env."""

from __future__ import annotations

import os
from pathlib import Path

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

try:
    from .coding_tools_env_environment import CodingToolsEnvironment
    from .gradio_ui import coding_tools_ui_builder
except ImportError:  # pragma: no cover
    from server.coding_tools_env_environment import CodingToolsEnvironment  # type: ignore
    from server.gradio_ui import coding_tools_ui_builder  # type: ignore


def _load_env_file() -> None:
    candidate = Path(__file__).resolve().parents[1] / ".env"
    if not candidate.exists():
        return
    for raw in candidate.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

app = create_app(
    CodingToolsEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="coding_tools_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
    gradio_builder=coding_tools_ui_builder,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
