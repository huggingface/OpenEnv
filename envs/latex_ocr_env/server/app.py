# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the LaTeX OCR Environment.

Exposes LatexOCREnvironment over HTTP + WebSocket (Gym-style) endpoints and the
Task API for dataset-backed task discovery.

Usage:
    # Development:
    PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    python -m server.app
"""

import os

try:
    from openenv.core.env_server import create_app

    from ..models import LatexOCRAction, LatexOCRObservation
    from .gradio_ui import latex_ocr_ui_builder
    from .latex_ocr_environment import LatexOCREnvironment
except ImportError:  # standalone (uvicorn server.app:app)
    from models import LatexOCRAction, LatexOCRObservation
    from openenv.core.env_server import create_app
    from server.gradio_ui import latex_ocr_ui_builder
    from server.latex_ocr_environment import LatexOCREnvironment


# The custom "Try it" tab is the documented way to poke this env by hand, and `create_app` only
# mounts /web when this is set. Without it the tab was dark on Docker, on a Space and on the local
# run path in this docstring. `setdefault` so ENABLE_WEB_INTERFACE=false still turns it off.
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

app = create_app(
    LatexOCREnvironment,
    LatexOCRAction,
    LatexOCRObservation,
    env_name="latex_ocr_env",
    max_concurrent_envs=int(os.environ.get("LATEX_OCR_MAX_SESSIONS", "16")),
    gradio_builder=latex_ocr_ui_builder,
    custom_tab_name="Try it",
    custom_tab_primary=True,
)


@app.get("/healthz")
async def health() -> dict[str, str]:
    """Lightweight service health endpoint for orchestration checks."""
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
