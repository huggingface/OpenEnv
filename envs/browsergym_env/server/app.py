"""FastAPI server for the BrowserGym environment."""

import os
from functools import partial

from browsergym_env.models import BrowserGymAction, BrowserGymObservation
from browsergym_env.server.browsergym_environment import BrowserGymEnvironment
from openenv.core.env_server.http_server import create_app

# Get configuration from environment variables
benchmark = os.environ.get("BROWSERGYM_BENCHMARK", "miniwob")
task_name = os.environ.get("BROWSERGYM_TASK_NAME")  # Optional, can be None
headless = os.environ.get("BROWSERGYM_HEADLESS", "true").lower() == "true"
viewport_width = int(os.environ.get("BROWSERGYM_VIEWPORT_WIDTH", "1280"))
viewport_height = int(os.environ.get("BROWSERGYM_VIEWPORT_HEIGHT", "720"))
timeout = float(os.environ.get("BROWSERGYM_TIMEOUT", "10000"))
port = int(os.environ.get("BROWSERGYM_PORT", "8000"))

max_concurrent = int(os.environ.get("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    partial(
        BrowserGymEnvironment,
        benchmark=benchmark,
        task_name=task_name,
        headless=headless,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        timeout=timeout,
    ),
    BrowserGymAction,
    BrowserGymObservation,
    env_name="browsergym_env",
    max_concurrent_envs=max_concurrent,
)


def main():
    """Main entry point for running the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
