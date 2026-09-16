# SPDX-License-Identifier: BSD-3-Clause

"""FastAPI application for the Harbor environment.

Run it directly:

```bash
HARBOR_TASKS_DIR=./tasks uv run --project envs/harbor_env server
```

or through the image built from `server/Dockerfile`.

Configuration (all optional):

| Variable                  | Default            | Meaning                                                    |
| ------------------------- | ------------------ | ---------------------------------------------------------- |
| `HARBOR_TASKS_DIR`        | bundled examples   | Task directory, or `hf://datasets/<org>/<name>`            |
| `HARBOR_MODE`             | `docker`           | Sandbox backend: `docker` or `local`                       |
| `HARBOR_DEFAULT_TASK_ID`  | —                  | Task used when `reset()` names none                        |
| `HARBOR_COMMAND_TIMEOUT_S`| `120`              | Timeout for agent `exec` actions                           |
| `HARBOR_DEFAULT_IMAGE`    | `python:3.12-slim` | Image for self-contained tasks in `docker` mode            |
| `MAX_CONCURRENT_ENVS`     | `8`                | Concurrent WebSocket sessions                              |
| `HARBOR_ALLOW_CONTROL_ACTIONS` | `1`           | Accept `evaluate` / `solve`; set `0` for agent-facing runs  |
"""

import os


try:
    # In-repo imports (when running from OpenEnv repository)
    from harbor_env.models import HarborAction, HarborObservation
    from openenv.core.env_server.http_server import create_app

    from .harbor_env_environment import (
        DEFAULT_COMMAND_TIMEOUT_S,
        DEFAULT_MODE,
        HarborEnvironment,
    )
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from models import HarborAction, HarborObservation
    from openenv.core.env_server.http_server import create_app
    from server.harbor_env_environment import (
        DEFAULT_COMMAND_TIMEOUT_S,
        DEFAULT_MODE,
        HarborEnvironment,
    )


_MODE = (os.getenv("HARBOR_MODE") or DEFAULT_MODE).lower()


def _build_environment() -> HarborEnvironment:
    """Factory called once per session by the server."""
    return HarborEnvironment(
        command_timeout_s=float(
            os.getenv("HARBOR_COMMAND_TIMEOUT_S", DEFAULT_COMMAND_TIMEOUT_S)
        ),
    )


app = create_app(
    _build_environment,
    HarborAction,
    HarborObservation,
    env_name=f"harbor_env ({_MODE} mode)",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for `uv run --project envs/harbor_env server`."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
