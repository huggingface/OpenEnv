# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Sophistry-Bench Sprint Environment."""

from openenv.core.env_server import create_app

try:
    # Installed-package context (e.g. import sophistry_bench_sprint_env.server.app)
    from ..models import AdvocacyAction, AdvocacyObservation
    from .sophistry_bench_sprint_environment import SophistryBenchSprintEnvironment
except ImportError:
    # Container runtime context (uvicorn server.app:app, PYTHONPATH=/app/env)
    from sophistry_bench_sprint_env.models import AdvocacyAction, AdvocacyObservation
    from sophistry_bench_sprint_env.server.sophistry_bench_sprint_environment import (
        SophistryBenchSprintEnvironment,
    )

app = create_app(
    SophistryBenchSprintEnvironment,
    AdvocacyAction,
    AdvocacyObservation,
    env_name="sophistry_bench_sprint_env",
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
