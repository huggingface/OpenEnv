# SPDX-License-Identifier: BSD-3-Clause

# Multi-stage build using openenv-base.
# Works for both in-repo builds (local OpenEnv sources) and standalone builds
# (openenv from PyPI); `openenv build` sets BUILD_MODE accordingly.

ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is needed to install dependencies from VCS refs; curl bootstraps uv when
# the base image does not already carry it.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=harbor_env

COPY . /app/env
WORKDIR /app/env

# uv may be absent from locally built base images.
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Harbor verifiers commonly shell out to git, bash and coreutils; `timeout`
# bounds commands in the local backend's subprocesses, and curl backs HEALTHCHECK.
RUN apt-get update && \
    apt-get install -y --no-install-recommends bash coreutils curl git && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:${PYTHONPATH:-}"
ENV ENABLE_WEB_INTERFACE=true

# Point at your own task set with `-e HARBOR_TASKS_DIR=...`; the default serves
# the bundled example task so the image is useful out of the box.
# The library default is `docker`. Inside this image there is no daemon unless
# the host socket is mounted, so the image ships `local` and the docs show the
# `-e HARBOR_MODE=docker -v /var/run/docker.sock:...` invocation for real task sets.
ENV HARBOR_MODE=local

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
