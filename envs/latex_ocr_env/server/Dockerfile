# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base. Works for in-repo and standalone builds.
#
# Build from the env directory:
#   docker build -t latex-ocr-env:latest -f server/Dockerfile .
# Run:
#   docker run -p 8000:8000 latex-ocr-env:latest

ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app
ARG BUILD_MODE=in-repo
COPY . /app/env
WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

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

FROM ${BASE_IMAGE}
WORKDIR /app
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Pre-caching the dataset at build time is optional; by default it downloads on
# first request. Set LATEX_OCR_SPLITS=test to limit to the small split.

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "latex_ocr_env.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "120", "--ws-ping-timeout", "600"]
