# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build for dm_control environment
# Uses pip for package installation

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies including OpenGL for MuJoCo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglx-mesa0 \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglfw3 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Install dependencies using pip
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

# Final runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (OpenGL for MuJoCo rendering, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglx-mesa0 \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the environment code
COPY . /app/env

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env"

# Set MuJoCo to use OSMesa for headless rendering
ENV MUJOCO_GL="osmesa"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
# Use exec to replace the shell with uvicorn so it receives SIGINT/SIGTERM directly
CMD ["sh", "-c", "cd /app/env && exec uvicorn server.app:app --host 0.0.0.0 --port 8000"]
