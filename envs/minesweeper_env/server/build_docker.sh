#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Script to build the Minesweeper environment Docker image
# Usage: ./build_docker.sh [tag]

set -e

TAG="${1:-latest}"
IMAGE_NAME="minesweeper-env:${TAG}"

echo "🐳 Building Minesweeper Environment Docker Image"
echo "================================================"
echo "Image: $IMAGE_NAME"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to OpenEnv root (4 levels up from server/)
OPENENV_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "📁 OpenEnv root: $OPENENV_ROOT"
echo ""

# Build Minesweeper environment image
echo "⏳ Building..."
docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$OPENENV_ROOT"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Run with:"
    echo "  docker run -p 8000:8000 $IMAGE_NAME"
    echo ""
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi
