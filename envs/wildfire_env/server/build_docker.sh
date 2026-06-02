#!/bin/bash
set -e

TAG="${1:-latest}"
IMAGE_NAME="wildfire-env:${TAG}"

echo "🔥 Building Wildfire Environment Docker Image"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OPENENV_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$OPENENV_ROOT"
