#!/bin/bash
# Harbor oracle agent entry point: leave the working directory in the solved state.
#
# Harbor copies this whole directory to /solution and runs it, so the reference
# implementation sitting next to this script is what we install.
set -euo pipefail

WORKDIR="${HARBOR_WORKDIR:-$(pwd)}"
SOLUTION_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$SOLUTION_DIR/stats.py" "$WORKDIR/stats.py"
echo "installed the reference stats.py into $WORKDIR"
