#!/bin/bash
# Harbor verifier entry point.
#
# Contract: write the reward into the verifier log directory — `reward.json` for a
# graded score, `reward.txt` for a single number. The exit code is NOT the verdict.
#
# Portability: prefer the $HARBOR_* variables when they are set, and fall back to
# Harbor's absolute container paths otherwise. That keeps one script working under
# `harbor run` (absolute paths inside a container) and under OpenEnv's local
# backend (a per-episode directory tree).
set -uo pipefail

LOGS_DIR="${HARBOR_LOGS_DIR:-/logs/verifier}"
TESTS_DIR="${HARBOR_TESTS_DIR:-/tests}"
WORKDIR="${HARBOR_WORKDIR:-$(pwd)}"

mkdir -p "$LOGS_DIR"

PYTHON="$(command -v python3 || command -v python)"
if [ -z "$PYTHON" ]; then
    echo "no python interpreter available" >&2
    exit 0
fi

HARBOR_LOGS_DIR="$LOGS_DIR" HARBOR_WORKDIR="$WORKDIR" \
    "$PYTHON" "$TESTS_DIR/grade.py" 2>&1 | tee "$LOGS_DIR/verifier.log"

exit 0
