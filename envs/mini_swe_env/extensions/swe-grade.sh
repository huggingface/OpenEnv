#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# swe-grade.sh — Run swebench-style evaluation in the sandbox.
#
# This script is deployed into the sandbox by SWESessionFactory alongside
# the swe-answer.ts Pi extension.  When the agent calls the `answer` tool,
# the extension executes this script and parses stdout for the result.
#
# The script:
#   1. Applies the test_patch (new/changed tests for this task).
#   2. Runs the test command for the repo/version.
#   3. Reverts the test_patch so the working tree is clean.
#   4. Prints the eval exit code so the extension can parse it.
#
# NOTE: This script does NOT write reward.txt.  The authoritative reward
# is computed host-side by SWESession.verify() using swebench grading.
# The output here is only used by the answer extension as a fast feedback
# signal for the agent ("Resolved: true/false").
#
# Environment variables (set by the harness):
#   SWE_INSTANCE_ID   — e.g. "django__django-11099"
#   SWE_TESTBED       — working directory, e.g. "/testbed"
#   SWE_TEST_PATCH    — path to the test patch file
#   SWE_EVAL_SCRIPT   — path to the swebench eval script
#   SWE_LOG_FILE      — where to write the eval log

set -uo pipefail

TESTBED="${SWE_TESTBED:-/testbed}"
TEST_PATCH="${SWE_TEST_PATCH:-/home/user/swe_test.patch}"
EVAL_SCRIPT="${SWE_EVAL_SCRIPT:-/home/user/swe_eval.sh}"
LOG_FILE="${SWE_LOG_FILE:-/home/user/logs/verifier/eval.log}"

mkdir -p "$(dirname "$LOG_FILE")"

cd "$TESTBED" || { echo "ERROR: cannot cd to $TESTBED"; exit 1; }

# --- Step 1: Apply test patch (if present) ---
if [ -f "$TEST_PATCH" ] && [ -s "$TEST_PATCH" ]; then
    echo ">>>>> Applying test patch"
    git apply --allow-empty "$TEST_PATCH" 2>&1 || true
fi

# --- Step 2: Run eval script ---
EVAL_EXIT=0
if [ -f "$EVAL_SCRIPT" ] && [ -s "$EVAL_SCRIPT" ]; then
    echo ">>>>> Running eval script"
    bash "$EVAL_SCRIPT" > "$LOG_FILE" 2>&1 || EVAL_EXIT=$?
else
    echo ">>>>> No eval script found, skipping"
    echo "No eval script" > "$LOG_FILE"
fi

# --- Step 3: Revert test patch ---
if [ -f "$TEST_PATCH" ] && [ -s "$TEST_PATCH" ]; then
    echo ">>>>> Reverting test patch"
    git apply --allow-empty -R "$TEST_PATCH" 2>&1 || true
fi

# --- Step 4: Print result for the answer extension to parse ---
if [ "$EVAL_EXIT" -eq 0 ]; then
    echo "GRADE: resolved=true eval_exit=0"
else
    echo "GRADE: resolved=false eval_exit=$EVAL_EXIT"
fi

echo ">>>>> Grade complete (eval_exit=$EVAL_EXIT)"
