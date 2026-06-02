#!/bin/bash
# PreToolUse hook for Edit/Write: Block direct code edits in TDD mode
#
# Design: Only block when TDD is activated via /work-on-issue.
# Worktrees without TDD marker and the main repo allow direct edits.

# Check if TDD is active (marker file from /work-on-issue)
source "$(dirname "$0")/tdd-state.sh"
if ! is_tdd_active; then
    exit 0  # TDD not active, allow all edits
fi

# Read JSON from stdin (hook input format)
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)

# If no file path or jq failed, allow
if [[ -z "$FILE_PATH" ]]; then
    exit 0
fi

# Only check Python implementation files
if [[ "$FILE_PATH" != *.py ]]; then
    exit 0  # Not a Python file, allow
fi

# Allow test files
if [[ "$FILE_PATH" == *test* ]] || [[ "$FILE_PATH" == */tests/* ]]; then
    exit 0  # Test file, allow (tester persona can write these)
fi

# Allow non-src files (scripts, configs, etc.)
if [[ "$FILE_PATH" != */src/* ]] && [[ "$FILE_PATH" != */envs/* ]]; then
    exit 0
fi

# Block with helpful message
ISSUE=$(get_tdd_issue)
cat >&2 << EOF

===================================================================
  TDD MODE: Direct code edit blocked  (issue #${ISSUE:-?})
===================================================================

In TDD mode, use the TDD workflow:

  1. /write-tests  ->  tester writes failing tests
  2. /implement    ->  implementer makes tests pass

To bypass this check, say "skip TDD" in your message.

===================================================================

EOF

exit 2
