#!/bin/bash
# SessionStart hook: Show context and set mode based on TDD state

echo ""

# Check if we're in a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    exit 0
fi

TOPLEVEL=$(git rev-parse --show-toplevel)

# Source TDD state helpers
source "$(dirname "$0")/tdd-state.sh"

if is_tdd_active; then
    # TDD mode activated via /work-on-issue
    ISSUE=$(get_tdd_issue)
    FEATURE=$(basename "$TOPLEVEL")
    BRANCH=$(git branch --show-current 2>/dev/null)

    echo "==================================================================="
    echo "  TDD MODE ACTIVE  (issue #${ISSUE:-?})"
    echo "==================================================================="
    echo "  Worktree: $FEATURE"
    echo "  Branch: $BRANCH"
    echo ""
    echo "  Direct code edits blocked."
    echo ""
    echo "  Workflow:"
    echo "    /write-tests    ->  create failing tests"
    echo "    /implement      ->  make tests pass"
    echo "    /update-docs    ->  fix stale docs"
    echo "    /simplify       ->  clean up (optional)"
    echo "    /pre-submit-pr  ->  validate before commit"
    echo ""
    echo "  Say \"skip TDD\" to bypass blocking"
    echo "==================================================================="
elif [[ "$TOPLEVEL" == *".worktrees"* ]]; then
    # In a worktree but TDD not activated
    FEATURE=$(basename "$TOPLEVEL")
    BRANCH=$(git branch --show-current 2>/dev/null)

    echo "==================================================================="
    echo "  WORKTREE: $FEATURE"
    echo "==================================================================="
    echo "  Branch: $BRANCH"
    echo ""
    echo "  Direct edits allowed. To enable TDD enforcement:"
    echo "    /work-on-issue #<N>  ->  start TDD workflow"
    echo "==================================================================="
else
    echo "==================================================================="
    echo "  MAIN REPO (Explore Mode)"
    echo "==================================================================="
    echo ""
    echo "  Direct edits allowed. For focused work:"
    echo "    /work-on-issue #42  ->  start TDD workflow"
    echo ""
    echo "  Or manually:"
    echo "    .claude/scripts/worktree-create.sh <name>"
    echo "==================================================================="
fi

echo ""
