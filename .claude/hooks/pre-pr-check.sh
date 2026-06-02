#!/bin/bash
# PreToolUse hook for Bash: Block PR creation if branch is stale
#
# Intercepts `gh pr create` and checks branch freshness against the
# base branch. Unlike git hooks, this cannot be bypassed with --no-verify.

# Read JSON from stdin
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

# Only check gh pr create commands
if [[ "$COMMAND" != *"gh pr create"* ]]; then
    exit 0
fi

# Determine base branch (default: main)
BASE="main"
if echo "$COMMAND" | grep -qoP '(?<=--base\s)\S+'; then
    BASE=$(echo "$COMMAND" | grep -oP '(?<=--base\s)\S+')
fi

# Fetch latest base and check freshness
git fetch origin "$BASE" --quiet 2>/dev/null || true
BEHIND=$(git rev-list --count HEAD.."origin/$BASE" 2>/dev/null || echo "?")

if [[ "$BEHIND" != "0" && "$BEHIND" != "?" ]]; then
    cat >&2 << EOF

===================================================================
  PR BLOCKED: Branch is $BEHIND commit(s) behind $BASE
===================================================================

  Your PR will show "out of date with base branch" on GitHub.

  Fix with:
    git fetch origin $BASE
    git rebase origin/$BASE
    git push --force-with-lease

  Then retry gh pr create.

===================================================================

EOF
    exit 2
fi

# Check we're not on main/master
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [[ "$BRANCH" == "main" || "$BRANCH" == "master" ]]; then
    cat >&2 << EOF

===================================================================
  PR BLOCKED: Cannot create PR from $BRANCH
===================================================================

  Create a feature branch first:
    git checkout -b <branch-name>
    git push -u origin <branch-name>

===================================================================

EOF
    exit 2
fi

exit 0
