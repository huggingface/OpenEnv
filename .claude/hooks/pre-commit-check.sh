#!/bin/bash
# PreToolUse hook for Bash: Warn on git commit without /pre-submit-pr

# Read JSON from stdin
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

# Only check git commit commands
if [[ "$COMMAND" != *"git commit"* ]]; then
    exit 0
fi

# Only warn when TDD is active
source "$(dirname "$0")/tdd-state.sh"
if ! is_tdd_active; then
    exit 0  # TDD not active, just allow
fi

# Soft warning - don't block, just remind
cat >&2 << 'EOF'

===================================================================
  REMINDER: Consider running /pre-submit-pr before committing
===================================================================

This ensures:
- Lint check passes
- Tests pass
- No debug code left in
- Alignment with principles

Proceeding with commit...

===================================================================

EOF

exit 0
