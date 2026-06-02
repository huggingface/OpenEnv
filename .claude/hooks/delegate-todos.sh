#!/bin/bash
# PostToolUse hook for TodoWrite: Remind about TDD workflow when TDD is active

# Check if TDD is active
source "$(dirname "$0")/tdd-state.sh"
if ! is_tdd_active; then
    exit 0  # TDD not active, no reminder needed
fi

# Soft reminder about the workflow
cat << 'EOF'

TDD Workflow Reminder:
  For each todo that requires implementation:
  1. /write-tests  ->  create failing tests first
  2. /implement    ->  make tests pass
  3. Mark todo complete

EOF

exit 0
