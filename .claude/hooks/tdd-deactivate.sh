#!/bin/bash
# Standalone script to deactivate TDD enforcement.
# Usage: bash .claude/hooks/tdd-deactivate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/tdd-state.sh" deactivate
