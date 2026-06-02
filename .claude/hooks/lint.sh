#!/bin/bash
# Lint check for OpenEnv
# Replicates the exact arc f pipeline from fbsource:
#   1. usort format — sort imports (matches arc f's usort pass)
#   2. ruff format  — code formatting, line-length 88 (matches arc f's ruff-api pass)
#   3. ruff check   — lint rules (E, F, W)
#
# usort is scoped to src/ and tests/ only. envs/ uses ruff format only
# because standalone usort and pyfmt's usort disagree on import ordering
# inside try/except blocks in some env files.

set -e

# Check for required tools
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed or not in PATH"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "=== Running import sort + format check ==="
# Run the same pipeline as arc f: usort then ruff format.
# If any file changes, the code wasn't properly formatted.
uv run usort format src/ tests/ >/dev/null 2>&1
uv run ruff format src/ tests/ envs/ >/dev/null 2>&1

# Check if any files were modified (means they weren't formatted before)
CHANGED=$(git diff --name-only -- '*.py' 2>/dev/null || true)
if [ -n "$CHANGED" ]; then
    echo "ERROR: The following files need formatting:"
    echo "$CHANGED"
    echo ""
    echo "Run: uv run usort format src/ tests/ && uv run ruff format src/ tests/ envs/"
    # Undo the formatting so the working tree stays as-is
    git checkout -- $CHANGED 2>/dev/null || true
    exit 1
fi
echo "Import sort + format check passed!"

echo "=== Running lint rules check ==="
uv run ruff check src/ tests/

echo "=== Lint check passed ==="
