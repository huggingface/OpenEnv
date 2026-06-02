#!/bin/bash
# Verification script for chess_env
# Run from repository root: bash envs/chess_env/scripts/verify.sh

set -e

echo "=== Chess Environment Verification ==="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓ $1${NC}"
}

check_fail() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# 1. Run unit tests
echo "1. Running unit tests..."
if PYTHONPATH=src:. uv run pytest tests/envs/test_chess_environment.py -v; then
    check_pass "Unit tests passed"
else
    check_fail "Unit tests failed"
fi
echo

# 2. Run linting
echo "2. Running linting..."
if uv run ruff check envs/chess_env/ tests/envs/test_chess_environment.py; then
    check_pass "Linting passed"
else
    check_fail "Linting failed"
fi
echo

# 3. Check formatting
echo "3. Checking formatting..."
if uv run ruff format envs/chess_env/ tests/envs/test_chess_environment.py --check; then
    check_pass "Formatting passed"
else
    check_fail "Formatting failed"
fi
echo

# 4. Verify imports
echo "4. Verifying module imports..."
if PYTHONPATH=src:. uv run python -c "from envs.chess_env import ChessEnv, ChessAction, ChessObservation, ChessState; print('  Imports: ChessEnv, ChessAction, ChessObservation, ChessState')"; then
    check_pass "Module imports successful"
else
    check_fail "Module imports failed"
fi
echo

echo "=== All checks passed! ==="
