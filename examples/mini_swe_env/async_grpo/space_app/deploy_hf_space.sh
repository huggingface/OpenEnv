#!/bin/bash
# deploy_hf_space.sh — Deploy SWE Async GRPO training to HF Space + HF Sandbox
#
# This script:
#   1. Prepares the Space directory (minimal repo subset)
#   2. Creates/updates the HF Space
#   3. Configures secrets and environment variables
#   4. Sets hardware (a10g-largex2)
#   5. Pushes code and triggers build
#   6. Monitors build/startup
#
# Prerequisites:
#   - hf CLI installed and authenticated (hf auth login)
#   - HF_TOKEN set or in ~/.cache/huggingface/token
#   - Python with huggingface_hub installed in repo .venv (preferred), or uv/python3 available
#
# Usage:
#   bash examples/mini_swe_env/async_grpo/space_app/deploy_hf_space.sh [OPTIONS]
#
# Options:
#   --space-id OWNER/NAME     Space ID (default: $HF_SPACE_ID or rycerzes/swe-async-grpo-train)
#   --model MODEL_ID          Model to train (default: Qwen/Qwen3-0.6B)
#   --max-tasks N             Number of SWE tasks (default: 5)
#   --max-steps N             Training steps (default: 10)
#   --max-turns N             Max agent turns per rollout (default: 30)
#   --hardware HW             Hardware tier (default: a10g-largex2)
#   --sandbox-backend BACKEND Backend for agent (default: hf)
#   --skip-build              Only configure, don't push code
#   --monitor                 Monitor logs after deploy
#   --pause                   Pause the Space (stop billing)
#   --resume                  Resume a paused Space
#
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

SPACE_ID="${HF_SPACE_ID:-rycerzes/swe-async-grpo-train}"
MODEL="${SWE_MODEL:-Qwen/Qwen3-1.7B}"
MAX_TASKS=5
MAX_STEPS=10
MAX_TURNS=30
HARDWARE="a10g-largex2"
SANDBOX_BACKEND="hf"
SKIP_BUILD=false
MONITOR=false
PAUSE=false
RESUME=false

# ── Tooling prerequisites ───────────────────────────────────────────────
# Prefer repo-local venv Python, then uv, then system python3.
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_CMD=("$REPO_ROOT/.venv/bin/python")
elif [ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
    PYTHON_CMD=("$REPO_ROOT/.venv/Scripts/python.exe")
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=("$(command -v python3)")
else
    echo "ERROR: Python interpreter not found."
    echo "Expected one of:"
    echo "  - $REPO_ROOT/.venv/bin/python"
    echo "  - $REPO_ROOT/.venv/Scripts/python.exe"
    echo "  - uv run python"
    echo "  - python3"
    exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: hf CLI not found. Install huggingface_hub CLI and run 'hf auth login'."
    exit 1
fi

# Ensure Python subprocesses use UTF-8 output (avoids cp1252 emoji crashes on Windows).
export PYTHONUTF8=1

# ── Parse args ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --space-id) SPACE_ID="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --max-tasks) MAX_TASKS="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --max-turns) MAX_TURNS="$2"; shift 2 ;;
        --hardware) HARDWARE="$2"; shift 2 ;;
        --sandbox-backend) SANDBOX_BACKEND="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --monitor) MONITOR=true; shift ;;
        --pause) PAUSE=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Resolve HF_TOKEN ────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    if [ -f "$HOME/.cache/huggingface/token" ]; then
        HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    else
        echo "ERROR: HF_TOKEN not set and not found in ~/.cache/huggingface/token"
        echo "Run: hf auth login"
        exit 1
    fi
fi
export HF_TOKEN

# ── Pause/Resume shortcuts ──────────────────────────────────────────────
if [ "$PAUSE" = true ]; then
    echo "⏸  Pausing Space $SPACE_ID..."
    "${PYTHON_CMD[@]}" -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.pause_space('$SPACE_ID')
print('Space paused.')
"
    exit 0
fi

if [ "$RESUME" = true ]; then
    echo "▶  Resuming Space $SPACE_ID..."
    "${PYTHON_CMD[@]}" -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.restart_space('$SPACE_ID', factory_reboot=True)
print('Space restarted.')
"
    exit 0
fi

# ── Print config ────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   SWE Async GRPO — HF Space Deployment                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  %-54s  ║\n" "Space:    $SPACE_ID"
printf "║  %-54s  ║\n" "Model:    $MODEL"
printf "║  %-54s  ║\n" "Hardware: $HARDWARE"
printf "║  %-54s  ║\n" "Tasks:    $MAX_TASKS"
printf "║  %-54s  ║\n" "Steps:    $MAX_STEPS"
printf "║  %-54s  ║\n" "Turns:    $MAX_TURNS"
printf "║  %-54s  ║\n" "Sandbox:  $SANDBOX_BACKEND"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Create Space (idempotent) ──────────────────────────────────
echo "[1/6] Creating Space (if needed)..."
hf repo create "$SPACE_ID" --repo-type space --space-sdk docker --no-private --exist-ok >/dev/null
echo "  ✓ Space exists: https://huggingface.co/spaces/$SPACE_ID"

# ── Step 2: Configure secrets ──────────────────────────────────────────
echo "[2/6] Configuring secrets and variables..."
"${PYTHON_CMD[@]}" << PYEOF
from huggingface_hub import HfApi
import os, secrets

api = HfApi(token=os.environ['HF_TOKEN'])
space_id = "$SPACE_ID"
checkpoint_repo = f"{space_id}-checkpoints"

# Create/ensure checkpoint bucket repo for restart-safe training state.
try:
    api.create_repo(repo_id=checkpoint_repo, repo_type="model", private=True, exist_ok=True)
    print(f"  ✓ Checkpoint repo: {checkpoint_repo}")
except Exception as e:
    print(f"  ⚠ Checkpoint repo {checkpoint_repo}: {e}")

# Secrets (sensitive)
space_secrets = {
    "HF_TOKEN": os.environ["HF_TOKEN"],
    "INTERCEPTION_AUTH_TOKEN": secrets.token_urlsafe(32),
    "SWE_MODEL": "$MODEL",
    "VLLM_API_KEY": "token",
}
for key, value in space_secrets.items():
    try:
        api.add_space_secret(space_id, key, value)
        print(f"  ✓ Secret: {key}")
    except Exception as e:
        print(f"  ⚠ Secret {key}: {e}")

# Variables (non-sensitive)
variables = {
    "MAX_MODEL_LEN": "4096",
    "GPU_MEMORY_UTILIZATION": "0.70",
    "VLLM_GPU": "0",
    "TRAINER_GPU": "1",
    "INTERCEPTION_HOST": "0.0.0.0",
    "INTERCEPTION_PORT": "7860",
    "TRL_EXPERIMENTAL_SILENCE": "1",
    "TRACKIO_SPACE_ID": "rycerzes/swe-grpo-dashboard",
    "SWE_CHECKPOINT_TO_HUB": "1",
    "SWE_HUB_MODEL_ID": checkpoint_repo,
    "SWE_RESUME_FROM_CHECKPOINT": "auto",
    "SWE_CHECKPOINT_SAVE_STEPS": "2",
    "SWE_CHECKPOINT_SAVE_TOTAL_LIMIT": "2",
    "SWE_HUB_PRIVATE_REPO": "1",
}
for key, value in variables.items():
    try:
        api.add_space_variable(space_id, key, value)
        print(f"  ✓ Var: {key}={value}")
    except Exception as e:
        print(f"  ⚠ Var {key}: {e}")
PYEOF

# ── Step 3: Set hardware ───────────────────────────────────────────────
echo "[3/6] Setting hardware to $HARDWARE..."
"${PYTHON_CMD[@]}" -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.request_space_hardware('$SPACE_ID', '$HARDWARE')
print('  ✓ Hardware: $HARDWARE')
"

# ── Step 4: Prepare staging directory ──────────────────────────────────
if [ "$SKIP_BUILD" = true ]; then
    echo "[4/6] Skipping build (--skip-build)"
    echo "[5/6] Skipping upload"
else
    echo "[4/6] Preparing Space directory..."
    STAGE_DIR=$(mktemp -d)
    trap "rm -rf $STAGE_DIR" EXIT

    cd "$REPO_ROOT"

    # Copy essential files only (Python-based to avoid rsync shell/path quirks on Windows).
    export REPO_ROOT STAGE_DIR
    "${PYTHON_CMD[@]}" << 'PYEOF'
import fnmatch
import os
import shutil
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
stage = Path(os.environ["STAGE_DIR"])


def _ignore(patterns: list[str]):
    def _inner(_dir: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            for pat in patterns:
                if fnmatch.fnmatch(name, pat):
                    ignored.add(name)
                    break
        return ignored

    return _inner


def copy_tree(src_rel: str, dst_rel: str, patterns: list[str]) -> None:
    src = repo / src_rel
    dst = stage / dst_rel
    if not src.exists():
        raise FileNotFoundError(f"Missing source path: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=_ignore(patterns))


common_ignores = [
    "__pycache__",
    "*.pyc",
    ".venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "*.egg-info",
    "uv.lock",
]

copy_tree("src", "src", common_ignores)
copy_tree("envs/mini_swe_env", "envs/mini_swe_env", common_ignores)
copy_tree("examples/mini_swe_env", "examples/mini_swe_env", ["__pycache__", "*.pyc"])
PYEOF

    cp pyproject.toml "$STAGE_DIR/"
    cp LICENSE "$STAGE_DIR/"
    cp .gitignore "$STAGE_DIR/"

    # Space-specific files at root
    cp examples/mini_swe_env/async_grpo/space_app/Dockerfile "$STAGE_DIR/Dockerfile"
    cp examples/mini_swe_env/async_grpo/space_app/start.sh "$STAGE_DIR/start.sh"

    # Space README (metadata)
    cat > "$STAGE_DIR/README.md" << EOF
---
title: SWE Async GRPO Training
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: $HARDWARE
startup_duration_timeout: 30m
preload_from_hub:
  - $MODEL
---

# SWE Async GRPO Training

Model: \`$MODEL\` | Tasks: $MAX_TASKS | Steps: $MAX_STEPS | Backend: $SANDBOX_BACKEND
EOF

    # Patch start.sh to pass our training args
    cat >> "$STAGE_DIR/start.sh" << EOF

# Auto-generated training args from deploy script
# --sandbox-backend $SANDBOX_BACKEND --max-tasks $MAX_TASKS --max-steps $MAX_STEPS --max-turns $MAX_TURNS
EOF
    # Append deploy-time args to the trainer exec line (interpreter-agnostic).
    sed -i "s|\(exec .*examples/mini_swe_env/train_swe_async_grpo.py\)|\1 --sandbox-backend $SANDBOX_BACKEND --max-tasks $MAX_TASKS --max-steps $MAX_STEPS --max-turns $MAX_TURNS|" "$STAGE_DIR/start.sh"

    FILE_COUNT=$(find "$STAGE_DIR" -type f | wc -l)
    echo "  ✓ Staged $FILE_COUNT files"

    # ── Step 5: Upload to Space ────────────────────────────────────────
    echo "[5/6] Uploading to Space..."
    cd "$STAGE_DIR"
    hf upload "$SPACE_ID" . --repo-type space \
      --commit-message "Deploy: $MODEL, $MAX_TASKS tasks, $MAX_STEPS steps" 2>&1 | tail -3
    echo "  ✓ Upload complete"
fi

# ── Step 6: Monitor ────────────────────────────────────────────────────
echo "[6/6] Deployment triggered."
echo ""
echo "  🔗 Space: https://huggingface.co/spaces/$SPACE_ID"
echo "  🔗 Logs:  https://huggingface.co/spaces/$SPACE_ID?logs=container"
echo ""

if [ "$MONITOR" = true ]; then
    echo "Monitoring build (Ctrl+C to stop)..."
    "${PYTHON_CMD[@]}" << PYEOF
from huggingface_hub import HfApi
import time, os

api = HfApi(token=os.environ['HF_TOKEN'])
space_id = "$SPACE_ID"
start = time.time()
last_stage = ""

while True:
    info = api.space_info(space_id)
    stage = info.runtime.stage if info.runtime else "UNKNOWN"
    elapsed = int(time.time() - start)

    if stage != last_stage:
        print(f"  [{elapsed:>4}s] {last_stage} → {stage}")
        last_stage = stage

    if stage == "RUNNING":
        print(f"\n  ✅ Space RUNNING after {elapsed}s")
        print(f"  Health: https://{space_id.replace('/', '-')}.hf.space/health")
        break
    elif stage in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
        print(f"\n  ❌ Failed: {stage}")
        print(f"  Check logs: https://huggingface.co/spaces/{space_id}?logs=build")
        break

    time.sleep(15)
PYEOF
fi

echo ""
echo "Done. Useful commands:"
echo "  # Pause (stop billing):"
echo "  bash $0 --pause"
echo "  # Resume:"
echo "  bash $0 --resume"
echo "  # Monitor logs:"
echo "  curl -N -H 'Authorization: Bearer \$HF_TOKEN' \\"
echo "    'https://huggingface.co/api/spaces/$SPACE_ID/logs/run'"
