#!/bin/bash
# deploy_collect_space.sh — Deploy teacher trajectory collection to a CPU-only HF Space
#
# This script:
#   1. Creates a CPU-only HF Space
#   2. Configures secrets (LLM API key, interception auth token)
#   3. Prepares a minimal Docker image with the collection script
#   4. Pushes code and triggers build
#   5. The Space runs collect_rollouts_best_of_n.py autonomously
#
# The Space itself IS the interception server — HF sandboxes call back to the
# Space URL directly (no external tunnel needed).
#
# Prerequisites:
#   - hf CLI installed and authenticated (hf auth login)
#   - HF_TOKEN set or in ~/.cache/huggingface/token
#
# Usage:
#   bash examples/mini_swe_env/deploy_collect_space.sh [OPTIONS]
#
# Options:
#   --space-id OWNER/NAME     Space ID (default: rycerzes/swe-teacher-collect)
#   --model MODEL_NAME        Model name on vLLM endpoint (default: qwen-3.6-27b)
#   --api-url URL             vLLM API base URL (default: https://api.siemens.com/llm/v1)
#   --api-key KEY             vLLM API key (required, or set SWE_LLM_API_KEY)
#   --cf-api-token TOKEN      Cloudflare API token (or set CF_API_TOKEN)
#   --cf-account-id ID        Cloudflare account ID (or set CF_ACCOUNT_ID)
#   --cf-zone-id ID           Cloudflare zone ID (or set CF_ZONE_ID)
#   --cf-domain DOMAIN        Domain for sandbox tunnels (or set CF_DOMAIN)
#   --n-rollouts N            Rollouts per task (default: 4)
#   --max-concurrent N        Concurrent rollouts (default: 3)
#   --max-turns N             Max agent turns (default: 50)
#   --max-tasks N             Limit tasks for testing (default: all 230 Lite)
#   --rate-limit N            Max LLM requests/min (default: 30)
#   --hardware HW             Hardware tier (default: cpu-basic)
#   --skip-build              Only configure, don't push code
#   --monitor                 Monitor logs after deploy
#   --pause                   Pause the Space
#   --resume                  Resume a paused Space
#
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT

SPACE_ID="${HF_SPACE_ID:-rycerzes/swe-teacher-collect}"
MODEL="${SWE_LLM_MODEL:-qwen-3.6-27b}"
API_URL="${SWE_LLM_BASE_URL:-https://api.siemens.com/llm/v1}"
API_KEY="${SWE_LLM_API_KEY:-}"
CF_API_TOKEN="${CF_API_TOKEN:-}"
CF_ACCOUNT_ID="${CF_ACCOUNT_ID:-}"
CF_ZONE_ID="${CF_ZONE_ID:-}"
CF_DOMAIN="${CF_DOMAIN:-}"
N_ROLLOUTS=4
MAX_CONCURRENT=3
MAX_TURNS=50
MAX_TASKS=""
RATE_LIMIT=30
HARDWARE="cpu-basic"
SKIP_BUILD=false
MONITOR=false
PAUSE=false
RESUME=false

# ── Python detection ────────────────────────────────────────────────────
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_CMD=("$REPO_ROOT/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=("$(command -v python3)")
else
    echo "ERROR: Python interpreter not found."
    exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: hf CLI not found. Install huggingface_hub CLI and run 'hf auth login'."
    exit 1
fi

export PYTHONUTF8=1

# ── Parse args ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --space-id) SPACE_ID="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --api-url) API_URL="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --cf-api-token) CF_API_TOKEN="$2"; shift 2 ;;
        --cf-account-id) CF_ACCOUNT_ID="$2"; shift 2 ;;
        --cf-zone-id) CF_ZONE_ID="$2"; shift 2 ;;
        --cf-domain) CF_DOMAIN="$2"; shift 2 ;;
        --n-rollouts) N_ROLLOUTS="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --max-turns) MAX_TURNS="$2"; shift 2 ;;
        --max-tasks) MAX_TASKS="$2"; shift 2 ;;
        --rate-limit) RATE_LIMIT="$2"; shift 2 ;;
        --hardware) HARDWARE="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --monitor) MONITOR=true; shift ;;
        --pause) PAUSE=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate ────────────────────────────────────────────────────────────
if [ -z "$API_KEY" ] && [ "$PAUSE" = false ] && [ "$RESUME" = false ]; then
    echo "ERROR: API key required. Set SWE_LLM_API_KEY or pass --api-key"
    exit 1
fi

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
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.pause_space('$SPACE_ID')
print('Space paused.')
"
    exit 0
fi

if [ "$RESUME" = true ]; then
    echo "▶  Resuming Space $SPACE_ID..."
    "${PYTHON_CMD[@]}" -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.restart_space('$SPACE_ID', factory_reboot=True)
print('Space restarted.')
"
    exit 0
fi

# ── Print config ────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   SWE Teacher Trajectory Collection — HF Space Deploy    ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  %-54s  ║\n" "Space:       $SPACE_ID"
printf "║  %-54s  ║\n" "Model:       $MODEL"
printf "║  %-54s  ║\n" "API URL:     $API_URL"
printf "║  %-54s  ║\n" "Hardware:    $HARDWARE"
printf "║  %-54s  ║\n" "N Rollouts:  $N_ROLLOUTS"
printf "║  %-54s  ║\n" "Concurrent:  $MAX_CONCURRENT"
printf "║  %-54s  ║\n" "Max Turns:   $MAX_TURNS"
printf "║  %-54s  ║\n" "Max Tasks:   ${MAX_TASKS:-all (230 Lite)}"
printf "║  %-54s  ║\n" "Rate Limit:  $RATE_LIMIT req/min"
if [ -n "$CF_API_TOKEN" ]; then
printf "║  %-54s  ║\n" "CF Tunnel:   named (${CF_DOMAIN})"
else
printf "║  %-54s  ║\n" "CF Tunnel:   quick (trycloudflare.com)"
fi
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Create Space ────────────────────────────────────────────────
echo "[1/6] Creating Space (if needed)..."
hf repo create "$SPACE_ID" --repo-type space --space-sdk docker --no-private --exist-ok >/dev/null
echo "  ✓ Space exists: https://huggingface.co/spaces/$SPACE_ID"

# ── Step 2: Configure secrets ───────────────────────────────────────────
echo "[2/6] Configuring secrets and variables..."
"${PYTHON_CMD[@]}" << PYEOF
from huggingface_hub import HfApi
import os, secrets

api = HfApi(token=os.environ['HF_TOKEN'])
space_id = "$SPACE_ID"

# Secrets (sensitive)
space_secrets = {
    "HF_TOKEN": os.environ["HF_TOKEN"],
    "SWE_LLM_API_KEY": "$API_KEY",
    "INTERCEPTION_AUTH_TOKEN": secrets.token_urlsafe(32),
}

# Cloudflare named tunnel secrets (optional — enables reliable sandbox connectivity)
cf_secrets = {
    "CF_API_TOKEN": "$CF_API_TOKEN",
    "CF_ACCOUNT_ID": "$CF_ACCOUNT_ID",
    "CF_ZONE_ID": "$CF_ZONE_ID",
    "CF_DOMAIN": "$CF_DOMAIN",
}
for key, value in cf_secrets.items():
    if value:
        space_secrets[key] = value
for key, value in space_secrets.items():
    try:
        api.add_space_secret(space_id, key, value)
        print(f"  ✓ Secret: {key}")
    except Exception as e:
        print(f"  ⚠ Secret {key}: {e}")

# Variables (non-sensitive)
variables = {
    "SWE_LLM_BASE_URL": "$API_URL",
    "SWE_LLM_MODEL": "$MODEL",
    "N_ROLLOUTS": "$N_ROLLOUTS",
    "MAX_CONCURRENT": "$MAX_CONCURRENT",
    "MAX_TURNS": "$MAX_TURNS",
    "MAX_TASKS": "$MAX_TASKS",
    "RATE_LIMIT": "$RATE_LIMIT",
    "TASK_VARIANT": "lite",
    "HF_SANDBOX_FLAVOR": "cpu-basic",
    "TRAJECTORY_HUB_REPO": f"{space_id.split('/')[0]}/swe-teacher-trajectories",
    "HUB_UPLOAD_EVERY": "5",
}
for key, value in variables.items():
    if not value:
        # Remove variable if empty (e.g. MAX_TASKS not set = use all tasks)
        try:
            api.delete_space_variable(space_id, key)
            print(f"  ✓ Var removed: {key} (using default)")
        except Exception:
            pass
        continue
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
    export STAGE_DIR
    trap "rm -rf $STAGE_DIR" EXIT

    cd "$REPO_ROOT"

    # Copy essential files
    "${PYTHON_CMD[@]}" << 'PYEOF'
import fnmatch
import os
import shutil
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
stage = Path(os.environ["STAGE_DIR"])

def _ignore(patterns):
    def _inner(_dir, names):
        ignored = set()
        for name in names:
            for pat in patterns:
                if fnmatch.fnmatch(name, pat):
                    ignored.add(name)
                    break
        return ignored
    return _inner

common_ignores = [
    "__pycache__", "*.pyc", ".venv", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "*.egg-info", "uv.lock",
]

shutil.copytree(repo / "src", stage / "src", dirs_exist_ok=True, ignore=_ignore(common_ignores))
shutil.copytree(repo / "envs/mini_swe_env", stage / "envs/mini_swe_env", dirs_exist_ok=True, ignore=_ignore(common_ignores))

# Only copy the collection-related files from examples
examples_dst = stage / "examples/mini_swe_env"
examples_dst.mkdir(parents=True, exist_ok=True)
for f in ["collect_rollouts_best_of_n.py", "trajectory_store.py"]:
    src = repo / "examples/mini_swe_env" / f
    if src.exists():
        shutil.copy2(src, examples_dst / f)

PYEOF

    cp pyproject.toml "$STAGE_DIR/"
    [ -f LICENSE ] && cp LICENSE "$STAGE_DIR/"

    # ── Dockerfile (CPU-only, lightweight) ──────────────────────────────
    cat > "$STAGE_DIR/Dockerfile" << 'DOCKERFILE'
FROM python:3.12-slim

WORKDIR /app

# System deps for sandbox operations
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e ".[core]" && \
    pip install --no-cache-dir \
    httpx \
    aiohttp \
    datasets \
    huggingface_hub \
    "hf-sandbox[named-tunnels] @ git+https://github.com/rycerzes/hf-sandbox@feat/named-tunnels" \
    fastmcp

# Copy remaining source
COPY envs/ envs/
COPY examples/ examples/

# The start script runs the collection
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
DOCKERFILE

    # ── start.sh ────────────────────────────────────────────────────────
    cat > "$STAGE_DIR/start.sh" << 'STARTSH'
#!/bin/bash
# Teacher trajectory collection — runs inside HF Space.
#
# The Space URL is automatically the INTERCEPTION_BASE_URL since the
# InterceptionServer binds to the app port (7860).
set -e

echo "========================================"
echo "SWE Teacher Trajectory Collection"
echo "Model:       ${SWE_LLM_MODEL}"
echo "API URL:     ${SWE_LLM_BASE_URL}"
echo "Variant:     ${TASK_VARIANT:-lite}"
echo "N Rollouts:  ${N_ROLLOUTS:-4}"
echo "Concurrent:  ${MAX_CONCURRENT:-3}"
echo "Max Turns:   ${MAX_TURNS:-50}"
echo "Max Tasks:   ${MAX_TASKS:-all}"
echo "Rate Limit:  ${RATE_LIMIT:-30} req/min"
echo "========================================"

# Resolve INTERCEPTION_BASE_URL from Space environment
if [ -z "${INTERCEPTION_BASE_URL:-}" ]; then
    if [ -n "${SPACE_HOST:-}" ]; then
        INTERCEPTION_BASE_URL="https://${SPACE_HOST}"
    elif [ -n "${SPACE_ID:-}" ]; then
        # Convert owner/name → owner-name.hf.space
        OWNER="${SPACE_ID%%/*}"
        NAME="${SPACE_ID##*/}"
        INTERCEPTION_BASE_URL="https://${OWNER}-${NAME}.hf.space"
    else
        echo "ERROR: Cannot determine INTERCEPTION_BASE_URL."
        echo "Set INTERCEPTION_BASE_URL, SPACE_HOST, or SPACE_ID."
        exit 1
    fi
fi
export INTERCEPTION_BASE_URL

echo "Interception URL: ${INTERCEPTION_BASE_URL}"
echo ""

# Build args
ARGS=(
    --task-variant "${TASK_VARIANT:-lite}"
    --n-rollouts "${N_ROLLOUTS:-4}"
    --max-concurrent "${MAX_CONCURRENT:-3}"
    --max-turns "${MAX_TURNS:-50}"
    --rate-limit "${RATE_LIMIT:-30}"
    --output-dir /app/trajectories
    --hf-flavor "${HF_SANDBOX_FLAVOR:-cpu-basic}"
    --max-retries 3
    --interception-port 7860
    --interception-host 0.0.0.0
    --agent-timeout-s 1800
    --export-sft
    --hub-repo-id "${TRAJECTORY_HUB_REPO:-}"
    --hub-upload-every "${HUB_UPLOAD_EVERY:-5}"
)

# Optional: limit tasks
if [ -n "${MAX_TASKS:-}" ]; then
    ARGS+=(--max-tasks "${MAX_TASKS}")
fi

echo "Starting collection..."
export PYTHONPATH=/app/src:/app/envs:/app/examples/mini_swe_env
exec python examples/mini_swe_env/collect_rollouts_best_of_n.py "${ARGS[@]}"
STARTSH

    # ── Space README ────────────────────────────────────────────────────
    cat > "$STAGE_DIR/README.md" << EOF
---
title: SWE Teacher Trajectory Collection
emoji: 📚
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: $HARDWARE
startup_duration_timeout: 10m
---

# SWE Teacher Trajectory Collection

Collects Best-of-N teacher trajectories from \`$MODEL\` on SWE-Gym Lite (230 tasks).

- **Model:** \`$MODEL\` via \`$API_URL\`
- **Rollouts/task:** $N_ROLLOUTS
- **Concurrent:** $MAX_CONCURRENT
- **Max turns:** $MAX_TURNS
- **Rate limit:** $RATE_LIMIT req/min

Trajectories are stored at \`/app/trajectories/\` inside the container.
EOF

    FILE_COUNT=$(find "$STAGE_DIR" -type f | wc -l)
    echo "  ✓ Staged $FILE_COUNT files"

    # ── Step 5: Upload to Space ────────────────────────────────────────
    echo "[5/6] Uploading to Space..."
    cd "$STAGE_DIR"
    hf upload "$SPACE_ID" . --repo-type space \
      --commit-message "Deploy: teacher collection $MODEL, $N_ROLLOUTS rollouts, $MAX_CONCURRENT concurrent" 2>&1 | tail -3
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
        print(f"  [{elapsed:>4}s] {last_stage} -> {stage}")
        last_stage = stage

    if stage == "RUNNING":
        print(f"\n  ✅ Space RUNNING after {elapsed}s")
        print(f"  Logs: https://huggingface.co/spaces/{space_id}?logs=container")
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
echo "  # Download trajectories:"
echo "  hf download $SPACE_ID trajectories/ --repo-type space --local-dir ./trajectories_download"
