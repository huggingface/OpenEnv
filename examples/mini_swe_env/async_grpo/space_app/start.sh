#!/bin/bash
# Startup script for SWE Async GRPO training.
#
# Used both in HF Spaces and local Docker testing.
#
# Process 1 (background): vLLM server
# Process 2 (foreground):  Trainer + InterceptionServer
#
# HF Spaces monitors app_port (7860) for health. The InterceptionServer
# binds to 7860 and serves /health, so the Space is marked Running once
# the trainer starts.
set -e

MODEL="${SWE_MODEL:?ERROR: Set SWE_MODEL (e.g. Qwen/Qwen3-1.7B)}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_KEY="${VLLM_API_KEY:-token}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.9}"

# Prefer repo-local venv Python for local/dev runs; fall back to image python.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "$HOME/app/.venv/bin/python" ]; then
    PYTHON_BIN="$HOME/app/.venv/bin/python"
elif [ -x "$HOME/app/.venv/Scripts/python.exe" ]; then
    PYTHON_BIN="$HOME/app/.venv/Scripts/python.exe"
fi

# GPU assignment. On 2-GPU Spaces: vLLM=0, trainer=1.
# On single GPU: both empty (share GPU 0).
VLLM_GPU="${VLLM_GPU:-0}"
TRAINER_GPU="${TRAINER_GPU:-1}"

echo "========================================"
echo "SWE Async GRPO Training"
echo "Model:         $MODEL"
echo "vLLM port:     $VLLM_PORT"
echo "Max model len: $MAX_MODEL_LEN"
echo "GPU mem util:  $GPU_MEM_UTIL"
echo "vLLM GPU:      $VLLM_GPU"
echo "Trainer GPU:   $TRAINER_GPU"
echo "Checkpointing: ${SWE_CHECKPOINT_TO_HUB:-auto}"
echo "Checkpoint repo: ${SWE_HUB_MODEL_ID:-<auto>}"
echo "========================================"

# ── 1. Start vLLM ─────────────────────────────────────────────
echo "[start.sh] Starting vLLM on GPU $VLLM_GPU..."
# Async GRPO weight sync requires vLLM dev mode + NCCL transfer endpoints.
CUDA_VISIBLE_DEVICES="$VLLM_GPU" VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --api-key "$VLLM_KEY" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}' \
    > /tmp/vllm.log 2>&1 &

VLLM_PID=$!
echo "[start.sh] vLLM PID=$VLLM_PID"

# ── 2. Wait for vLLM health ───────────────────────────────────
echo "[start.sh] Waiting for vLLM to be ready..."
WAITED=0
MAX_WAIT=300
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[start.sh] vLLM ready after ${WAITED}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[start.sh] ERROR: vLLM process died. Log tail:"
        tail -50 /tmp/vllm.log
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[start.sh] ERROR: vLLM did not become ready within ${MAX_WAIT}s. Log tail:"
    tail -50 /tmp/vllm.log
    exit 1
fi

# ── 3. Start trainer (foreground) ──────────────────────────────
echo "[start.sh] Starting trainer on GPU $TRAINER_GPU..."
CUDA_VISIBLE_DEVICES="$TRAINER_GPU" exec "$PYTHON_BIN" examples/mini_swe_env/train_swe_async_grpo.py \
    --vllm-url "http://127.0.0.1:${VLLM_PORT}" \
    "$@"
