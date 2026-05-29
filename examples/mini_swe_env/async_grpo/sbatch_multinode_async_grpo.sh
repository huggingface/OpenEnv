#!/usr/bin/env bash
#SBATCH --job-name=mini-swe-async-grpo
#SBATCH --partition=hopper-prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO_ROOT=${SLURM_SUBMIT_DIR:-}
if [[ -z "$DEFAULT_REPO_ROOT" ]]; then
  DEFAULT_REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
fi
REPO_ROOT=${REPO_ROOT:-$DEFAULT_REPO_ROOT}
RUNS_ROOT=${RUNS_ROOT:-$REPO_ROOT/runs/mini_swe_async_grpo}
RUN_DIR=${RUN_DIR:-$RUNS_ROOT/${SLURM_JOB_ID:-manual}}
CLOUDFLARED=${CLOUDFLARED:-/fsx/benjamin_burtenshaw/bin/cloudflared}

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}
JOB_PORT_BASE=${JOB_PORT_BASE:-$((20000 + (${SLURM_JOB_ID:-0} % 10000)))}
VLLM_PORT=${VLLM_PORT:-$JOB_PORT_BASE}
INTERCEPTION_PORT=${INTERCEPTION_PORT:-$((JOB_PORT_BASE + 1000))}
MASTER_PORT=${MASTER_PORT:-$((JOB_PORT_BASE + 2000))}

SWE_MODEL=${SWE_MODEL:-Qwen/Qwen3-1.7B}
SWE_SANDBOX_BACKEND=${SWE_SANDBOX_BACKEND:-hf}
SWE_AGENT=${SWE_AGENT:-pi}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
MAX_TASKS=${MAX_TASKS:-4}
MAX_STEPS=${MAX_STEPS:-3}
MAX_TURNS=${MAX_TURNS:-30}
SWE_ROLLOUT_MAX_INFLIGHT=${SWE_ROLLOUT_MAX_INFLIGHT:-}
SWE_TRAIN_DTYPE=${SWE_TRAIN_DTYPE:-bf16}
SWE_TORCH_EMPTY_CACHE_STEPS=${SWE_TORCH_EMPTY_CACHE_STEPS:-1}
SWE_LORA=${SWE_LORA:-0}
SWE_LORA_R=${SWE_LORA_R:-16}
SWE_LORA_ALPHA=${SWE_LORA_ALPHA:-}
SWE_LORA_DROPOUT=${SWE_LORA_DROPOUT:-0.0}
SWE_LORA_TARGET_MODULES=${SWE_LORA_TARGET_MODULES:-}
SWE_LORA_BIAS=${SWE_LORA_BIAS:-none}
SWE_LORA_USE_RSLORA=${SWE_LORA_USE_RSLORA:-0}

if [[ -z "$MAX_MODEL_LEN" ]]; then
  MAX_MODEL_LEN=$(
    SWE_MODEL="$SWE_MODEL" "$REPO_ROOT/.venv/bin/python" - <<'PY' 2>/dev/null || true
from transformers import AutoConfig
import os

cfg = AutoConfig.from_pretrained(os.environ["SWE_MODEL"])
for key in ("max_position_embeddings", "model_max_length", "max_seq_len", "seq_length"):
    value = getattr(cfg, key, None)
    if isinstance(value, int) and value > 0:
        print(value)
        break
PY
  )
fi
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

if [[ -z "$SWE_ROLLOUT_MAX_INFLIGHT" ]]; then
  if [[ "$SWE_SANDBOX_BACKEND" == "hf" ]]; then
    SWE_ROLLOUT_MAX_INFLIGHT=1
  else
    SWE_ROLLOUT_MAX_INFLIGHT=2
  fi
fi

MODEL_LOWER=$(printf '%s' "$SWE_MODEL" | tr '[:upper:]' '[:lower:]')
VLLM_TOOL_CALL_PARSER=${VLLM_TOOL_CALL_PARSER:-}
if [[ -z "$VLLM_TOOL_CALL_PARSER" ]]; then
  case "$MODEL_LOWER" in
    *qwen3*coder*)
      VLLM_TOOL_CALL_PARSER=qwen3_coder
      ;;
    *qwen3*)
      VLLM_TOOL_CALL_PARSER=qwen3_xml
      ;;
  esac
fi

mkdir -p "$RUN_DIR/home"
cd "$REPO_ROOT"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -s /admin/home/benjamin_burtenshaw/.cache/huggingface/token ]]; then
    HF_TOKEN="$(< /admin/home/benjamin_burtenshaw/.cache/huggingface/token)"
  elif [[ -s /fsx/benjamin_burtenshaw/.cache/huggingface/token ]]; then
    HF_TOKEN="$(< /fsx/benjamin_burtenshaw/.cache/huggingface/token)"
  fi
fi
if [[ "$SWE_SANDBOX_BACKEND" == "hf" && -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required for HF sandbox creation" >&2
  exit 2
fi

INTERCEPTION_AUTH_TOKEN=${INTERCEPTION_AUTH_TOKEN:-$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)}

mapfile -t ALLOC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if (( ${#ALLOC_NODES[@]} < 2 )); then
  echo "ERROR: expected at least 2 allocated nodes, got ${#ALLOC_NODES[@]}" >&2
  exit 3
fi

VLLM_NODE=${ALLOC_NODES[0]}
TRAINER_NODES=("${ALLOC_NODES[@]:1}")
TRAINER_MASTER=${TRAINER_NODES[0]}
TRAINER_NODE_COUNT=${#TRAINER_NODES[@]}
TRAINER_NODELIST=$(IFS=,; echo "${TRAINER_NODES[*]}")
TOTAL_TRAINER_PROCS=$((TRAINER_NODE_COUNT * GPUS_PER_NODE))
VLLM_URL="http://${VLLM_NODE}:${VLLM_PORT}"
GPU_IDS=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

export RUN_DIR REPO_ROOT GPUS_PER_NODE VLLM_PORT MASTER_PORT
export MAX_MODEL_LEN GPU_MEMORY_UTILIZATION MAX_TASKS MAX_STEPS MAX_TURNS
export TRAINER_NODE_COUNT TOTAL_TRAINER_PROCS TRAINER_MASTER VLLM_URL GPU_IDS
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/envs"
export PYTHONUNBUFFERED=1
export TRL_EXPERIMENTAL_SILENCE=1
export HF_HOME=/fsx/benjamin_burtenshaw/.cache/huggingface
export HF_HUB_CACHE=/fsx/benjamin_burtenshaw/.cache/huggingface/hub
export HF_DATASETS_CACHE=/fsx/benjamin_burtenshaw/.cache/huggingface/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_CACHE_ROOT=/fsx/benjamin_burtenshaw/.cache/vllm
export XDG_CACHE_HOME=/fsx/benjamin_burtenshaw/.cache
export TORCH_HOME=/fsx/benjamin_burtenshaw/.cache/torch
export TRITON_CACHE_DIR=/fsx/benjamin_burtenshaw/.cache/triton
export FLASHINFER_CACHE_DIR=/fsx/benjamin_burtenshaw/.cache/flashinfer
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/tmp/${USER}/openenv-pycache}
export VLLM_NO_USAGE_STATS=1
export HF_TOKEN
export HOME="$RUN_DIR/home"
export SWE_MODEL
export SWE_SANDBOX_BACKEND
export SWE_AGENT
export VLLM_API_KEY=${VLLM_API_KEY:-token}
export INTERCEPTION_HOST=0.0.0.0
export INTERCEPTION_PORT
export INTERCEPTION_AUTH_TOKEN
export TRACKIO_SPACE_ID=${TRACKIO_SPACE_ID:-burtenshaw/swe-grpo-dashboard}
export SWE_TRACKIO_PROJECT=${SWE_TRACKIO_PROJECT:-swe-async-grpo}
export SWE_CHECKPOINT_TO_HUB=${SWE_CHECKPOINT_TO_HUB:-0}
export SWE_HUB_MODEL_ID=${SWE_HUB_MODEL_ID:-}
export SWE_HUB_PRIVATE_REPO=${SWE_HUB_PRIVATE_REPO:-1}
export SWE_CHECKPOINT_SAVE_STEPS=${SWE_CHECKPOINT_SAVE_STEPS:-2}
export SWE_CHECKPOINT_SAVE_TOTAL_LIMIT=${SWE_CHECKPOINT_SAVE_TOTAL_LIMIT:-2}
export SWE_RESUME_FROM_CHECKPOINT=${SWE_RESUME_FROM_CHECKPOINT:-none}
export SWE_ROLLOUT_MAX_INFLIGHT
export SWE_VLLM_MAX_MODEL_LEN=${SWE_VLLM_MAX_MODEL_LEN:-$MAX_MODEL_LEN}
export SWE_TRAIN_DTYPE SWE_TORCH_EMPTY_CACHE_STEPS
export SWE_LORA SWE_LORA_R SWE_LORA_ALPHA SWE_LORA_DROPOUT
export SWE_LORA_TARGET_MODULES SWE_LORA_BIAS SWE_LORA_USE_RSLORA
export SWE_HF_SANDBOX_CREATE_RETRIES=${SWE_HF_SANDBOX_CREATE_RETRIES:-6}
export SWE_HF_SANDBOX_CREATE_BACKOFF_S=${SWE_HF_SANDBOX_CREATE_BACKOFF_S:-20}
export OPENENV_HF_SANDBOX_URL_TIMEOUT_S=${OPENENV_HF_SANDBOX_URL_TIMEOUT_S:-120}
export SWE_ROLLOUT_QUEUE_TIMEOUT_S=${SWE_ROLLOUT_QUEUE_TIMEOUT_S:-900}
export SWE_ROLLOUT_FAILURE_BACKOFF_S=${SWE_ROLLOUT_FAILURE_BACKOFF_S:-30}
export SWE_ROLLOUT_MAX_ATTEMPTS=${SWE_ROLLOUT_MAX_ATTEMPTS:-4}
export SWE_DISABLE_WEIGHT_TRANSFER=${SWE_DISABLE_WEIGHT_TRANSFER:-0}
export VLLM_TOOL_CALL_PARSER
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENENV_LOCAL_SANDBOX_ROOT=${OPENENV_LOCAL_SANDBOX_ROOT:-$RUN_DIR/local_sandboxes}
mkdir -p "$OPENENV_LOCAL_SANDBOX_ROOT"

VLLM_STEP_PID=
CLOUDFLARED_PID=
cleanup() {
  local rc=$?
  if [[ -n "${CLOUDFLARED_PID:-}" ]]; then
    kill "$CLOUDFLARED_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${VLLM_STEP_PID:-}" ]]; then
    kill "$VLLM_STEP_PID" >/dev/null 2>&1 || true
    wait "$VLLM_STEP_PID" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT

echo "job_start=$(date -Is)"
echo "run_dir=$RUN_DIR"
echo "nodes=${ALLOC_NODES[*]}"
echo "vllm_node=$VLLM_NODE trainer_nodes=$TRAINER_NODELIST trainer_master=$TRAINER_MASTER"
echo "ports=vllm:$VLLM_PORT interception:$INTERCEPTION_PORT master:$MASTER_PORT"
echo "model=$SWE_MODEL agent=$SWE_AGENT parser=${VLLM_TOOL_CALL_PARSER:-none} sandbox=$SWE_SANDBOX_BACKEND dtype=$SWE_TRAIN_DTYPE lora=$SWE_LORA max_model_len=$MAX_MODEL_LEN max_tasks=$MAX_TASKS max_steps=$MAX_STEPS max_turns=$MAX_TURNS inflight=$SWE_ROLLOUT_MAX_INFLIGHT"

: > "$RUN_DIR/vllm.log"
: > "$RUN_DIR/trainer.log"
: > "$RUN_DIR/cloudflared.log"

echo "starting_vllm=$(date -Is)"
srun \
  --nodes=1 \
  --ntasks=1 \
  --nodelist="$VLLM_NODE" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --gres="gpu:h100:${GPUS_PER_NODE}" \
  --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    cd "$REPO_ROOT"
    mkdir -p "$PYTHONPYCACHEPREFIX"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    export VLLM_SERVER_DEV_MODE=1
    TOOL_ARGS=()
    if [[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
      TOOL_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$VLLM_TOOL_CALL_PARSER")
    fi
    exec .venv/bin/vllm serve "$SWE_MODEL" \
      --tensor-parallel-size "$GPUS_PER_NODE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --host 0.0.0.0 \
      --port "$VLLM_PORT" \
      --api-key "$VLLM_API_KEY" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --logprobs-mode processed_logprobs \
      --weight-transfer-config '\''{"backend":"nccl"}'\'' \
      "${TOOL_ARGS[@]}"
  ' > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_STEP_PID=$!

VLLM_BIND_MARKER="http://0.0.0.0:${VLLM_PORT}"
VLLM_READY_MARKER="Application startup complete."
for _ in $(seq 1 900); do
  if grep -Fq "Address already in use" "$RUN_DIR/vllm.log"; then
    echo "ERROR: vLLM failed to bind on ${VLLM_PORT}" >&2
    tail -100 "$RUN_DIR/vllm.log" >&2 || true
    exit 4
  fi
  if ! kill -0 "$VLLM_STEP_PID" >/dev/null 2>&1; then
    echo "ERROR: vLLM srun exited before readiness" >&2
    tail -100 "$RUN_DIR/vllm.log" >&2 || true
    exit 4
  fi
  if grep -Fq "$VLLM_BIND_MARKER" "$RUN_DIR/vllm.log" \
    && grep -Fq "$VLLM_READY_MARKER" "$RUN_DIR/vllm.log" \
    && curl -fsS "$VLLM_URL/health" >/dev/null 2>&1 \
    && curl -fsS "$VLLM_URL/get_world_size" >/dev/null 2>&1; then
    echo "vllm_ready=$(date -Is)"
    break
  fi
  sleep 2
done
if ! grep -Fq "$VLLM_BIND_MARKER" "$RUN_DIR/vllm.log" \
  || ! grep -Fq "$VLLM_READY_MARKER" "$RUN_DIR/vllm.log" \
  || ! curl -fsS "$VLLM_URL/health" >/dev/null 2>&1 \
  || ! curl -fsS "$VLLM_URL/get_world_size" >/dev/null 2>&1; then
  echo "ERROR: vLLM did not become ready on ${VLLM_URL}" >&2
  tail -100 "$RUN_DIR/vllm.log" >&2 || true
  exit 5
fi

INTERCEPTION_BASE_URL=
if [[ "$SWE_SANDBOX_BACKEND" == "docker" ]]; then
  INTERCEPTION_BASE_URL="http://host.docker.internal:${INTERCEPTION_PORT}"
elif [[ "$SWE_SANDBOX_BACKEND" == "local" ]]; then
  INTERCEPTION_BASE_URL="http://127.0.0.1:${INTERCEPTION_PORT}"
else
  echo "starting_cloudflared=$(date -Is)"
  "$CLOUDFLARED" tunnel \
    --url "http://${TRAINER_MASTER}:${INTERCEPTION_PORT}" \
    --no-autoupdate \
    --protocol quic \
    --ha-connections 1 \
    > "$RUN_DIR/cloudflared.log" 2>&1 &
  CLOUDFLARED_PID=$!
  echo "$CLOUDFLARED_PID" > "$RUN_DIR/cloudflared.pid"

  for _ in $(seq 1 120); do
    INTERCEPTION_BASE_URL=$(grep -Eo 'https://[A-Za-z0-9.-]+\.trycloudflare\.com' "$RUN_DIR/cloudflared.log" | tail -n 1 || true)
    if [[ -n "$INTERCEPTION_BASE_URL" ]]; then
      break
    fi
    if ! kill -0 "$CLOUDFLARED_PID" >/dev/null 2>&1; then
      echo "ERROR: cloudflared exited before URL creation" >&2
      tail -100 "$RUN_DIR/cloudflared.log" >&2 || true
      exit 6
    fi
    sleep 1
  done
  if [[ -z "$INTERCEPTION_BASE_URL" ]]; then
    echo "ERROR: cloudflared did not publish a tunnel URL" >&2
    tail -100 "$RUN_DIR/cloudflared.log" >&2 || true
    exit 7
  fi
fi
export INTERCEPTION_BASE_URL
echo "$INTERCEPTION_BASE_URL" > "$RUN_DIR/interception_base_url.txt"

{
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "RUN_DIR=$RUN_DIR"
  echo "SWE_MODEL=$SWE_MODEL"
  echo "SWE_SANDBOX_BACKEND=$SWE_SANDBOX_BACKEND"
  echo "SWE_AGENT=$SWE_AGENT"
  echo "VLLM_NODE=$VLLM_NODE"
  echo "VLLM_URL=$VLLM_URL"
  echo "VLLM_PORT=$VLLM_PORT"
  echo "TRAINER_NODELIST=$TRAINER_NODELIST"
  echo "TRAINER_MASTER=$TRAINER_MASTER"
  echo "TOTAL_TRAINER_PROCS=$TOTAL_TRAINER_PROCS"
  echo "INTERCEPTION_BASE_URL=$INTERCEPTION_BASE_URL"
  echo "INTERCEPTION_PORT=$INTERCEPTION_PORT"
  echo "MASTER_PORT=$MASTER_PORT"
  echo "TRACKIO_SPACE_ID=$TRACKIO_SPACE_ID"
  echo "SWE_TRACKIO_PROJECT=$SWE_TRACKIO_PROJECT"
  echo "SWE_CHECKPOINT_TO_HUB=$SWE_CHECKPOINT_TO_HUB"
  echo "SWE_HUB_MODEL_ID=$SWE_HUB_MODEL_ID"
  echo "SWE_HUB_PRIVATE_REPO=$SWE_HUB_PRIVATE_REPO"
  echo "SWE_CHECKPOINT_SAVE_STEPS=$SWE_CHECKPOINT_SAVE_STEPS"
  echo "SWE_CHECKPOINT_SAVE_TOTAL_LIMIT=$SWE_CHECKPOINT_SAVE_TOTAL_LIMIT"
  echo "SWE_RESUME_FROM_CHECKPOINT=$SWE_RESUME_FROM_CHECKPOINT"
  echo "MAX_TASKS=$MAX_TASKS"
  echo "MAX_STEPS=$MAX_STEPS"
  echo "MAX_TURNS=$MAX_TURNS"
  echo "SWE_TASK_INDICES=${SWE_TASK_INDICES:-}"
  echo "SWE_REPEAT_TASKS=${SWE_REPEAT_TASKS:-}"
  echo "SWE_NUM_GENERATIONS=${SWE_NUM_GENERATIONS:-}"
  echo "SWE_TEMPERATURE=${SWE_TEMPERATURE:-}"
  echo "SWE_LEARNING_RATE=${SWE_LEARNING_RATE:-}"
  echo "SWE_REWARD_MODE=${SWE_REWARD_MODE:-}"
  echo "SWE_ENABLE_ANSWER_TOOL=${SWE_ENABLE_ANSWER_TOOL:-}"
  echo "SWE_ROLLOUT_MAX_INFLIGHT=$SWE_ROLLOUT_MAX_INFLIGHT"
  echo "SWE_ROLLOUT_MAX_ATTEMPTS=$SWE_ROLLOUT_MAX_ATTEMPTS"
  echo "SWE_VLLM_MAX_MODEL_LEN=$SWE_VLLM_MAX_MODEL_LEN"
  echo "SWE_TRAIN_DTYPE=$SWE_TRAIN_DTYPE"
  echo "SWE_LORA=$SWE_LORA"
  echo "SWE_LORA_R=$SWE_LORA_R"
  echo "SWE_LORA_ALPHA=$SWE_LORA_ALPHA"
  echo "SWE_LORA_DROPOUT=$SWE_LORA_DROPOUT"
  echo "SWE_LORA_TARGET_MODULES=$SWE_LORA_TARGET_MODULES"
  echo "SWE_LORA_BIAS=$SWE_LORA_BIAS"
  echo "SWE_LORA_USE_RSLORA=$SWE_LORA_USE_RSLORA"
  echo "SWE_OPTIM=${SWE_OPTIM:-}"
  echo "SWE_TORCH_EMPTY_CACHE_STEPS=$SWE_TORCH_EMPTY_CACHE_STEPS"
  echo "VLLM_TOOL_CALL_PARSER=${VLLM_TOOL_CALL_PARSER:-}"
  echo "SWE_DISABLE_WEIGHT_TRANSFER=$SWE_DISABLE_WEIGHT_TRANSFER"
} > "$RUN_DIR/run.env"

echo "starting_trainer=$(date -Is)"
srun \
  --nodes="$TRAINER_NODE_COUNT" \
  --ntasks="$TRAINER_NODE_COUNT" \
  --ntasks-per-node=1 \
  --nodelist="$TRAINER_NODELIST" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --gres="gpu:h100:${GPUS_PER_NODE}" \
  --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    cd "$REPO_ROOT"
    mkdir -p "$PYTHONPYCACHEPREFIX"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    .venv/bin/python .venv/bin/accelerate launch \
      --num_processes "$TOTAL_TRAINER_PROCS" \
      --num_machines "$TRAINER_NODE_COUNT" \
      --machine_rank "$SLURM_PROCID" \
      --main_process_ip "$TRAINER_MASTER" \
      --main_process_port "$MASTER_PORT" \
      --mixed_precision no \
      --num_cpu_threads_per_process "$OMP_NUM_THREADS" \
      examples/mini_swe_env/train_swe_async_grpo.py \
        --sandbox-backend "$SWE_SANDBOX_BACKEND" \
        --agent "$SWE_AGENT" \
        --vllm-url "$VLLM_URL" \
        --task-variant lite \
        --max-tasks "$MAX_TASKS" \
        --max-steps "$MAX_STEPS" \
        --max-turns "$MAX_TURNS"
  ' > "$RUN_DIR/trainer.log" 2>&1

echo "trainer_done=$(date -Is)"
echo "job_done=$(date -Is)"
