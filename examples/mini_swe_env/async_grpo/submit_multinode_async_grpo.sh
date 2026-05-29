#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
SBATCH_TEMPLATE="$SCRIPT_DIR/sbatch_multinode_async_grpo.sh"
RUNS_ROOT=${RUNS_ROOT:-$REPO_ROOT/runs/mini_swe_async_grpo}

NODES=${NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
PARTITION=${PARTITION:-hopper-prod}
TIME_LIMIT=${TIME_LIMIT:-08:00:00}
JOB_NAME=${JOB_NAME:-mini-swe-async-grpo}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-r$RANDOM}
RUN_DIR=${RUN_DIR:-$RUNS_ROOT/$RUN_ID}

if (( NODES < 2 || NODES > 4 )); then
  echo "ERROR: NODES must be between 2 and 4, got ${NODES}" >&2
  exit 2
fi
if (( GPUS_PER_NODE < 1 )); then
  echo "ERROR: GPUS_PER_NODE must be >= 1, got ${GPUS_PER_NODE}" >&2
  exit 2
fi
if (( CPUS_PER_TASK < 1 )); then
  echo "ERROR: CPUS_PER_TASK must be >= 1, got ${CPUS_PER_TASK}" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/home"

PORT_SEED=${PORT_SEED:-$(( (RANDOM + $(date -u +%S)) % 1000 ))}
VLLM_PORT=${VLLM_PORT:-$((31000 + PORT_SEED))}
INTERCEPTION_PORT=${INTERCEPTION_PORT:-$((32000 + PORT_SEED))}
MASTER_PORT=${MASTER_PORT:-$((33000 + PORT_SEED))}

export REPO_ROOT RUNS_ROOT RUN_DIR
export GPUS_PER_NODE CPUS_PER_TASK
export VLLM_PORT INTERCEPTION_PORT MASTER_PORT

{
  echo "RUN_ID=$RUN_ID"
  echo "RUN_DIR=$RUN_DIR"
  echo "NODES=$NODES"
  echo "GPUS_PER_NODE=$GPUS_PER_NODE"
  echo "CPUS_PER_TASK=$CPUS_PER_TASK"
  echo "PARTITION=$PARTITION"
  echo "TIME_LIMIT=$TIME_LIMIT"
  echo "JOB_NAME=$JOB_NAME"
  echo "VLLM_PORT=$VLLM_PORT"
  echo "INTERCEPTION_PORT=$INTERCEPTION_PORT"
  echo "MASTER_PORT=$MASTER_PORT"
} > "$RUN_DIR/submission.env"

SBATCH_ARGS=(
  --job-name="$JOB_NAME"
  --partition="$PARTITION"
  --nodes="$NODES"
  --ntasks-per-node=1
  --gres="gpu:h100:${GPUS_PER_NODE}"
  --cpus-per-task="$CPUS_PER_TASK"
  --time="$TIME_LIMIT"
  --output="$RUN_DIR/slurm-%j.out"
  --error="$RUN_DIR/slurm-%j.err"
)

if [[ -n "${SBATCH_NODELIST:-}" ]]; then
  SBATCH_ARGS+=(--nodelist="$SBATCH_NODELIST")
fi
if [[ -n "${SBATCH_EXCLUDE:-}" ]]; then
  SBATCH_ARGS+=(--exclude="$SBATCH_EXCLUDE")
fi

submit_output=$(sbatch "${SBATCH_ARGS[@]}" "$SBATCH_TEMPLATE")
printf '%s\n' "$submit_output"
job_id=$(awk '{print $4}' <<<"$submit_output")
if [[ -n "$job_id" ]]; then
  printf '%s\n' "$job_id" > "$RUN_DIR/job_id.txt"
fi

echo "run_dir=$RUN_DIR"
echo "ports=vllm:${VLLM_PORT} interception:${INTERCEPTION_PORT} master:${MASTER_PORT}"
