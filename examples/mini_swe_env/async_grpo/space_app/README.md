---
title: SWE Async GRPO Training
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: a10g-largex4
startup_duration_timeout: 30m
preload_from_hub:
  - Qwen/Qwen3.5-4B
---

# SWE Async GRPO Training

Trains a language model on SWE-Gym tasks using TRL's AsyncGRPOTrainer
with Pi as the coding agent.

**Architecture**: Pi runs in HF Sandboxes, its LLM calls are intercepted
and forwarded to a co-located vLLM server for generation with exact
token IDs and logprobs. The trainer runs GRPO updates on the remaining
three GPUs (default `TRAINER_GPU=1,2,3`).

See `SWE_ASYNC_GRPO_SPACE_DEPLOYMENT.md` in the repo for full details.

## Required Secrets

Set these in the Space Settings tab:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | HF token for sandbox creation, model downloads, and checkpoint uploads |
| `INTERCEPTION_AUTH_TOKEN` | Shared auth token for Pi ↔ InterceptionServer |
| `SWE_MODEL` | Model ID to serve and train (e.g. `Qwen/Qwen3.5-4B`) |

## Optional Variables (recommended)

| Variable | Description |
|----------|-------------|
| `TRACKIO_SPACE_ID` | Trackio dashboard Space for metrics (e.g. `user/swe-grpo-dashboard`) |
| `SWE_REWARD_MODE` | Reward mode (`binary` recommended, default set by deploy script) |
| `SWE_CHECKPOINT_TO_HUB` | Enable checkpoint upload to HF Hub (`1`/`0`, default `1` on Spaces) |
| `SWE_HUB_MODEL_ID` | Hub model repo used as checkpoint bucket (e.g. `user/swe-async-grpo-checkpoints`) |
| `SWE_RESUME_FROM_CHECKPOINT` | `auto` (default) to resume from `last-checkpoint`, or explicit path/name |
| `SWE_CHECKPOINT_SAVE_STEPS` | Save/upload frequency in trainer steps (default `2`) |
| `SWE_CHECKPOINT_SAVE_TOTAL_LIMIT` | Number of local checkpoints to keep (default `2`) |
| `SWE_HUB_PRIVATE_REPO` | Create/use private checkpoint repo (`1`/`0`, default `1`) |
