# SWE-Gym Training Example

End-to-end training pipeline: SWE-Gym tasks → Pi agent → InterceptionServer → GRPO.

## Overview

This example demonstrates how to train a coding agent on real-world
software engineering tasks from [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym)
using the OpenEnv harness infrastructure.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  train_swe_grpo.py                                          │
│                                                             │
│  1. Load SWE-Gym tasks from HuggingFace                     │
│  2. Start InterceptionServer on trainer host                 │
│  3. For each batch:                                         │
│     a. SWESessionFactory.create(task) → sandbox + Pi        │
│     b. Pi makes LLM call → intercepted by server            │
│     c. Training loop runs vLLM forward, captures logprobs   │
│     d. Delivers response back to Pi                         │
│     e. Repeat until Pi exits or calls answer()              │
│     f. session.verify() → host-side SWE-Gym grading         │
│  4. Compute GRPO advantages and update policy               │
└─────────────────────────────────────────────────────────────┘
```

### Reward Integrity

The training reward is computed **host-side** by `SWESession.verify()`:

1. Reverts any test files the agent modified (anti-reward-hacking)
2. Applies the known-good `test_patch`
3. Runs the explicit `FAIL_TO_PASS` and `PASS_TO_PASS` pytest cases
4. Computes binary reward from those case outcomes
5. Returns binary reward: `1.0` (resolved) or `0.0` (not resolved)

The agent **cannot** influence the training reward.  The in-sandbox
`answer` tool only gives the agent a feedback signal ("Resolved: true/false").

## Files

| File | Purpose |
|------|---------|
| `config.py` | All training knobs (`SWETrainingConfig`) |
| `smoke_swe.py` | Single-task smoke test (3 modes) |
| `run_swe_sample.py` | Real end-to-end interception rollout with a live LLM |
| `train_swe_grpo.py` | GRPO training loop (reference impl) |

## Quick Start

### 1. Dry Run (no sandbox needed)

Validates config and task loading:

```bash
PYTHONPATH=src:envs python examples/mini_swe_env/smoke_swe.py --dry-run
```

### 2. Smoke Test — Interception Gate (no LLM needed)

Spins up InterceptionServer, creates a real sandbox, and auto-responds
to prove the full pipeline works:

```bash
PYTHONPATH=src:envs python examples/mini_swe_env/smoke_swe.py --interception
```

Requires Docker and the per-task SWE-Gym image to be pullable.

### 3. Smoke Test — Black Box (requires LLM endpoint)

Pi talks directly to an LLM endpoint:

```bash
SWE_BASE_URL=http://localhost:8000/v1 \
SWE_API_KEY=test \
SWE_MODEL=qwen3-8b \
PYTHONPATH=src:envs python examples/mini_swe_env/smoke_swe.py
```

### 4. Real end-to-end run (live LLM)

```bash
SWE_LLM_BASE_URL=https://api.openai.com/v1 \
SWE_LLM_API_KEY=... \
SWE_LLM_MODEL=gpt-4o-mini \
PYTHONPATH=src:envs python examples/mini_swe_env/run_swe_sample.py \
  --task-variant lite --task-index 0 --assert-host-answer
```

Notes:
- Uses `interception_gate` mode and forwards every intercepted request to the
  configured OpenAI-compatible endpoint.
- If `--assert-host-answer` is set, the script fails unless reward comes from
  the host-side `answer` tool path (`reward_source=host_answer_tool`).

### 5. GRPO Training (reference)

```bash
PYTHONPATH=src:envs python examples/mini_swe_env/train_swe_grpo.py --smoke
```

For full training, adjust `config.py` or use environment variables:

```bash
SWE_TASK_VARIANT=lite \
SWE_MODEL=Qwen/Qwen3-32B \
SWE_GRPO_BATCH_SIZE=4 \
SWE_SANDBOX_BACKEND=docker \
PYTHONPATH=src:envs python examples/mini_swe_env/train_swe_grpo.py
```

## Configuration

All settings are in `config.py` (`SWETrainingConfig`).  Override via
`SWE_*` environment variables:

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `SWE_TASK_VARIANT` | `lite` | `"lite"` (230) or `"full"` (2,438 tasks) |
| `SWE_MODEL` | `Qwen/Qwen3-8B` | HuggingFace model id |
| `SWE_BASE_URL` | *(empty)* | LLM endpoint (black-box mode) |
| `SWE_API_KEY` | *(empty)* | LLM bearer token |
| `SWE_SANDBOX_BACKEND` | `docker` | `"docker"`, `"e2b"`, or `"hf"` |
| `SWE_INTERCEPTION_PORT` | `9090` | InterceptionServer port |
| `SWE_INTERCEPTION_BASE_URL` | auto | URL reachable from sandbox |
| `SWE_AGENT_TIMEOUT` | `1800` | Agent timeout (seconds) |
| `SWE_MAX_TURNS` | `30` | Max agent turns per rollout |
| `SWE_GRPO_BATCH_SIZE` | `4` | Tasks per GRPO batch |
| `SWE_GRPO_LR` | `1e-6` | Learning rate |
| `SWE_GRPO_BETA` | `0.04` | KL penalty |
| `SWE_MAX_TASKS` | *(all)* | Cap tasks for debugging |

## Sandbox Connectivity

The InterceptionServer runs on the trainer host.  The sandbox must be
able to reach it:

- **Docker (same host)**: Automatic — uses `http://host.docker.internal:<port>`
- **Remote sandbox (E2B, HF)**: You must provide a tunnel URL via
  `SWE_INTERCEPTION_BASE_URL`.  Options: [bore](https://github.com/ekzhang/bore),
  [ngrok](https://ngrok.com/), [frp](https://github.com/fatedier/frp),
  or a public IP.

## Task Source

Tasks are loaded at runtime from HuggingFace:

- **Lite** (230 tasks): `SWE-Gym/SWE-Gym-Lite` — good for development
- **Full** (2,438 tasks): `SWE-Gym/SWE-Gym` — for production training

Each task comes with a per-task Docker image (`xingyaoww/sweb.eval.x86_64.*`)
that has the repo pre-cloned at `/testbed` with all dependencies installed.

## What the Model Learns

During training, the model sees Pi's system prompt, tool calls, and
results — the exact format it will use at inference time:

```
[user] <problem_statement>

[assistant] Let me investigate...
[tool_call] bash(command="cd /testbed && python -m pytest tests/test_foo.py -x -q")
[tool_result] FAILED tests/test_foo.py::test_bar - AssertionError...

[assistant] I see the bug.
[tool_call] edit(path="/testbed/src/foo.py", edits=[{oldText: "if x > 0", newText: "if x >= 0"}])
[tool_result] ✅ Applied 1 edit

[assistant] Fix verified. Submitting.
[tool_call] answer()
[tool_result] ✅ Resolved: true
```
