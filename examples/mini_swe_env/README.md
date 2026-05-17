# SWE-Gym Training Example

End-to-end training pipeline: SWE-Gym tasks → Pi agent → InterceptionServer → GRPO.

## Overview

This example demonstrates how to train a coding agent on real-world
software engineering tasks from [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym)
using the OpenEnv harness infrastructure.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  train_swe_async_grpo.py                                    │
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
│  4. AsyncGRPOTrainer consumes queue + updates policy        │
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
| `run_swe_sample.py` | Real end-to-end interception rollout with a live LLM |
| `train_swe_async_grpo.py` | AsyncGRPO trainer entrypoint wired to SWE custom rollout worker |
| `envs/mini_swe_env/async_grpo/` | Control plane, worker, and trainer wiring modules |

## Quick Start

### 1. Real end-to-end interception rollout (live LLM)

```bash
SWE_LLM_BASE_URL=https://api.openai.com/v1 \
SWE_LLM_API_KEY=... \
SWE_LLM_MODEL=gpt-4o-mini \
PYTHONPATH=src:envs python examples/mini_swe_env/run_swe_sample.py \
  --task-variant lite --task-index 16 --assert-host-answer
```

Notes:
- Uses `interception_gate` mode and forwards every intercepted request to the
  configured OpenAI-compatible endpoint.
- If `--assert-host-answer` is set, the script fails unless reward comes from
  the host-side `answer` tool path (`reward_source=host_answer_tool`).

### 2. AsyncGRPO training (Track D wiring)

```bash
SWE_ASYNC_MODEL=Qwen/Qwen3-8B \
SWE_LLM_BASE_URL=http://127.0.0.1:8000/v1 \
SWE_LLM_API_KEY=test \
SWE_LLM_MODEL=Qwen/Qwen3-8B \
INTERCEPTION_AUTH_TOKEN=... \
INTERCEPTION_BASE_URL=https://<space-url>.hf.space \
PYTHONPATH=src:envs uv run python examples/mini_swe_env/train_swe_async_grpo.py \
  --task-variant lite --max-tasks 1 --sandbox-backend hf --max-steps 2
```

Notes:
- Use separate GPUs for vLLM and trainer, per TRL async docs.
- `INTERCEPTION_BASE_URL` must be reachable from HF sandboxes.

## Configuration

Main env vars for `train_swe_async_grpo.py`:

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `SWE_ASYNC_MODEL` | *(required)* | Policy model id for `AsyncGRPOTrainer`. |
| `SWE_LLM_BASE_URL` | *(required)* | OpenAI-compatible generation endpoint (typically vLLM `/v1`). |
| `SWE_LLM_API_KEY` | *(required)* | Bearer token for the generation endpoint. |
| `SWE_LLM_MODEL` | *(required)* | Model id exposed by the generation endpoint. |
| `INTERCEPTION_AUTH_TOKEN` | *(required)* | Auth token for interception server + sandbox client wiring. |
| `INTERCEPTION_BASE_URL` | *(required for remote sandboxes)* | Public URL reachable from HF/E2B sandboxes. |
| `SWE_ASYNC_MAX_STALENESS` | `2` | Async sample staleness bound. |
| `SWE_ASYNC_WEIGHT_SYNC_STEPS` | `1` | Weight sync interval. |
| `SWE_ASYNC_MAX_INFLIGHT_TASKS` | `2` | Inflight rollout concurrency. |
| `SWE_ASYNC_QUEUE_MAXSIZE` | `64` | Rollout queue size cap. |
| `SWE_ASYNC_MAX_STEPS` | `10` | Trainer step budget (override with CLI `--max-steps`). |
| `HF_TOKEN` | *(optional)* | Needed when calling private Space URLs from trainer-side answer bridge. |

## Sandbox Connectivity

The InterceptionServer runs on the trainer host.  The sandbox must be
able to reach it:

- **Docker (same host)**: Automatic — uses `http://host.docker.internal:<port>`
- **Remote sandbox (E2B, HF)**: You must provide a tunnel/public URL via
  `INTERCEPTION_BASE_URL`. Options: [bore](https://github.com/ekzhang/bore),
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
