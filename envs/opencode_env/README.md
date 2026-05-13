---
title: OpenCode Environment Server
emoji: 🛠️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
short_description: OpenCode coding agent in an E2B sandbox with logprob capture
---

# OpenCode Environment for OpenEnv

`opencode_env` runs the [OpenCode](https://opencode.ai) coding agent inside
an isolated [E2B](https://e2b.dev) sandbox against any OpenAI-compatible
LLM endpoint, optionally capturing per-token logprobs for GRPO training.

**🚀 Try it live**: [`AdithyaSK/opencode-env`](https://huggingface.co/spaces/AdithyaSK/opencode-env)

The deployed Space exposes:

- **Web UI** at [`/web`](https://adithyask-opencode-env.hf.space/web) — pick endpoint, write task, hit Run, watch live phase log + reward + logprobs.
- **MCP tool API** at [`/mcp`](https://adithyask-opencode-env.hf.space/mcp) — programmatic `run_rollout` calls.
- **OpenAPI docs** at [`/docs`](https://adithyask-opencode-env.hf.space/docs).
- **Health** at [`/health`](https://adithyask-opencode-env.hf.space/health).

The env is **task-agnostic** — every rollout is configured at call-time
with a uniform Task shape:

  - **`instruction`** — prompt for the agent
  - **`setup`** — list of bash commands run *before* the agent (pip
    install, git clone, file downloads — anything you need staged in the
    sandbox)
  - **`verify`** — list of bash commands run *after* the agent (asserts,
    pytest invocations, score-file writes)

Reward = `passed_verify / total_verify` unless any `verify` command writes
a float to `/home/user/logs/verifier/reward.txt` (override).

## Quick Start

### Async (default — talk to the deployed Space)

```python
import asyncio
import os
from opencode_env import OpenCodeEnv
from opencode_env.client import _extract_text
from opencode_env.models import RolloutResult


async def main():
    SPACE = "https://adithyask-opencode-env.hf.space"

    async with OpenCodeEnv(base_url=SPACE) as env:
        await env.reset()

        # The MCP tool returns JSON; deserialize via the typed model.
        raw = await env.call_tool(
            "run_rollout",
            endpoint="openai",                          # vllm | openai | hf_router
            api_key=os.environ["OPENAI_API_KEY"],       # or set as a Space secret
            instruction=(
                "Create binary_search.py exposing def binary_search(arr, target) -> int "
                "that returns the index of target in arr, or -1 if absent. Use a "
                "relative path."
            ),
            setup=[],
            verify=[
                "test -f /home/user/workdir/binary_search.py",
                "python -c \"import sys; sys.path.insert(0, '/home/user/workdir'); "
                "import binary_search; "
                "assert binary_search.binary_search([1,2,3], 2) == 1; print('OK')\"",
            ],
            template="opencode-rl",                     # prebaked E2B template
            task_id="binary_search_v1",
        )
        result = RolloutResult.model_validate_json(_extract_text(raw))

        print("reward:", result.reward)
        print("turns:", len(result.proxy_turns))
        print("files:", list(result.files.keys()))
        print("wall:", result.wall_s, "s")


asyncio.run(main())
```

Expected output (~20s with the prebaked template):

```
reward: 1.0
turns: 3
files: ['/home/user/workdir/binary_search.py', ...]
wall: 19.8 s
```

### Sync wrapper

```python
import os
from opencode_env import OpenCodeEnv

# .sync() returns a synchronous wrapper around the async client.
with OpenCodeEnv(base_url="https://adithyask-opencode-env.hf.space").sync() as env:
    env.reset()
    # MCP tools are reachable via env.call_tool(...) / env.step(...) sync-wrapped.
    # See the async example above for the full run_rollout signature.
```

Point `base_url` at `http://localhost:8000` to talk to a local container
instead of the public Space.

### In-process primitive (no HTTP)

For trainers that want to drive a sandbox directly without an HTTP boundary:

```python
import os
from opencode_env import (
    OpenCodeConfig, OpenCodeSessionFactory, OpenCodeTask, E2BSandboxBackend,
)

factory = OpenCodeSessionFactory(
    config=OpenCodeConfig(
        provider="openai_compatible",
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
    ),
    sandbox_backend=E2BSandboxBackend(),
    mode="transparent_proxy",                   # captures per-token logprobs
)
session = factory.create(task=OpenCodeTask(instruction="..."))
session.wait_for_completion()
turns = session.fetch_proxy_trace()             # per-turn (tokens, logprobs)
session.close()
```

## Building the Docker Image

The Dockerfile lives at `server/Dockerfile`. Use the `openenv` CLI from
the env root:

```bash
cd envs/opencode_env

openenv validate               # check pyproject.toml + openenv.yaml + server/app.py + uv.lock
openenv build -t opencode-env  # builds the image (uses server/Dockerfile)

# run locally with E2B credentials
docker run -p 8000:8000 -e E2B_API_KEY=e2b_... opencode-env

# push to HF Spaces (Docker variant)
openenv push --repo-id <user>/opencode-env
```

Or build directly without the CLI:

```bash
docker build -t opencode-env -f envs/opencode_env/server/Dockerfile envs/opencode_env
```

The image:

- Runs `uvicorn server.app:app --host 0.0.0.0 --port 8000`
- Exposes the MCP API at `/mcp` and `/step`, the Gradio UI at `/web`,
  health at `/health`, and OpenAPI docs at `/docs`.
- Reads `E2B_API_KEY` and (optionally) endpoint-specific env vars at
  runtime (see [Environment Variables](#environment-variables)).

## The MCP Tool: `run_rollout`

Single tool, two ways to specify the LLM endpoint:

**Option A — endpoint shorthand (recommended)**: pass
`endpoint="vllm"` (or `"openai"` / `"hf_router"`). The server resolves
`base_url`, `api_key`, and `model` from env vars + catalog defaults.
Any explicit field overrides the catalog.

**Option B — fully explicit**: pass `base_url` + `api_key` + `model`
directly.

| Arg | Type | Default | Notes |
|---|---|---|---|
| `endpoint` | `str` | `""` | One of `"vllm"` / `"openai"` / `"hf_router"`. |
| `base_url` / `api_key` / `model` | `str` | `""` | Override / supply explicitly. |
| `instruction` | `str` | required | Prompt passed to `opencode run`. |
| `setup` | `list[str]` | `[]` | Bash commands run **before** the agent. |
| `verify` | `list[str]` | `[]` | Bash commands run **after** the agent. |
| `task_id` | `str` | `""` | Echoed back in result. |
| `mode` | `str` | `"transparent_proxy"` | Or `"black_box"` (no logprobs). |
| `disable_thinking` | `bool \| None` | `None` (catalog default) | Inject `chat_template_kwargs.enable_thinking=false`. |
| `max_tokens_cap` | `int` | `4096` | Per-turn `max_tokens` clamp. |
| `top_logprobs` | `int` | `5` | HF Router cap is 5; OpenAI 0–20; vLLM unbounded. |
| `agent_timeout_s` | `float` | `600.0` | Hard wall budget for opencode. |
| `template` | `str` | `""` | E2B template name; `"opencode-rl"` skips ~2 min of install per rollout. |

Returns `RolloutResult` JSON with: `reward`, `setup_results[]`,
`verify_results[]`, `proxy_turns[]`, `files{}`, `agent_log_tail`,
`proxy_log_tail`, `wall_s`, `agent_exit_code`, `sandbox_id`, `error`.

## Two Operating Modes

| Mode | What it does | Best for |
|---|---|---|
| **`transparent_proxy`** (default) | In-sandbox proxy at `localhost:7000` forwards opencode's LLM calls to `base_url`, injects `logprobs=true`, captures per-turn `(messages, completion_tokens, logprobs)` to `proxy_trace.jsonl`. | GRPO / RL training, observability, top-k distillation. |
| **`black_box`** | No proxy. opencode talks straight to `base_url`. | Smoke tests, eval, SFT data collection. |

## Environment Variables

The server reads these at runtime. Local dev auto-loads them from a
sibling `.env` file; on HF Spaces, set them as **Space secrets**.

| Variable | Required | Purpose |
|---|---|---|
| `E2B_API_KEY` | **yes** for any rollout | E2B sandbox credentials. |
| `MAX_CONCURRENT_ENVS` | no | Env-instance pool size. Default `4`. |
| `ENABLE_WEB_INTERFACE` | no | Set `false` to disable the `/web` Gradio mount. Default `true`. |
| **vLLM endpoint** | | |
| `VLLM_URL` | required for `endpoint="vllm"` | OAI-compatible base URL. |
| `VLLM_API_KEY` | no | Defaults to `intercepted`. |
| `VLLM_MODEL` | no | Defaults to `Qwen/Qwen3.5-4B`. |
| **OpenAI endpoint** | | |
| `OPENAI_API_KEY` | required for `endpoint="openai"` | Standard OpenAI key. |
| `OPENAI_BASE_URL` | no | Defaults to `https://api.openai.com/v1`. |
| `OPENAI_MODEL` | no | Defaults to `gpt-4o-mini` (gpt-5.x and o-series refuse logprobs). |
| **HF Router endpoint** | | |
| `HF_ROUTER_API_KEY` | required for `endpoint="hf_router"` | HF user token. |
| `HF_ROUTER_BASE_URL` | no | Defaults to `https://router.huggingface.co/v1`. |
| `HF_ROUTER_MODEL` | no | Defaults to `Qwen/Qwen3-4B-Instruct-2507:nscale`. |

Pick `provider:` suffixes that actually return logprobs:
**Together / Nscale / Scaleway / SambaNova / Cerebras**. Avoid Novita /
Hyperbolic / Featherless (silent drop) and Groq (HTTP 400).

## Pre-baked E2B Template

The first rollout in a fresh E2B sandbox spends ~2 min installing
opencode and the proxy's Python deps. Build a one-time template that
ships those pre-installed:

```bash
.venv/bin/python envs/opencode_env/sandbox/build_template.py
# → builds `opencode-rl` template in your E2B account (~1m20s, one-time)
```

After this, pass `template="opencode-rl"` on every `run_rollout` call —
each rollout drops to ~20–30s end-to-end.

## Project Structure

```
opencode_env/
├── README.md                       # this file
├── openenv.yaml                    # OpenEnv space spec
├── pyproject.toml                  # deps + ``server`` entrypoint
├── uv.lock                         # frozen deps (required by ``openenv validate``)
├── .gitignore / .dockerignore      # excludes .env / __pycache__
├── __init__.py                     # re-exports primitive + client + models
│
├── client.py                       # OpenCodeEnv(MCPToolClient)
├── models.py                       # RolloutResult / RolloutTurn / OpenCodeState
│
├── config.py                       # OpenCodeConfig (primitive)
├── harness.py                      # OpenCodeSession / OpenCodeSessionFactory (CLI-only)
├── opencode_runtime.py             # opencode.json builder + cmds
├── task.py                         # OpenCodeTask
│
├── server/
│   ├── __init__.py
│   ├── app.py                      # FastAPI factory; mounts Gradio at /web
│   ├── opencode_environment.py     # MCPEnvironment with single ``run_rollout`` tool
│   ├── gradio_ui.py                # the /web Gradio Blocks UI
│   ├── catalog.py                  # endpoint shorthand resolver
│   └── Dockerfile                  # multi-stage uv build (used by ``openenv build``)
│
└── sandbox/
    ├── __init__.py
    ├── base.py                     # SandboxBackend / SandboxHandle Protocols
    ├── e2b.py                      # E2B implementation
    ├── interception.py             # in-sandbox FastAPI proxy (logprob capture)
    └── build_template.py           # one-time E2B template builder
```

## References

- [OpenEnv docs](https://meta-pytorch.org/OpenEnv/)
- [OpenCode CLI](https://opencode.ai/docs/cli/)
- [E2B Python SDK](https://e2b.dev/docs)
- [HF Inference Providers logprob matrix](../../../DOCS/HF/hf_inference_providers_logprobs.md)
