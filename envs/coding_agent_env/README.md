---
title: Coding Agent Environment Server
emoji: 🛠️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
short_description: Multi-harness coding-agent env (OpenCode + Pi) in E2B
---

# Coding Agent Environment for OpenEnv

`coding_agent_env` runs coding-agent harnesses (currently
[OpenCode](https://opencode.ai) and [Pi](https://github.com/badlogic/pi-mono))
inside an isolated [E2B](https://e2b.dev) sandbox against any OpenAI-compatible
LLM endpoint with optional trainer-owned interception for RL training.

**🚀 Try it live**: [`AdithyaSK/coding-agent-env`](https://huggingface.co/spaces/AdithyaSK/coding-agent-env)

The deployed Space exposes:

- **Web UI** at [`/web`](https://adithyask-coding-agent-env.hf.space/web) — pick endpoint, write task, hit Run, watch live phase log + reward.
- **MCP tool API** at [`/mcp`](https://adithyask-coding-agent-env.hf.space/mcp) — programmatic `run_rollout` calls.
- **OpenAPI docs** at [`/docs`](https://adithyask-coding-agent-env.hf.space/docs).
- **Health** at [`/health`](https://adithyask-coding-agent-env.hf.space/health).

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
from coding_agent_env import CodingAgentEnv
from coding_agent_env.client import _extract_text
from coding_agent_env.models import RolloutResult


async def main():
    SPACE = "https://adithyask-coding-agent-env.hf.space"

    async with CodingAgentEnv(base_url=SPACE) as env:
        await env.reset()

        # The MCP tool returns JSON; deserialize via the typed model.
        raw = await env.call_tool(
            "run_rollout",
            agent="opencode",                          # opencode | pi
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
            template="coding-agent-rl",                     # prebaked E2B template
            task_id="binary_search_v1",
        )
        result = RolloutResult.model_validate_json(_extract_text(raw))

        print("reward:", result.reward)
        print("files:", list(result.files.keys()))
        print("wall:", result.wall_s, "s")


asyncio.run(main())
```

Expected output (~20s with the prebaked template):

```
reward: 1.0
files: ['/home/user/workdir/binary_search.py', ...]
wall: 19.8 s
```

### Sync wrapper

```python
import os
from coding_agent_env import CodingAgentEnv

# .sync() returns a synchronous wrapper around the async client.
with CodingAgentEnv(base_url="https://adithyask-coding-agent-env.hf.space").sync() as env:
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
from coding_agent_env import (
    CodingAgentConfig, CodingAgentSessionFactory, CodingAgentTask, E2BSandboxBackend,
)

factory = CodingAgentSessionFactory(
    config=CodingAgentConfig(
        provider="openai_compatible",
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
    ),
    sandbox_backend=E2BSandboxBackend(),
    mode="interception_gate",                  # trainer-owned interception mode
)
session = factory.create(task=CodingAgentTask(instruction="..."))
session.wait_for_completion()
session.close()
```

## Building the Docker Image

The Dockerfile lives at `server/Dockerfile`. Use the `openenv` CLI from
the env root:

```bash
cd envs/coding_agent_env

openenv validate               # check pyproject.toml + openenv.yaml + server/app.py + uv.lock
openenv build -t coding-agent-env  # builds the image (uses server/Dockerfile)

# run locally with E2B credentials
docker run -p 8000:8000 -e E2B_API_KEY=e2b_... coding-agent-env

# push to HF Spaces (Docker variant)
openenv push --repo-id <user>/coding-agent-env
```

Or build directly without the CLI:

```bash
docker build -t coding-agent-env -f envs/coding_agent_env/server/Dockerfile envs/coding_agent_env
```

The image:

- Runs `uvicorn server.app:app --host 0.0.0.0 --port 8000`
- Exposes the MCP API at `/mcp` and `/step`, the Gradio UI at `/web`,
  health at `/health`, and OpenAPI docs at `/docs`.
- Reads `E2B_API_KEY` and (optionally) endpoint-specific env vars at
  runtime (see [Environment Variables](#environment-variables)).

## The MCP Tool: `run_rollout`

Single tool, with an ``agent`` selector plus two ways to specify the LLM endpoint:

**Option A — endpoint shorthand (recommended)**: pass
`endpoint="vllm"` (or `"openai"` / `"hf_router"`). The server resolves
`base_url`, `api_key`, and `model` from env vars + catalog defaults.
Any explicit field overrides the catalog.

**Option B — fully explicit**: pass `base_url` + `api_key` + `model`
directly.

| Arg | Type | Default | Notes |
|---|---|---|---|
| `agent` | `str` | `"opencode"` | Harness to run: `"opencode"` or `"pi"`. |
| `endpoint` | `str` | `""` | One of `"vllm"` / `"openai"` / `"hf_router"`. |
| `base_url` / `api_key` / `model` | `str` | `""` | Override / supply explicitly. |
| `instruction` | `str` | required | Prompt passed to the selected harness CLI. |
| `setup` | `list[str]` | `[]` | Bash commands run **before** the agent. |
| `verify` | `list[str]` | `[]` | Bash commands run **after** the agent. |
| `task_id` | `str` | `""` | Echoed back in result. |
| `mode` | `str` | `"black_box"` | Or `"interception_gate"` for trainer-owned generation. |
| `disable_thinking` | `bool \| None` | `None` (catalog default) | Inject `chat_template_kwargs.enable_thinking=false`. |
| `max_tokens_cap` | `int` | `4096` | Per-turn `max_tokens` clamp. |
| `top_logprobs` | `int` | `5` | Reserved for trainer-owned interception workflows. |
| `agent_timeout_s` | `float` | `600.0` | Hard wall budget for the selected harness. |
| `template` | `str` | `""` | E2B template name; `"coding-agent-rl"` skips ~2 min of install per rollout. |

Returns `RolloutResult` JSON with: `reward`, `setup_results[]`,
`verify_results[]`, `files{}`, `agent_log_tail`, `wall_s`,
`agent_exit_code`, `sandbox_id`, `error`.

## Two Operating Modes

| Mode | What it does | Best for |
|---|---|---|
| **`black_box`** (default) | The selected harness talks directly to `base_url`. | Smoke tests, eval, SFT data collection. |
| **`interception_gate`** | Agent calls are routed through trainer-host interception endpoints. Trainer owns forward pass + trajectory capture. | RL training with trainer-owned generation. |

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
| `OPENAI_MODEL` | no | Defaults to `gpt-4o-mini`. |
| **HF Router endpoint** | | |
| `HF_ROUTER_API_KEY` | required for `endpoint="hf_router"` | HF user token. |
| `HF_ROUTER_BASE_URL` | no | Defaults to `https://router.huggingface.co/v1`. |
| `HF_ROUTER_MODEL` | no | Defaults to `Qwen/Qwen3-4B-Instruct-2507:nscale`. |


## Pre-baked E2B Template

The first rollout in a fresh E2B sandbox spends ~2 min installing
harness tooling. Build a one-time template that ships those pre-installed:

```bash
.venv/bin/python envs/coding_agent_env/sandbox/build_template.py
# → builds `coding-agent-rl` template in your E2B account (~1m20s, one-time)
```

After this, pass `template="coding-agent-rl"` on every `run_rollout` call —
each rollout drops to ~20–30s end-to-end.

## Project Structure

```
coding_agent_env/
├── README.md                       # this file
├── openenv.yaml                    # OpenEnv space spec
├── pyproject.toml                  # deps + ``server`` entrypoint
├── uv.lock                         # frozen deps (required by ``openenv validate``)
├── .gitignore / .dockerignore      # excludes .env / __pycache__
├── __init__.py                     # re-exports primitive + client + models
│
├── client.py                       # CodingAgentEnv(MCPToolClient)
├── models.py                       # RolloutResult / CodingAgentState
│
├── config.py                       # CodingAgentConfig (primitive)
├── harness.py                      # CodingAgentSession / CodingAgentSessionFactory (CLI-only)
├── opencode_runtime.py             # opencode.json builder + cmds
├── task.py                         # CodingAgentTask
│
├── server/
│   ├── __init__.py
│   ├── app.py                      # FastAPI factory; mounts Gradio at /web
│   ├── coding_environment.py      # MCPEnvironment with single ``run_rollout`` tool
│   ├── gradio_ui.py                # the /web Gradio Blocks UI
│   ├── catalog.py                  # endpoint shorthand resolver
│   └── Dockerfile                  # multi-stage uv build (used by ``openenv build``)
│
└── sandbox/
    ├── __init__.py
    └── build_template.py           # one-time E2B template builder

# Shared sandbox runtime (moved to core):
src/openenv/core/harness/sandbox/
├── base.py                         # SandboxBackend / SandboxHandle protocols
├── e2b_backend.py                  # E2B implementation
├── docker_backend.py               # local Docker backend
├── hf_backend.py                   # HF sandbox backend
└── _util.py                        # shared sandbox shell utilities
```

## References

- [OpenEnv docs](https://meta-pytorch.org/OpenEnv/)
- [OpenCode CLI](https://opencode.ai/docs/cli/)
- [Pi](https://github.com/badlogic/pi-mono)
- [E2B Python SDK](https://e2b.dev/docs)

