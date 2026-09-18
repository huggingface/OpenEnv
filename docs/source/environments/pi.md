<!-- openenv-source: pi_env -->
# Pi Environment for OpenEnv

`pi_env` runs the [Pi](https://github.com/badlogic/pi-mono) coding agent
inside an isolated [Hugging Face sandbox](https://huggingface.co/docs/huggingface_hub/package_reference/sandbox)
against any OpenAI-compatible LLM endpoint, optionally capturing per-token
logprobs for GRPO training.

It mirrors `opencode_env`: same two-layer design (an
in-process harness primitive + a deployable HTTP env), same transparent-proxy
logprob capture, same uniform `(instruction, setup, verify)` Task shape. The
agent is Pi instead of OpenCode, and the default sandbox backend is Hugging
Face instead of E2B.

The env is **task-agnostic** — every rollout is configured at call-time
with a uniform Task shape:

  - **`instruction`** — prompt for the agent
  - **`setup`** — list of bash commands run *before* the agent (pip install,
    git clone, file downloads — anything you need staged in the sandbox)
  - **`verify`** — list of bash commands run *after* the agent (asserts,
    pytest invocations, score-file writes)

Reward = `passed_verify / total_verify` unless any `verify` command writes
a float to `/root/logs/verifier/reward.txt` (override). The file is deleted
before the verify commands run, so a value the agent writes during its run
is discarded. The override must be a number in `[0, 1]`; anything else is
ignored in favour of the pass rate, with the reason in
`RolloutResult.reward_override_ignored`.

## In-process primitive (no HTTP)

For trainers that drive a sandbox directly without an HTTP boundary — this is
what loop-owning GRPO training uses. The primitive reuses the sandbox backend +
proxy from `opencode_env`, so install it alongside `pi_env`:

```bash
pip install "openenv-opencode-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/opencode_env"
```

```python
import os
from pi_env import PiConfig, PiSessionFactory, PiTask, HFSandboxBackend

factory = PiSessionFactory(
    config=PiConfig(
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
        sandbox_home="/root",                   # HF sandbox execs as root
    ),
    sandbox_backend=HFSandboxBackend(image="python:3.12"),
    mode="transparent_proxy",                   # captures per-token logprobs
)
session = factory.create(task=PiTask(instruction="..."))
session.wait_for_completion()
turns = session.fetch_proxy_trace()             # per-turn (tokens, logprobs)
session.close()
```

Pi is pointed at the endpoint via a `models.json` provider block written under
`PI_CODING_AGENT_DIR` (`api: openai-completions`), then launched headless with
`pi --print --no-session --mode json`. In `transparent_proxy` mode the
in-sandbox proxy fronts `base_url`, injects `logprobs=true`, and writes each
turn's `(messages, completion_token_ids, per_token_logps)` to
`proxy_trace.jsonl`.

### Sandbox backend

`HFSandboxBackend` (from `opencode_env.sandbox`, shared with `opencode_env`)
runs the agent in a Hugging Face sandbox. `image="python:3.12"` cold-installs
Node 22 (bootstrapped) + the Pi CLI (`npm install -g @mariozechner/pi-coding-agent`)
+ the proxy's Python deps on every rollout. For faster rollouts use the
pre-baked image (Node + Pi + proxy deps already installed), built by CI from
`hf_image/Dockerfile`:

```python
sandbox_backend=HFSandboxBackend(image="ghcr.io/huggingface/openenv-pi-sandbox:latest")
```

Any backend satisfying the `SandboxBackend` / `SandboxHandle` / `BgJob`
protocols in `opencode_env.sandbox.base` can be plugged in the same way.

> The sandbox backend and interception proxy live in `opencode_env` for now;
> the plan is to consolidate both into `openenv.core` so `pi_env` and
> `opencode_env` share them without a cross-package import.

## Deployed env (HTTP)

The deployed Space exposes:

- **Web UI** at `/web` — pick endpoint, write task, hit Run, watch live phase
  log + reward + logprobs.
- **MCP tool API** at `/mcp` — programmatic `run_rollout` calls.
- **OpenAPI docs** at `/docs`, **health** at `/health`.

```python
import os
from pi_env import PiEnv

with PiEnv(base_url="https://<user>-pi-env.hf.space") as env:
    env.reset()
    result = env.run_rollout(
        endpoint="openai",                          # vllm | openai | hf_router
        api_key=os.environ["OPENAI_API_KEY"],       # or set as a Space secret
        instruction=(
            "Create binary_search.py exposing def binary_search(arr, target) -> int "
            "that returns the index of target in arr, or -1 if absent."
        ),
        setup=[],
        verify=[
            "test -f /root/workdir/binary_search.py",
            "python -c \"import sys; sys.path.insert(0, '/root/workdir'); "
            "import binary_search; "
            "assert binary_search.binary_search([1,2,3], 2) == 1; print('OK')\"",
        ],
        task_id="binary_search_v1",
    )
    print("reward:", result.reward)
    print("turns:", len(result.proxy_turns))
```

## The MCP Tool: `run_rollout`

Single tool, two ways to specify the LLM endpoint:

**Option A — endpoint shorthand (recommended)**: pass `endpoint="vllm"` (or
`"openai"` / `"hf_router"`). The server resolves `base_url`, `api_key`, and
`model` from env vars + catalog defaults. Any explicit field overrides.

**Option B — fully explicit**: pass `base_url` + `api_key` + `model` directly.

| Arg | Type | Default | Notes |
|---|---|---|---|
| `endpoint` | `str` | `""` | One of `"vllm"` / `"openai"` / `"hf_router"`. |
| `base_url` / `api_key` / `model` | `str` | `""` | Override / supply explicitly. |
| `instruction` | `str` | required | Prompt passed to `pi`. |
| `setup` | `list[str]` | `[]` | Bash commands run **before** the agent. |
| `verify` | `list[str]` | `[]` | Bash commands run **after** the agent. |
| `task_id` | `str` | `""` | Echoed back in result. |
| `mode` | `str` | `"transparent_proxy"` | Or `"black_box"` (no logprobs). |
| `disable_thinking` | `bool \| None` | `None` (catalog default) | Inject `chat_template_kwargs.enable_thinking=false`. |
| `max_tokens_cap` | `int` | `4096` | Per-turn `max_tokens` clamp. |
| `top_logprobs` | `int` | `5` | HF Router cap is 5; OpenAI 0–20; vLLM unbounded. |
| `agent_timeout_s` | `float` | `600.0` | Hard wall budget for one `pi` run. |
| `image` | `str` | `""` | HF sandbox image; blank → `python:3.12` (cold-installs Node + Pi). |

Returns `RolloutResult` JSON with: `reward`, `setup_results[]`,
`verify_results[]`, `proxy_turns[]`, `files{}`, `agent_log_tail`,
`proxy_log_tail`, `wall_s`, `agent_exit_code`, `sandbox_id`, `error`.

## Two Operating Modes

| Mode | What it does | Best for |
|---|---|---|
| **`transparent_proxy`** (default) | In-sandbox proxy at `localhost:7000` forwards Pi's LLM calls to `base_url`, injects `logprobs=true`, captures per-turn `(messages, completion_tokens, logprobs)` to `proxy_trace.jsonl`. | GRPO / RL training, observability, top-k distillation. |
| **`black_box`** | No proxy. Pi talks straight to `base_url`. | Smoke tests, eval, SFT data collection. |

## Building the Docker Image

```bash
cd envs/pi_env

openenv validate           # check pyproject.toml + openenv.yaml + server/app.py + uv.lock
openenv build -t pi-env    # builds the image (uses server/Dockerfile)

# run locally with an HF token (Sandbox + Jobs access)
docker run -p 8000:8000 -e HF_TOKEN=hf_... pi-env
```

Or build directly:

```bash
docker build -t pi-env -f envs/pi_env/server/Dockerfile envs/pi_env
```

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | **yes** for any rollout | Hugging Face sandbox credentials. |
| `MAX_CONCURRENT_ENVS` | no | Env-instance pool size. Default `4`. |
| `ENABLE_WEB_INTERFACE` | no | Set `false` to disable the `/web` Gradio mount. Default `true`. |
| `VLLM_URL` / `VLLM_API_KEY` / `VLLM_MODEL` | for `endpoint="vllm"` | OAI-compatible base URL (key defaults to `intercepted`). |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | for `endpoint="openai"` | Standard OpenAI. |
| `HF_ROUTER_API_KEY` / `HF_ROUTER_BASE_URL` / `HF_ROUTER_MODEL` | for `endpoint="hf_router"` | HF Router. |

Pick `provider:` suffixes that actually return logprobs:
**Together / Nscale / Scaleway / SambaNova / Cerebras**. Avoid Novita /
Hyperbolic / Featherless (silent drop) and Groq (HTTP 400).

## Project Structure

```
pi_env/
├── README.md                       # this file
├── openenv.yaml                    # OpenEnv space spec
├── pyproject.toml                  # deps + ``server`` entrypoint
├── __init__.py                     # re-exports primitive + client + models
│
├── client.py                       # PiEnv(MCPToolClient)
├── models.py                       # RolloutResult / RolloutTurn / PiState
│
├── config.py                       # PiConfig (primitive)
├── harness.py                      # PiSession / PiSessionFactory (CLI-only)
├── pi_runtime.py                   # models.json builder + install/run cmds
├── task.py                         # PiTask
│
└── server/
    ├── __init__.py
    ├── app.py                      # FastAPI factory; mounts Gradio at /web
    ├── pi_environment.py           # MCPEnvironment with single ``run_rollout`` tool
    ├── gradio_ui.py                # the /web Gradio Blocks UI
    ├── catalog.py                  # endpoint shorthand resolver
    └── Dockerfile                  # multi-stage uv build (used by ``openenv build``)
```

The sandbox backend + interception proxy are imported from
`opencode_env.sandbox`; `pi_env` ships no `sandbox/` of its own.
