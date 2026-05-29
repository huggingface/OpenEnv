---
title: Terminus Environment Server
emoji: 🛠️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - terminus
  - hf-sandbox
  - coding
short_description: Single-tool coding environment
---

# Terminus Environment

`terminus_env` is a single-tool coding environment. Each OpenEnv episode
creates a fresh sandbox, runs optional setup commands, keeps shell state and
files isolated for that episode, and runs optional verify commands when the
agent submits a final answer.

The tool shape follows the Terminus-style "one tool" idea: agents do their work
through a single terminal entrypoint rather than a notebook/toolbox surface.

## Tool

- `terminal(command="", final_answer="")`: run a shell command inside the
  session sandbox, or submit a final answer and run verification.

## Quick Start

```python
from terminus_env import TerminusEnv

with TerminusEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(
        setup=["mkdir -p /home/user/work"],
        verify=["test -f /home/user/work/answer.txt"],
    )
    print(env.call_tool("terminal", command="echo done > /home/user/work/answer.txt"))
    print(env.call_tool("terminal", final_answer="done"))
```

## Local Server

```bash
cd envs/terminus_env
TERMINUS_SANDBOX_BACKEND=local uv run --project . server
```

The API and custom terminal web UI are served on port 8000. The UI is mounted
at `/web`.

## Docker

```bash
cd envs/terminus_env
openenv build -t terminus-env
docker run -p 8000:8000 -e HF_TOKEN=hf_... terminus-env
```

## Configuration

- `TERMINUS_SANDBOX_BACKEND`: `local` for the lightweight cluster smoke
  backend, or `hf` for `hf-sandbox`. Defaults to `hf`.
- `HF_TOKEN`: required by the optional `hf-sandbox` backend to launch
  Hugging Face Jobs.
- `HF_SANDBOX_IMAGE`: sandbox image. Defaults to `python:3.12`.
- `HF_SANDBOX_FLAVOR`: Hugging Face Jobs flavor. Defaults to `cpu-basic`.
- `HF_SANDBOX_TIMEOUT`: Hugging Face Jobs timeout. Defaults to `1h`.
- `HF_SANDBOX_FORWARD_HF_TOKEN`: forward `HF_TOKEN` into the sandbox. Defaults
  to `false`.
- `MAX_CONCURRENT_ENVS`: maximum concurrent WebSocket sessions. Defaults to `4`.

The local backend requires `bwrap` on the server node and is intended for simple
cluster smoke tasks. Install the `hf` extra and use the `hf` backend for
stronger remote sandboxing.

## Setup and Verify Commands

`reset()` accepts either `setup` / `verify` or `setup_scripts` /
`verify_scripts`.

```python
env.reset(
    setup=["pip install -q pytest"],
    verify=["pytest -q /home/user/work/tests"],
)
```

Setup failure ends the reset response with `done=True` and returns captured
setup results. Verify commands run when `terminal(final_answer="...")` is
called. Reward defaults to `passed_verify_commands / total_verify_commands`. A
verify command can override this by writing a float to:

```text
/home/user/logs/verifier/reward.txt
```
