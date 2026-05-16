# Mini SWE Environment

A training environment for SWE-bench Lite tasks using OpenEnv.

## Overview

`mini_swe_env` provides an MCP-based environment that:

1. **Provisions a sandbox** (Docker, E2B, or HF Sandbox)
2. **Clones a Git repository** at a specified base commit
3. **Runs setup commands** to prepare the environment
4. **Starts an in-sandbox MCP tool server** exposing a `terminal` tool
5. **Launches a coding agent** (Pi, OpenCode) that uses `terminal` to explore and fix code
6. **Runs verification commands** after the agent finishes
7. **Computes reward** (passed/total verify commands, or explicit override)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Training Script / Client                           │
│  env.run_swe_rollout(task=..., agent="pi", ...)     │
└─────────────┬───────────────────────────────────────┘
              │ MCP tool call
              ▼
┌─────────────────────────────────────────────────────┐
│  SWEEnvironment (OpenEnv Server)                    │
│  - Creates sandbox                                  │
│  - Clones repo at base_commit                       │
│  - Runs setup commands                              │
│  - Deploys in-sandbox MCP server                    │
│  - Launches agent                                   │
│  - Runs verify → computes reward                    │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  Sandbox (Docker / E2B / HF)                        │
│  ┌─────────────────────────────────────────────┐    │
│  │  sandbox_mcp_server.py (port 8765)          │    │
│  │  └─ terminal(command=...) → execute shell   │    │
│  │  └─ terminal(final_answer=...) → verify     │    │
│  └─────────────────┬───────────────────────────┘    │
│                    │ MCP tool calls                 │
│  ┌─────────────────▼───────────────────────────┐    │
│  │  Agent (Pi / OpenCode)                      │    │
│  │  Explores code, makes changes, submits fix  │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  /home/user/workdir/  ← cloned repo at base_commit  │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### As a Client

```python
from mini_swe_env import MiniSWEEnv

with MiniSWEEnv(base_url="http://localhost:8000") as env:
    env.reset()
    result = env.run_swe_rollout(
        instance_id="requests__requests-12345",
        repo="psf/requests",
        base_commit="abc123...",
        instruction="Fix the redirect edge case in Session.send.",
        verify=["python -m pytest tests/test_redirect.py -q"],
        agent="pi",
        base_url="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o-mini",
    )
    print(f"Reward: {result.reward}")
    print(f"Files changed: {list(result.files.keys())}")
```

### From a Task File

```python
from mini_swe_env import MiniSWEEnv, load_task_file
import json

tasks = load_task_file("examples/mini_swe_env/tasks/mini_swe_train.jsonl")

with MiniSWEEnv(base_url="http://localhost:8000") as env:
    env.reset()
    for task in tasks:
        result = env.run_swe_rollout(
            task_json=json.dumps(task.to_dict()),
            agent="pi",
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o-mini",
        )
        print(f"{task.instance_id}: reward={result.reward}")
```

### Running the Server

```bash
# Local development
PYTHONPATH=src:envs uvicorn mini_swe_env.server.app:app --host 0.0.0.0 --port 8000

# With Docker backend (default)
PYTHONPATH=src:envs python -m mini_swe_env.server.app
```

## Task Shape (SWETask)

```python
@dataclass
class SWETask:
    task_id: str              # Unique identifier
    source: str               # "swebench_lite"
    instance_id: str          # SWE-bench instance id
    repo: str                 # "org/repo"
    base_commit: str          # Git commit hash
    instruction: str          # Problem statement
    setup: list[str]          # Commands before agent starts
    verify: list[str]         # Commands after agent finishes
    timeout_s: int = 1800     # Total timeout
    sandbox_image: str | None # Optional custom image
    metadata: dict            # Additional info
```

## Terminal Tool

The in-sandbox MCP server exposes a single `terminal` tool:

### Execute Command
```json
{"command": "git diff HEAD"}
```
Returns: `{"output": "...", "stderr": "...", "exit_code": 0, "done": false}`

### Submit Final Answer
```json
{"final_answer": "I fixed the bug by changing line 42 in utils.py..."}
```
Returns: `{"reward": 0.75, "verify_results": [...], "done": true}`

## Reward Computation

1. **Default**: `passed_verify_commands / total_verify_commands`
2. **Override**: Any verify command (or the agent) can write a float to
   `/home/user/logs/verifier/reward.txt` to override the default.

## Components

| File | Purpose |
|------|---------|
| `task_loader_swebench_lite.py` | SWE-bench Lite → SWETask adapter |
| `models.py` | Pydantic models (SWERolloutResult, SWEState) |
| `server/swe_environment.py` | Main environment (MCPEnvironment) |
| `server/sandbox_mcp_server.py` | In-sandbox terminal tool server |
| `server/app.py` | FastAPI application |
| `client.py` | Typed MiniSWEEnv client |