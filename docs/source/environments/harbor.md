<!-- openenv-source: harbor_env -->
# Harbor Environment

Runs [Harbor](https://www.harborframework.com/docs/tasks) task directories as an
OpenEnv environment.

**The task directory is the interface.** A directory that `harbor run` accepts is
served here unchanged — no conversion, no re-authoring, no second copy of the
data. That covers Terminal-Bench-lineage tasks and everything
[Repo2RLEnv](https://github.com/huggingface/Repo2RLEnv) generates from a GitHub
repository, so a task set built for evaluation is also a training environment.

The reward always comes from the task's own verifier. This environment forwards
it and never computes one of its own.

## Quick start

```bash
# 1. Serve the bundled example task
uv run --project envs/harbor_env server

# 2. Solve it through the client
PYTHONPATH=src:envs uv run python envs/harbor_env/examples/quickstart.py
```

```python
import asyncio
from harbor_env import HarborEnv

async def main():
    env = HarborEnv(base_url="http://localhost:8000")
    start = await env.reset(task_id="fix-sum-bug")
    print(start.observation.instruction)          # the task's instruction.md

    await env.write_file("stats.py", my_fix)      # the agent works
    result = await env.evaluate()                 # the task's tests/test.sh grades it

    print(result.reward)                          # 1.0
    print(result.observation.info["reward_metrics"])
    await env.close()

asyncio.run(main())
```

Point it at your own tasks with `HARBOR_TASKS_DIR`, which accepts a local
directory or a Hugging Face dataset:

```bash
HARBOR_TASKS_DIR=./tasks                              uv run --project envs/harbor_env server
HARBOR_TASKS_DIR=hf://datasets/my-org/click-tasks     uv run --project envs/harbor_env server
HARBOR_TASKS_DIR=./tasks HARBOR_MODE=docker           uv run --project envs/harbor_env server
```

## Execution modes

| Mode | How it runs | Use it for |
|------|-------------|-----------|
| `docker` (default) | Inside the task's own image, files streamed over the Docker API | Everything — and the only backend that can enforce a task's network, resource and user policies |
| `local` | Subprocesses under a per-episode directory tree | Self-contained tasks on Docker-less hosts such as Hugging Face Spaces |

> **`local` mode is not a security boundary.** `exec` runs shell commands as
> the server's own user, so a task or a policy can read whatever that user can
> and can reach the network. It is opt-in for that reason.
>
> The server's environment is not passed through — only `PATH`, `HOME` and a
> short allowlist — which keeps API keys out of the common case. Treat that as
> hygiene, **not** containment: on Linux a child process can still recover the
> parent's environment through `/proc/<ppid>/environ`. If a policy you do not
> control can reach the socket, use `docker` mode.

`local` mode refuses any task whose `task.toml` declares a policy it cannot
enforce — `network_mode`, `cpus`/`memory_mb`, or an `[agent].user` — rather than
running it unconstrained and reporting a score the task never sanctioned.

`docker` mode needs the `docker` Python package. It ships in this environment's
own dependencies, so `uv run --project envs/harbor_env server` already has it;
add it explicitly when driving the environment from the repo root:

```bash
PYTHONPATH=src:envs uv run --with docker python envs/harbor_env/examples/validate_taskset.py --mode docker
```

A task is **self-contained** when it ships its starting files in `environment/`
and declares no `environment/Dockerfile` or `docker-compose.yaml`. Those run in
either mode. Anything else — including every `repo2rlenv` task with an
`environment/Dockerfile` — needs `docker`, and `local` mode says so explicitly
rather than grading an empty directory:

```
task 'pallets__click-2951' keeps its starting state in a container image
(environment/Dockerfile), which the local backend cannot reproduce.
Run the server with HARBOR_MODE=docker, or use a task that ships its files in environment/.
```

## Task policies

`task.toml` may constrain how a task runs. These are honoured in `docker` mode
and cause `local` mode to refuse the task:

| Declaration | Effect |
|---|---|
| `[environment].network_mode = "no-network"` | The container gets no network |
| `[environment].cpus` / `memory_mb` / `storage_mb` | Real container limits |
| `[environment].workdir` | Overrides the image's `WORKDIR` |
| `[agent].user` / `[verifier].user` | That phase runs as the named account |
| `network_mode = "allowlist"`, `gpus` | **Refused** — needs a filtering proxy / accelerators this backend does not provide. Grade with `harbor run`. |

A task whose phases disagree (say a `public` agent and a `no-network` verifier)
gets the **restrictive** setting for the whole episode: a container has one
network, and resolving the conflict permissively would hand a sandboxed
verifier the internet.

## The task format

```text
<task>/
├── task.toml              # config + metadata            (required)
├── instruction.md         # the prompt shown to the agent
├── environment/           # Dockerfile, compose file, or seed files
├── tests/
│   └── test.sh            # the verifier                 (required to grade)
└── solution/
    └── solve.sh           # the reference solution       (optional)
```

Both spellings of the schema field are read: Harbor writes `schema_version`,
Repo2RLEnv writes `version`. Harbor's special paths are honoured exactly:

| Path | Contents | When |
|------|----------|------|
| working directory | what the agent edits — the image's `WORKDIR` | for the whole episode |
| `/tests` | the task's `tests/` | staged just before the verifier runs |
| `/solution` | the task's `solution/` | staged only for `solve` |
| `/logs/verifier` | `reward.json` / `reward.txt` | written by the verifier |
| `/logs/agent` | scratch space for agent logs | for the whole episode |

## Rewards

The verifier writes its verdict into `/logs/verifier`; `reward.json` is read
first and `reward.txt` is the fallback, matching Harbor.

| Verifier wrote | `result.reward` | `observation.info["reward_source"]` |
|----------------|-----------------|--------------------------------------|
| `reward.json` with a `reward` key | that value | `reward.json` |
| `reward.json` with one metric | that value | `reward.json` |
| `reward.json` with several metrics, no `reward` key | falls back to `reward.txt` | `reward.txt` |
| `reward.txt` only | that value | `reward.txt` |
| nothing | `None`, `success=False` | `missing` |

Every numeric metric from `reward.json` is surfaced in
`observation.info["reward_metrics"]`, and Repo2RLEnv's nested
`reward-details.json` sidecar (F2P/P2P counts, diff-similarity components) in
`observation.info["reward_details"]`.

A verifier that writes no reward file yields **no reward at all**, not a `0.0`.
A fabricated zero is indistinguishable from a genuine failure and would quietly
poison a training run.

## Action space

`HarborAction.action_type` is one of:

| Action | Fields | Purpose |
|--------|--------|---------|
| `exec` | `command`, `timeout_s` | Run a shell command in the working directory |
| `read` | `path` | Read a file, relative to the working directory |
| `write` | `path`, `content` | Write a file, relative to the working directory |
| `evaluate` | — | Run `tests/test.sh` and end the episode |
| `solve` | — | Apply `solution/solve.sh` — Harbor's oracle agent |

`exec`, `read` and `write` are the agent's action space. `evaluate` and `solve`
are training-orchestration controls: they decide when an episode ends and can
hand over the answer, so they belong on the infrastructure side of OpenEnv's
[dual API boundary](https://github.com/huggingface/OpenEnv/blob/main/rfcs/001-architecture.md).
Set `HARBOR_ALLOW_CONTROL_ACTIONS=0` to have the server refuse them outright,
for deployments where a policy can reach the same socket.

`evaluate` ends the episode. Every later action is refused and keeps reporting
`done=True` — the episode never returns to a running state without `reset()`.

`read` and `write` are confined to the working directory, so an agent cannot
reach `/tests` or `/solution`. The verifier log directory is also cleared
immediately before the verifier runs, so a planted reward file cannot survive
into the score.

## Working with Repo2RLEnv

[Repo2RLEnv](https://github.com/huggingface/Repo2RLEnv) turns a GitHub
repository into verifiable Harbor tasks. The output needs no conversion:

```bash
# 1. Synthesize tasks from a repository
repo2rlenv generate --repo pallets/click --pipeline pr_runtime \
    --pipeline-opt limit=10 --llm anthropic/claude-sonnet-4-6 --out ./tasks

# 2. Check every task is solvable by its own oracle before training on it
PYTHONPATH=src:envs uv run --with docker python envs/harbor_env/examples/validate_taskset.py \
    --tasks ./tasks --mode docker

# 3. Serve them
HARBOR_TASKS_DIR=./tasks HARBOR_MODE=docker uv run --project envs/harbor_env server
```

Tasks published with `repo2rlenv push` live under `tasks/<id>/` in a Hugging Face
dataset; that layout is discovered automatically, so
`HARBOR_TASKS_DIR=hf://datasets/<org>/<name>` is enough.

Rewards line up with what the pipeline produced: `test_execution`
(`f2p_rate × p2p_rate`) for the `_runtime` pipelines and `diff_similarity` for
`pr_diff`, forwarded exactly as the verifier computed them.

## Writing a task that runs in both places

Harbor's absolute paths only exist inside a container. A verifier that prefers
the `HARBOR_*` variables and falls back to those paths runs unchanged under
`harbor run`, under `docker` mode, and under `local` mode:

```bash
#!/bin/bash
set -uo pipefail
LOGS_DIR="${HARBOR_LOGS_DIR:-/logs/verifier}"
TESTS_DIR="${HARBOR_TESTS_DIR:-/tests}"
WORKDIR="${HARBOR_WORKDIR:-$(pwd)}"

mkdir -p "$LOGS_DIR"
cd "$WORKDIR"
# ...grade, then write the reward...
echo '{"reward": 1.0}' > "$LOGS_DIR/reward.json"
exit 0   # the reward file is the verdict, not the exit code
```

The bundled `examples/tasks/fix-sum-bug` task does exactly this, and grades
identically across all three runtimes (verified against `harbor` 0.20.0):

| Runtime | Untouched task | After `solution/solve.sh` |
|---------|----------------|---------------------------|
| `harbor run -a nop` / `-a oracle` | 0.600 | 1.000 |
| `harbor_env` in `local` mode | 0.600 | 1.000 |
| `harbor_env` in `docker` mode | 0.600 | 1.000 |

Reproduce the first row with `harbor run -p envs/harbor_env/examples/tasks/fix-sum-bug -a nop`
and the other two with `examples/validate_taskset.py`.

A verifier that hardcodes `/logs/verifier` still works in `docker` mode; in
`local` mode the episode reports that it could not be scored and names the fix.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `HARBOR_TASKS_DIR` | bundled examples | Task directory, or `hf://datasets/<org>/<name>[@rev]` |
| `HARBOR_MODE` | `docker` | `docker` or `local` |
| `HARBOR_ALLOW_CONTROL_ACTIONS` | `1` | Accept `evaluate`/`solve`; set `0` for agent-facing deployments |
| `HARBOR_DEFAULT_TASK_ID` | — | Task used when `reset()` names none |
| `HARBOR_COMMAND_TIMEOUT_S` | `120` | Timeout for agent `exec` actions |
| `HARBOR_DEFAULT_IMAGE` | `python:3.12-slim` | Image for self-contained tasks in `docker` mode |
| `MAX_CONCURRENT_ENVS` | `8` | Concurrent WebSocket sessions |

Verifier and oracle timeouts come from `[verifier].timeout_sec` and
`[agent].timeout_sec` in each task's `task.toml`.

## Running in Docker

```bash
docker build -t harbor-env:latest -f envs/harbor_env/server/Dockerfile envs/harbor_env

# local mode, your own tasks
docker run --rm -p 8000:8000 -v "$PWD/tasks:/tasks:ro" -e HARBOR_TASKS_DIR=/tasks harbor-env:latest

# docker mode needs the host daemon, so tasks can start their own containers
docker run --rm -p 8000:8000 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD/tasks:/tasks:ro" \
    -e HARBOR_TASKS_DIR=/tasks -e HARBOR_MODE=docker \
    harbor-env:latest
```

Hugging Face Spaces cannot run Docker-in-Docker, so a Space serves `local` mode
and therefore self-contained tasks that declare no restrictive policies.

## Examples

| Script | What it shows |
|--------|---------------|
| [`examples/quickstart.py`](https://github.com/huggingface/OpenEnv/blob/main/envs/harbor_env/examples/quickstart.py) | A full episode against a running server |
| [`examples/validate_taskset.py`](https://github.com/huggingface/OpenEnv/blob/main/envs/harbor_env/examples/validate_taskset.py) | Oracle sweep over a task set — run this before training |
| [`examples/tasks/fix-sum-bug/`](https://github.com/huggingface/OpenEnv/tree/main/envs/harbor_env/examples/tasks/fix-sum-bug) | A Harbor task written to run under every runtime |

## Layout

```text
envs/harbor_env/
├── models.py                        # HarborAction / HarborObservation / HarborState
├── client.py                        # HarborEnv
├── examples/
└── server/
    ├── task.py                      # task.toml parsing + task discovery
    ├── sandbox.py                   # Harbor's sandbox contract; local + docker backends
    ├── reward.py                    # the reward-file contract
    ├── harbor_env_environment.py    # Gymnasium orchestration over the three above
    └── app.py                       # FastAPI server
```

## Tests

```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_harbor_env.py -v
```

They are hermetic — no Docker, no network — and use the bundled example task as
the end-to-end fixture.
