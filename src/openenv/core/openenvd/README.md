# openenvd: policy-scoped environment runtime

`openenvd` implements [RFC 009](../../../../rfcs/009-openenvd.md).

- **The daemon** hosts the agent, grader, orchestrator and observer surfaces outside the
  workload.
- **The environment** runs as a worker inside a fresh sandbox for each episode.
- **Sandboxes** come from a pluggable enforcement backend:
  - `openshell` ([NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)) is
    kernel-enforced and provides every guarantee.
  - `local` runs host subprocesses and provides none; it is for development.

## Guarantees and refusal

An environment lists the guarantees it depends on:

| Guarantee | Meaning | `openshell` | `local` |
| --- | --- | :-: | :-: |
| `asset_isolation` | Privileged assets are unreachable from the workload | ✓ | ✗ |
| `egress_control` | Workload egress is limited to the declared policy | ✓ | ✗ |
| `privilege_drop` | The workload runs unprivileged and cannot escalate | ✓ | ✗ |
| `control_plane_isolation` | The workload cannot reach privileged surfaces | ✓ | ✗ |

At startup the daemon refuses with `EnforcementUnavailable`, before creating any sandbox,
in two cases:

- the selected backend lacks a required guarantee;
- the backend's host prerequisites are missing (for OpenShell: the CLI, OpenSSH, a
  supported version and a reachable gateway).

It never falls back to another backend.

## Prepare an OpenShell gateway and image

Install the [OpenShell v0.1.2 CLI](https://github.com/NVIDIA/OpenShell/releases/tag/v0.1.2)
and OpenSSH (`ssh`) on the daemon host. openenvd accepts stable OpenShell `>=0.1.2,<0.2`.
Configure a reachable gateway using the
[OpenShell gateway guide](https://docs.nvidia.com/openshell/latest/how-it-works/gateways/overview)
and name it in the manifest. The gateway must support Landlock ABI 3 or newer. The daemon
itself needs no root, network administration capabilities or writable cgroups.

The sandbox image must contain:

- Python and `/bin/tar`;
- this OpenEnv implementation;
- the environment's factory and action classes;
- any workload processes you declare.

Install them under read-only paths such as `/usr/local` and `/opt`. `/sandbox` must be
writable by the workload identity (UID and GID `1000` by default). For example:

```dockerfile
FROM python:3.13-slim
WORKDIR /opt/openenv
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY envs/echo_env ./envs/echo_env
RUN test -x /bin/tar \
    && pip install --no-cache-dir -e . -e ./envs/echo_env \
    && groupadd --gid 1000 sandbox \
    && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /sandbox sandbox \
    && mkdir -p /sandbox \
    && chown 1000:1000 /sandbox
USER 1000:1000
WORKDIR /sandbox
```

Make the image available to the gateway, for example through a registry it can pull from.
Building an image locally does not make it available to a remote gateway. The daemon uses
its own gateway credentials. Those credentials and the principal tokens never reach the
workload, and automatic provider attachment is disabled.

## Configure and launch

```yaml
openenvd:
  enabled: true
  enforcement:
    backend: openshell
    require: [asset_isolation, egress_control, privilege_drop, control_plane_isolation]
  openshell:
    image: openenv-echo:openshell
    gateway: local
    workspace: default
  surfaces:
    orchestrator: { allow_lifecycle: true }
    agent: { tools: [echo_message] }
    grader:
      tools: [grader.read_file, grader.fs_diff, grader.get_trajectory]
      fs_read: ['/workspace/**', '/openenvd/assets/**']
    observer: { stream: [harness_events, fs_diff, process] }
```

The two workspace settings are different things:

- `openshell.workspace` selects the OpenShell gateway workspace.
- The `--workspace` argument is a local directory whose contents seed each episode. It may
  contain only regular files and directories. The daemon captures it once and uploads a
  copy to `/sandbox/workspace` in every fresh sandbox; workload changes never modify it.

Keep privileged assets in a separate local directory, outside the seed.

Permissions are explicit allowlists, and anything omitted is denied:

- Agent patterns accept a literal name or a trailing `*`. They can't include lifecycle or
  `grader.*` tools.
- Observers get streams only.
- `fs_read` applies only to graders.

Give each configured privileged principal a distinct, high-entropy token:
`OPENENVD_ORCHESTRATOR_TOKEN`, `OPENENVD_GRADER_TOKEN` and `OPENENVD_OBSERVER_TOKEN`. Then
launch:

```bash
mkdir -p ./workspace-seed ./private-assets
chmod 700 ./private-assets
python -m openenv.core.openenvd \
  --manifest envs/echo_env/openenv.yaml \
  --factory echo_env.server.echo_environment:EchoEnvironment \
  --action-class openenv.core.env_server.mcp_types:CallToolAction \
  --workspace ./workspace-seed \
  --asset-root ./private-assets \
  --timeout 300
```

**Factory.** An explicit `--factory` builds the environment without arguments. For standard
apps built with `create_app`, omit `--factory` and the daemon discovers the factory and
action class by importing the manifest's app on the daemon host. Those modules must also be
importable inside the sandbox image.

**Arguments.**
- `--manifest` is required. If the `openenvd` section is absent or disabled, the manifest's
  original app runs unchanged.
- When it's enabled, `--workspace` and `--asset-root` are also required.
- `--host` defaults to `127.0.0.1` and `--port` to `8100`.

**Exposure.** Deployments that bind another address must supply TLS and network access
controls. The agent endpoint has no credentials of its own.

### Developing without a gateway

Set `enforcement.backend: local` and leave `require` empty. The daemon then runs the worker
and processes as host subprocesses on a private copy of the seed. This works for building
graders and harness wiring, but enforces nothing: the workload runs as your user. A manifest
that requires any guarantee refuses to start under `local`.

## Sandbox policy

By default the OpenShell policy is rendered from the backend-neutral `workload` and
`egress` declarations:

```yaml
openenvd:
  workload:
    read_only: [/bin, /usr, /lib, /lib64, /etc, /proc, /opt, /dev/urandom]   # default
    read_write: [/sandbox, /tmp, /dev/null]                                   # default
  egress:
    mode: allowlist          # default: none
    allow:
      - { host: api.anthropic.com, binaries: [/usr/local/bin/node] }
      - { host: pypi.org }
```

Rendering adds the following:

- `landlock.compatibility: hard_requirement`;
- the process identity from `openshell.run_as_user` and `run_as_group` (default `1000`);
- `include_workdir: false`;
- one TCP endpoint per egress rule.

A rule with `binaries` only applies to those programs. For example, only the harness may
reach its model API, while code the agent runs reaches nothing.

**Invariants.**
- Writable paths must be inside `/sandbox` or `/tmp`, or be exactly `/dev/null`.
- The worker interpreter (`openshell.python`, default `/usr/local/bin/python3`) must not be
  writable.
- Don't allow egress to the daemon's own address.

To use OpenShell's full schema (L7 rules, protocols, middlewares), put a complete native
policy under `openshell.policy` instead of `workload` and `egress`. It must keep the same
invariants: `hard_requirement`, positive numeric `run_as_user`/`run_as_group`,
`include_workdir: false`, and confined writable paths.

After creating each sandbox, the backend checks that the gateway admitted exactly the
requested policy. The only exception is OpenShell's documented baseline paths, which are
accepted when network rules are present.

## Principal surfaces and episode lifecycle

| Surface | Transport | Access |
| --- | --- | --- |
| Agent | HTTP POST or WebSocket `/mcp` | Allowed environment tools |
| Grader | HTTP POST or WebSocket `/mcp/grader` | Grader bearer token; allowed tools and reads |
| Orchestrator | WebSocket `/ws` | Orchestrator token and `allow_lifecycle`; reset, step, state, close |
| Observer | WebSocket `/observe` | Observer bearer token; allowed streams |
| Health | HTTP GET `/health` | Minimal unauthenticated status |

**Worker control.** The daemon controls the worker over the sandbox's stdio. The channel is
protected before any environment code is imported.

**Agent listener.** When an agent policy is configured, the sandbox also exposes an
agent-only `/mcp` listener at `127.0.0.1:8000` for local harnesses. It has no privileged
routes. Creating or closing an MCP session does not reset the episode.

**Reset.**
1. Stop the current sandbox and confirm it has been deleted.
2. Start a fresh sandbox from the captured seed.
3. Spawn the worker, then every workload process.

If cleanup fails, the next episode does not start. A worker exit or timeout also tears down
the sandbox; nothing restarts automatically. Only the authenticated orchestrator controls the
lifecycle.

**Crash backstop.** OpenShell sandboxes use ephemeral retention and a finite main process
lasting `4 × --timeout + 60` seconds, which bounds cleanup after a daemon crash. OpenShell
0.1 has no sandbox TTL.

## Processes

Declare processes beyond the environment worker by trust level:

```yaml
openenvd:
  privileged_assets:
    oracle: grade.sh
    rubric: rubric.py
  processes:
    harness:
      trust: workload
      argv: [/usr/local/bin/my-harness, --mcp, http://127.0.0.1:8000/mcp]
      env: { HARNESS_MODE: train }
    rubric:
      trust: privileged
      asset: rubric
      args: [--strict]
```

**Workload processes**
- Run inside the episode sandbox alongside the worker, under the same policy and identity.
- Are started after the worker on every reset, and stopped with the sandbox.
- Have their output drained and never retained.
- Record their exits as observer `process` events.

They can see everything the worker sees, so nothing they share a sandbox with should be worth
stealing or forging.

**Privileged processes**
- Run on demand through `grader.run_process` with `{"name": ...}`. This needs
  `allow_privileged_exec: true` and the tool in the grader's allowlist.
- Each run gets a new sandbox holding a workspace snapshot and the private assets, staged in
  an unguessable directory.
- The run executes the declared asset with its declared arguments; callers can't supply a
  command.
- Output is returned to the grader, nothing is copied back to the agent, and the sandbox is
  deleted.

An `oracle` asset is an implicit privileged process, also reachable as `grader.run_oracle`.

## Grading and observations

The grader tools are:

- `grader.read_file`
- `grader.fs_diff`
- `grader.get_full_state`
- `grader.get_trajectory`
- `grader.run_process`
- `grader.run_oracle`

Each needs an explicit allowlist entry.

**Snapshots.**
- Workspace reads and diffs use downloaded snapshots, addressed as `/workspace/...`.
- File reads also need `fs_read` permission, and accept regular UTF-8 files up to 1 MiB.
- Exports are capped at 64 MiB and 4096 entries, and are validated before anything is
  written on the host.
- Snapshots are not atomic. After a forced teardown, workspace grading is unavailable until
  the next reset.

**Observer streams.** Supported streams are `harness_events`, `fs_diff` and `process`.

- Harness events are reported by the workload itself.
- Process events cover the worker, workload processes and the sandbox lifecycle.
- Filesystem sampling can miss transient changes.
- `network` and `resource` streams are rejected until backend telemetry is integrated.

Harness adapters can publish events through
[`MCPHarnessAdapter(event_sink=HarnessEventSink())`](../harness/README.md#openenvd-event-publication).

### Connect from orchestration and grading code

```python
import os

from openenv.core.generic_client import GenericEnvClient
from openenv.core.openenvd import GraderClient, observer_stream


async def reset_episode():
    async with GenericEnvClient(
        base_url="http://127.0.0.1:8100",
        headers={"Authorization": f"Bearer {os.environ['OPENENVD_ORCHESTRATOR_TOKEN']}"},
    ) as env:
        return await env.reset()


async def grade():
    async with GraderClient(
        "http://127.0.0.1:8100/mcp/grader", os.environ["OPENENVD_GRADER_TOKEN"]
    ) as grader:
        return await grader.call_tool("grader.run_process", {"name": "rubric"})


async def monitor():
    async for event in observer_stream(
        "ws://127.0.0.1:8100/observe", os.environ["OPENENVD_OBSERVER_TOKEN"]
    ):
        print(event["type"], event["data"])
```

These clients don't compute rewards; environment-side graders and rubrics keep that
responsibility. RFC 008 validation preserves the policy under `manifest.openenvd` in
`openenv validate --json`, and an invalid policy produces a `static.manifest` failure.

## Verification

```bash
PYTHONPATH=src:envs uv run pytest tests/core/test_openenvd -q
```

The portable tests use two stand-ins:

- a simulated gateway, for the OpenShell backend;
- the real local backend, for a full episode with a worker, a workload process and
  privileged processes.

They don't verify a live OpenShell deployment. The live tests need a gateway and an image
built with this implementation and the echo environment:

```bash
OPENSHELL_TEST_GATEWAY=local \
OPENSHELL_TEST_IMAGE=openenv-echo:openshell \
PYTHONPATH=src:envs uv run pytest tests/core/test_openenvd -q
```

Without those variables the live tests skip.
