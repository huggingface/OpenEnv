# openenvd: policy-scoped environment runtime

`openenvd` implements the opt-in runtime for [RFC 009](../../../../rfcs/009-openenvd.md)
using NVIDIA OpenShell for workload isolation. The daemon keeps the agent, grader,
orchestrator, and observer surfaces outside the sandbox. Each episode runs one
environment in a fresh OpenShell sandbox.

## Prepare the gateway and image

Install the [OpenShell v0.1.2 CLI](https://github.com/NVIDIA/OpenShell/releases/tag/v0.1.2)
and OpenSSH (`ssh`) on the daemon host. The adapter accepts stable OpenShell
versions `>=0.1.2,<0.2`. Configure a reachable gateway using the
[OpenShell gateway guide](https://docs.nvidia.com/openshell/latest/how-it-works/gateways/overview),
and select its name explicitly in the manifest. The gateway must support the
policy's required Landlock enforcement. The daemon itself does not need Linux
root privileges, network administration capabilities, or a writable cgroup tree.

The sandbox image must contain Python, `/bin/tar`, this OpenEnv implementation,
and the environment factory and action classes. Install them under read-only
paths such as `/usr/local` and `/opt`; `/sandbox` must be writable by the configured process
identity (UID and GID `1000` by default). A minimal echo environment image can be
built from the repository root with this `Dockerfile.openshell`:

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

```bash
docker build -f Dockerfile.openshell -t openenv-echo:openshell .
```

Make the image available to the selected gateway, for example through a registry
it can pull from, and use that image reference in `openenvd.openshell.image`.
Building an image locally does not by itself make it available to a remote gateway.
The daemon uses its own OpenShell gateway credentials; those credentials and
principal tokens are not forwarded to the workload. Automatic provider attachment
is disabled.

## Configure and launch

Add the following section to the environment's `openenv.yaml`:

```yaml
openenvd:
  enabled: true
  openshell:
    image: openenv-echo:openshell
    gateway: local
    workspace: default
  surfaces:
    orchestrator:
      allow_lifecycle: true
    agent:
      tools: [echo_message]
    grader:
      tools: [grader.read_file, grader.fs_diff, grader.get_trajectory]
      fs_read: ['/workspace/**', '/openenvd/assets/**']
    observer:
      stream: [harness_events, fs_diff, process]
```

`openshell.workspace` selects the OpenShell gateway workspace. The separate
`--workspace` argument selects a local directory whose initial contents seed
each episode. It may contain only regular files and directories. The daemon
captures this seed once and uploads a copy to `/sandbox/workspace` in each fresh
sandbox; workload changes never modify the local seed. Keep privileged assets
in a separate local directory, outside the seed.

Replace agent tool names with the environment's tools. Permissions are explicit
allowlists; omitted permissions are denied. Agent patterns accept a literal name
or trailing `*` and cannot include lifecycle or `grader.*` tools. Observers expose
streams only, and `fs_read` permissions apply only to graders.

Supply distinct, high-entropy credentials through deployment configuration for
each configured privileged principal: `OPENENVD_ORCHESTRATOR_TOKEN`,
`OPENENVD_GRADER_TOKEN`, and `OPENENVD_OBSERVER_TOKEN`. For the echo image above,
launch with installed, fully qualified package names:

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

An explicit factory constructs the environment without arguments. For standard
apps built with `create_app`, omitting `--factory` discovers the factory and
action class by importing the manifest's app on the daemon host. Those discovered
module names must also be importable from the installed sandbox image. The host
manifest directory and ambient `PYTHONPATH` are not copied into the sandbox.

`--manifest` is required. An absent or disabled `openenvd` section runs the
manifest's original app unchanged. Enabled startup additionally requires
`openenvd.openshell`, `--workspace`, and `--asset-root`. The former `--uid`,
`--gid`, and `--cgroup-root` flags are replaced by OpenShell policy and gateway
configuration. `--host` defaults to `127.0.0.1`, and `--port` defaults to `8100`.
Deployments that expose another address must supply TLS and network access
controls; the agent endpoint has no additional credentials.

## Native sandbox policy

Omitting `openshell.policy` uses the following OpenShell policy:

```yaml
version: 1
filesystem_policy:
  include_workdir: false
  read_only: [/bin, /usr, /lib, /lib64, /etc, /proc, /opt, /dev/urandom]
  read_write: [/sandbox, /tmp, /dev/null]
landlock:
  compatibility: hard_requirement
process:
  run_as_user: '1000'
  run_as_group: '1000'
network_policies: {}
network_middlewares: {}
```

To customize it, put a complete native policy mapping under `openshell.policy`.
Network destinations, binaries, and protocols use
[OpenShell's policy schema](https://docs.nvidia.com/openshell/latest/how-it-works/policies/overview);
there is no translation from the former `openenvd.network` CIDR allowlist.
The default has no egress allowances. OpenShell validates the native policy, and
openenvd checks the effective policy before running the environment.

OpenEnv requires `landlock.compatibility: hard_requirement`, explicit positive
numeric user and group IDs, and `include_workdir: false`. Writable paths must be
inside `/sandbox` or `/tmp`, or exactly `/dev/null`. The Python interpreter
(default `/usr/local/bin/python3`, configurable as `openshell.python`) must be an
absolute path outside writable paths. Keep installed runtime and environment
code read-only.

## Principal surfaces and episode lifecycle

| Surface | Transport | Access |
| --- | --- | --- |
| Agent | HTTP POST or WebSocket `/mcp` | Allowed environment tools |
| Grader | HTTP POST or WebSocket `/mcp/grader` | Grader bearer token; allowed tools and reads |
| Orchestrator | WebSocket `/ws` | Orchestrator token and `allow_lifecycle`; reset, step, state, close |
| Observer | WebSocket `/observe` | Observer bearer token; allowed streams |
| Health | HTTP GET `/health` | Minimal unauthenticated status |

The daemon controls the worker over an authenticated SSH connection with a
private standard-input/output channel protected before environment imports.
When an agent policy is configured, the sandbox also provides an agent-only
HTTP/WebSocket `/mcp` listener at `127.0.0.1:8000` for local harnesses. It has no
privileged routes. MCP session creation and closure do not reset the episode.

Reset stops the current worker and requires confirmed deletion of its OpenShell
sandbox before starting a fresh sandbox from the captured seed. Cleanup failure
prevents the next episode. Workload exit and timeout also trigger sandbox
cleanup; there is no automatic workload restart. Only the authenticated
orchestrator controls lifecycle operations.

Sandboxes use OpenShell's ephemeral retention and a finite main process lasting
`4 × --timeout + 60` seconds. Main-process exit provides a best-effort cleanup
backstop after daemon crashes; OpenShell 0.1 does not enforce a sandbox TTL.

## Grading and observations

Grader tools are `grader.read_file`, `grader.fs_diff`, `grader.get_full_state`,
`grader.get_trajectory`, and `grader.run_oracle`. Each needs an explicit tool
allowlist entry. Workspace reads and diffs use downloaded snapshots, addressed
with logical paths under `/workspace`; the live sandbox path is
`/sandbox/workspace`. File reads additionally require `fs_read` permission and
accept regular UTF-8 files up to 1 MiB. Symlinks and special files are rejected.
Snapshots are exported over authenticated SSH with limits of 64 MiB and 4096
entries, and archive entries are checked before they are written on the host.
Snapshot exports are not atomic and may span workload writes.
Forced teardown without a final snapshot leaves workspace grading unavailable
until reset.

Declare private grading inputs with `privileged_assets`, for example:

```yaml
openenvd:
  # Keep the enabled, openshell, and surfaces sections shown above.
  privileged_assets:
    solution: solution.txt
    oracle: grade.sh
```

Asset sources are relative to `--asset-root`. The daemon keeps private copies
addressable by permitted graders as `/openenvd/assets/<name>`; assets are never
uploaded to the agent sandbox. Oracle execution also requires
`allow_privileged_exec: true` on the grader policy, the `grader.run_oracle` tool
allowance, and an executable asset named `oracle`. The oracle runs in a separate,
short-lived OpenShell sandbox using the configured image and policy. It receives
a downloaded workspace snapshot and the private grading assets, staged together
in a private asset directory inside that grader sandbox. Oracle output is
returned to the grader, and its writes are not copied back to the agent. The
grader sandbox is deleted after execution. Callers cannot supply an arbitrary
command.

Supported observer streams are `harness_events`, `fs_diff`, and `process`.
Harness events are workload self-reports, and process events describe worker and
sandbox lifecycle rather than all processes inside the sandbox. Filesystem
sampling can miss transient changes; downloaded snapshots are not atomic.
Fingerprints hash complete files up to 64 MiB and reject oversized files.
Events are retained in bounded daemon memory, and sequence numbers restart on
reset. `network` and `resource` observer streams are rejected until OpenShell
telemetry is integrated.

Harness adapters can publish through
[`MCPHarnessAdapter(event_sink=HarnessEventSink())`](../harness/README.md#openenvd-event-publication).
Public entry points include `Principal`, `SurfacePolicy`, `OpenShellConfig`,
`OpenEnvDConfig`, `Runtime`, and `create_surface_app`. Clients load without
importing server implementation modules.

### Connect from orchestration and grading code

`EnvClient` and `GenericEnvClient` accept optional authentication headers:

```python
import os

from openenv.core.generic_client import GenericEnvClient

async def reset_episode():
    async with GenericEnvClient(
        base_url="http://127.0.0.1:8100",
        headers={
            "Authorization": f"Bearer {os.environ['OPENENVD_ORCHESTRATOR_TOKEN']}"
        },
    ) as env:
        return await env.reset()
```

`GraderClient.call_tool` returns the MCP result envelope. `run_oracle` decodes
its JSON text into the oracle result and requires the optional oracle permissions
and asset described above.

```python
import os

from openenv.core.openenvd import GraderClient, observer_stream

async def grade():
    async with GraderClient(
        "http://127.0.0.1:8100/mcp/grader",
        os.environ["OPENENVD_GRADER_TOKEN"],
    ) as grader:
        return await grader.call_tool("grader.fs_diff", {"since": "reset"})

async def monitor():
    async for event in observer_stream(
        "ws://127.0.0.1:8100/observe",
        os.environ["OPENENVD_OBSERVER_TOKEN"],
    ):
        print(event["type"], event["data"])
```

Observer events contain `seq`, `ts`, `type`, and `data`. These clients do not
compute rewards; environment-side graders and rubrics retain that responsibility.

RFC 008 validation preserves the policy under `manifest.openenvd` in
`openenv validate --json`; invalid policies produce a `static.manifest` failure.
Its contract graders do not yet use this runtime or the privileged grader
surface. Separate harness process delegation and `env.trajectory` integration
are also not provided.

## Verification

Run the portable unit and contract tests with:

```bash
PYTHONPATH=src:envs uv run pytest tests/core/test_openenvd*.py -q
```

These tests exercise configuration, sandbox lifecycle commands, worker
transport, principal boundaries, and grading with a simulated gateway. They do
not verify a live OpenShell deployment. Optional live integration tests require
an available gateway and an image built with this implementation and the echo
environment:

```bash
OPENSHELL_TEST_GATEWAY=local \
OPENSHELL_TEST_IMAGE=openenv-echo:openshell \
PYTHONPATH=src:envs uv run pytest tests/core/test_openenvd*.py -q
```

Without those environment variables, live tests skip. Passing portable tests
alone does not establish that the gateway, image, and host support the required
isolation policy.
