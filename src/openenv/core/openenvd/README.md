# openenvd: policy-scoped environment runtime

`openenvd` provides an opt-in runtime for [RFC 009](../../../../rfcs/009-openenvd.md):
one environment workload, a dedicated episode workspace, and separate agent,
grader, orchestrator, and observer surfaces.

## Environment runtime

The runtime requires Linux root, permission to create named network namespaces,
veth interfaces and nftables rules, and writable cgroup v2 supporting
`cgroup.kill`. Startup fails if these requirements cannot be met. Reserve a
dedicated nonzero UID/GID for the workload. The outer container remains responsible
for host containment and resource limits. Install OpenEnv and the environment's
dependencies in the daemon's interpreter. Environment factory code is trusted;
its untrusted subprocesses run under the workload identity. Children receive a
minimal environment without daemon secrets.

The base container image includes `iproute2`, `nftables`, `conntrack`,
`libseccomp2`, and `libnetfilter-log1`. Source installations need those system
packages too (`libnetfilter-log1` is required when network observation is enabled).
Enable IPv4 forwarding in the daemon's network namespace (for Docker, `--sysctl net.ipv4.ip_forward=1`).
The runtime checks this setting without changing host-wide sysctls. The kernel
must support veth, nftables NAT/conntrack, and NFLOG when network observation is
enabled. Python manages these facilities; no additional language toolchain or
userspace TCP/IP stack is required. Existing host firewall rules may further
restrict allowed traffic; openenvd never flushes or replaces them.

Prepare an existing, dedicated workspace writable by the workload UID/GID, with
parent paths it can traverse. The immediate parent must be daemon-owned and
not writable by group or others, so the workload cannot rename its workspace
during restoration. Reset restores the startup snapshot and preserves the
original UID/GID of each path. Initial workspace permissions must already allow
the intended workload writes. Do not use a shared checkout
or a directory containing privileged assets. Keep asset sources outside the
workspace under a daemon-owned directory with mode `0700`; protect the originals
as well as the daemon's private copies.

Add a policy section to the environment's `openenv.yaml`, for example:

```yaml
openenvd:
  enabled: true
  surfaces:
    orchestrator:
      allow_lifecycle: true
    agent:
      tools: [echo_message]
    grader:
      tools: [grader.read_file, grader.fs_diff, grader.get_trajectory]
      fs_read: ['/workspace/**', '/openenvd/assets/**']
    observer:
      stream: [harness_events, fs_diff, process, resource]
  privileged_assets:
    solution: solution.txt
```

Replace the agent tool names with the environment's tools. Permissions are
allowlists; omitted permissions are denied. Agent tool patterns accept a literal
name or trailing `*`, and cannot include lifecycle or `grader.*` tools. Observers
receive streams only. Asset source paths are relative to `--asset-root`; graders
address copies as `/openenvd/assets/<name>`. `fs_read` is supported only for graders; declarations for other principals are
rejected. Agent tools access files under the workload's OS permissions, which also
protect privileged assets.

Supply distinct, high-entropy secrets through deployment configuration:
`OPENENVD_ORCHESTRATOR_TOKEN`, `OPENENVD_GRADER_TOKEN`, and
`OPENENVD_OBSERVER_TOKEN` for each configured privileged principal. Launch with:

```bash
python -m openenv.core.openenvd \
  --manifest /opt/environment/openenv.yaml \
  --workspace /workspace \
  --asset-root /opt/private-assets \
  --uid 65536 --gid 65536 \
  --timeout 300
```

For standard apps built with `create_app`, the runtime discovers the environment
factory and action class from the manifest's app entry. Alternatively pass
`--factory my_environment.server.environment:MyEnvironment` and, when needed,
`--action-class module:Class`. An explicit factory constructs an environment
without arguments. The manifest directory is available for worker imports;
install the environment package when it is not importable from that directory.
The worker does not inherit the daemon's ambient `PYTHONPATH`.

`--cgroup-root` selects a delegated writable cgroup v2 root (default
`/sys/fs/cgroup`); the runtime creates its workload cgroup beneath it.
`--host` defaults to `127.0.0.1` and `--port` to `8100`. Exposing another address
requires deployment-provided TLS and network access controls. The agent endpoint
has no additional credentials.

`--manifest` is required. If its `openenvd` section is absent or disabled, the
CLI runs the manifest's original app unchanged; workspace, asset-root, and
workload identity arguments are only required when `openenvd` is enabled.

| Surface | Transport | Access |
| --- | --- | --- |
| Agent | HTTP POST or WebSocket `/mcp` | Allowed environment tools |
| Grader | HTTP POST or WebSocket `/mcp/grader` | Grader bearer token; allowed tools and reads |
| Orchestrator | WebSocket `/ws` | Orchestrator token and `allow_lifecycle`; reset, step, state, close |
| Observer | WebSocket `/observe` | Observer bearer token; allowed streams |
| Health | HTTP GET `/health` | Minimal unauthenticated status |

When an agent policy is configured, an agent-only HTTP/WebSocket `/mcp`
listener inside the workload's network namespace runs at `127.0.0.1:8000` for local harnesses. It exposes no privileged
routes. HTTP MCP logical session creation and closure do not reset the episode;
only the authenticated orchestrator controls reset.

The runtime enables Linux child-subreaper behavior and reaps adopted workload
children. Reset kills the workload cgroup, restores the startup workspace snapshot, clears
episode observations, and starts a fresh environment before forwarding reset
arguments. Workload exit and timeout trigger cgroup cleanup. Reset covers the
dedicated workspace, not every writable container path. Workload state that
could outlive `cgroup.kill` and workspace restore is prevented before executing
workload code. `libseccomp` compiles a named syscall denylist for SysV IPC, POSIX
message queues, keyrings, and mounting, including rejection of unsupported ABIs.
Private IPC and mount namespaces contain IPC resources; root-only tmpfs mounts
over `/dev/shm` and `/dev/mqueue` deny direct filesystem-backed IPC access without
changing permissions on the daemon's mounts. `/dev/fuse` is hidden by a private
bind of `/dev/null`. Failed isolation setup prevents workload execution. Teardown
kills the cgroup and lets the kernel reclaim its namespaces; it does not sweep
shared host IPC objects or keyrings by UID. Reserve a dedicated UID and do not run
other, unsandboxed processes under it. If the cgroup will not empty, reset fails closed with
`teardown incomplete: workload cgroup did not empty; workspace restore refused`.

Grader tools are `grader.read_file`, `grader.fs_diff`, `grader.get_full_state`,
`grader.get_trajectory`, and `grader.run_oracle`. Each requires an explicit tool
allowlist entry. File reads require `fs_read` permission and accept regular UTF-8
files up to 1 MiB under managed roots, rejecting symlinks. Oracle execution also
requires `allow_privileged_exec: true` and an executable asset named `oracle`.
The oracle is trusted operator code running with daemon privileges and a minimal
environment; callers cannot supply an arbitrary command.

Programmatic entry points are `Principal`, `SurfacePolicy`, `OpenEnvDConfig`,
`Runtime`, and `create_surface_app` from `openenv.core.openenvd`. Client exports
load without importing server implementations. Harness adapters
can publish through
[`MCPHarnessAdapter(event_sink=HarnessEventSink())`](../harness/README.md#openenvd-event-publication).

### Connect from orchestration and grading code

Pass the orchestrator credential using the optional `EnvClient.headers`
parameter, also supported by `GenericEnvClient`:

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

Use the separate public client for grader MCP calls. `call_tool` returns the
MCP result envelope; `run_oracle` decodes its JSON text into the oracle result.
The latter requires the oracle permission and asset described above, which are
not enabled in the minimal policy example.

```python
import os

from openenv.core.openenvd import GraderClient, observer_stream

async def grade():
    async with GraderClient(
        "http://127.0.0.1:8100/mcp/grader",
        os.environ["OPENENVD_GRADER_TOKEN"],
    ) as grader:
        tools = await grader.list_tools()
        diff = await grader.call_tool("grader.fs_diff", {"since": "reset"})
        oracle = await grader.run_oracle()
        return tools, diff, oracle

async def monitor():
    async for event in observer_stream(
        "ws://127.0.0.1:8100/observe",
        os.environ["OPENENVD_OBSERVER_TOKEN"],
    ):
        print(event["type"], event["data"])
```

The observer client yields decoded JSON dictionaries with `seq`, `ts`, `type`,
and `data`. Sequence numbers restart on episode reset. These clients do not
compute rewards; environment-side graders or rubrics retain that responsibility.

### Observation and implementation limits

Typed episode events (`harness_event`, `fs_change`, `process`, `network`,
`resource`) are retained in bounded daemon memory, not a durable audit store.
Harness events are workload reports, not proof of OS operations. Filesystem,
process, and resource sampling can miss transient changes. Continuous filesystem,
process, and resource samples are collected only for explicitly configured observer
streams. Filesystem hashing runs outside the action/reset lock; samples spanning
a reset are discarded. The startup snapshot remains available for reset and
on-demand grader diffs. Snapshot capture copies files and records ownership
without hashing. Baseline fingerprints are computed lazily from the immutable
snapshot when filesystem observation or a grader diff first needs them. Workspace
scans are not atomic. File fingerprints hash complete files up to 64 MiB; oversized files
cause observation to fail explicitly rather than using a partial fingerprint.
Resource samples include CPU, memory, and workspace `disk_bytes`.

Each episode gets a private network namespace with loopback for the agent MCP
listener and a veth interface for external traffic. Python installs nftables
rules on the daemon side before enabling either interface. The Linux kernel
handles routing, TCP/UDP, connection tracking, and source NAT. All egress is denied
unless explicitly allowed in the manifest, for example:

```yaml
openenvd:
  enabled: true
  network:
    allow:
      - cidr: 1.1.1.1/32
        protocol: udp
        ports: [53]
      - cidr: 1.1.1.1/32
        protocol: tcp
        ports: [53, 443]
```

Rules require canonical IPv4 CIDRs, a protocol (`tcp` or `udp`), and explicit
ports. Loopback, private, link-local/metadata, shared-address, multicast, reserved
ranges, and all daemon-local addresses remain denied even under a broad allow
rule. IPv6 cannot leave the workload. DNS requires its own resolver allowances;
the runtime does not rewrite resolver configuration or implement hostname rules.
Allowing an external proxy authorizes that proxy's service, including any onward
access it provides; destination rules do not inspect application payloads.

Adding `network` to observer streams enables NFLOG packet-policy events, decoded
by the system `libnetfilter_log` library, from
before workload initialization. Events contain destination, protocol, and an
`allow` or `deny` outcome. An allowance records a firewall decision, not successful
upstream connection establishment. Retransmissions may produce repeated events;
established packets are not individually logged. Namespace-local loopback and
application payloads are not retained. No `strace` or ptrace permission is required.
Socket overruns, detected sequence gaps, or observation buffering failures stop
the episode rather than silently discard observations.

A Python supervisor owns the network lifecycle and detects daemon exit through a
private pipe. Reset and close disconnect the veth, remove that episode's conntrack
entries, and delete its firewall table and namespace before another episode starts.
One daemon-owned, atomically updated record under `/run/openenvd-networks`
serves as both the address reservation and cleanup journal. It supports cleanup
after supervisor death; incomplete cleanup
prevents reset. Episode addresses are allocated from unused /30 subnets within
198.18.0.0/15. Allocation and cleanup are serialized to prevent stale connection
state from surviving address reuse. Kernel connection-tracking capacity and the
outer container's resource limits apply to allowed traffic.

RFC 008 validation includes the top-level `openenvd` policy in its normalized
manifest. `openenv validate --json` preserves it under `manifest.openenvd`, and
invalid policies produce a `static.manifest` failure. Contract graders do not yet
use openenvd's runtime or privileged grader surface. Deployment still requires
explicit workspace, asset-root, and identity CLI arguments; there is no YAML-only
startup. A manifest with `openenvd` absent or disabled runs the original app.
These observation and deployment limits mean this implementation does not yet
provide every guarantee in RFC 009. The runtime does not automatically restart crashed workloads;
a subsequent reset starts a fresh workload. Harness event publication remains opt-in and keeps adapter
buffers; separate harness process delegation and `env.trajectory` integration
are not provided. `GraderClient.run_oracle()` is a reference consumer, not an
integration into a normalized RFC 008 contract-grader runner.

### Verification

```bash
PYTHONPATH=src:envs uv run pytest tests/core/test_openenvd*.py -q
```

The `openenvd.yml` CI workflow runs on native Linux as root with an `/opt` virtual
environment and the installed echo environment package. A mandatory capability
probe verifies UID separation, network namespaces and writable cgroup v2 before
running tests; portable tests alone do not validate those OS boundaries.
