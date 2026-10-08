# RFC: openenvd — Policy-Scoped Surfaces with Pluggable Enforcement

**Status**: Draft
**Created**: 2026-08-04 (revised 2026-10-08)
**Authors**: @zkwentz (idea credit: Adithya S Kolavi)
**RFC ID**: 009
**Issue**: [#1053](https://github.com/huggingface/OpenEnv/issues/1053)

## Summary

OpenEnv distinguishes two principals: the **agent** (MCP tools) and the **infrastructure**
(Gym-like `reset`/`step`/`state`). Everything else (agentic graders, oracle replay, harness
observation, telemetry) is hand-rolled per environment, usually by carefully *not* exposing a
port.

**openenvd** names four principals (orchestrator, agent, grader, observer) and gives each a
surface bound to a declared policy. It splits into two layers:

1. **The contract**, owned by OpenEnv and declared in `openenv.yaml`: surfaces, privileged
   assets, the workload's filesystem and egress, the processes that run, and the
   **guarantees** the environment depends on.
2. **Enforcement backends**, which are pluggable. A backend supplies sandboxes as primitives
   (create, spawn a process, download the workspace, delete) and declares which guarantees it
   makes structurally true. If the selected backend cannot provide every required guarantee,
   openenvd **refuses to start** instead of running with weaker isolation than the manifest
   claims.

The openenvd daemon runs on the trusted side and hosts every privileged surface. The
environment runs as a worker inside a fresh sandbox per episode. Oracles and other privileged
processes run in their own short-lived sandboxes. Two backends ship: `openshell`
([NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell), kernel-enforced, provides every
guarantee) and `local` (host subprocesses, provides none, for development).

openenvd is opt-in. Without an enabled `openenvd:` block nothing changes.

## Motivation

### Problem 1: Privilege separation is enforced by convention

RFC 005's harness security section is a list of things every adapter must remember: don't expose
the orchestration port in the container's network namespace, `unshare --net` outside Docker,
never inject reward tools. "Agents cannot reset" (RFC 001, INVARIANTS.md) deserves a structural
home.

### Problem 2: Graders need affordances the agent must never have

RFC 008's contract graders need to replay the oracle, read the solution to prove the agent can't,
and observe egress. Agentic graders need read-anything, filesystem diffs and state inspection.
Today the only tool surface is the agent's, so these either leak to the agent or become one-off
side channels.

### Problem 3: Observation happens inside the observed trust zone

RFC 005 collects `HarnessEvent`s in the adapter, next to the process being observed, which can
tamper with its own trace.

### Problem 4: In-container enforcement degrades without anyone noticing

The first draft of this RFC enforced isolation inside the environment container with UIDs,
loopback ports and namespaces "where supported". Review showed the gaps:

- Namespaces need `CAP_SYS_ADMIN`, which no OpenEnv provider can request.
- On HF Spaces the daemon and the workload end up as the same user.
- Container-wide reset kills every concurrent session.

An environment could declare openenvd, be deployed where none of that holds, and still claim its
solution was out of reach. Isolation claims must be checkable, and must fail closed.

### Goals

1. One trusted component (the daemon) owns lifecycle, privileged surfaces and privileged assets,
   outside the workload's reach.
2. Generalize the dual API boundary into policy-scoped surfaces for four principals.
3. A first-class grader surface: privileged tools never reachable from the agent's surface.
4. Make the harness boundary structural: harnesses run inside the workload sandbox, and are
   observed from outside it.
5. Declared guarantees that a backend either provides or refuses.
6. Fully backward compatible and opt-in.

### Non-Goals

- Building a sandbox. Kernel isolation is delegated to backends such as OpenShell.
- New reward semantics. Rewards stay inside the environment boundary (RFC 004).
- Multi-tenant serving. One daemon supervises one environment; one sandbox per episode.

## Design

### Principals and surfaces

| Principal | Surface | Credential | Policy |
|-----------|---------|------------|--------|
| `orchestrator` | WebSocket `/ws` (`reset`/`step`/`state`/`close`) | Bearer token | `allow_lifecycle` |
| `agent` | MCP `/mcp` (HTTP and WebSocket) | None | Domain tools only; never lifecycle names or `grader.*` |
| `grader` | MCP `/mcp/grader` | Bearer token | Domain tools plus `grader.*`; `fs_read` globs; `allow_privileged_exec` |
| `observer` | WebSocket `/observe` | Bearer token | Event streams only |

Agent tool patterns allow only a literal prefix with an optional trailing `*`, so overlap with
the reserved namespaces is decidable: `*`, `gr*`, `r*` and `[r]eset` are all rejected.

### Topology

```
              trusted side                         │   untrusted side (backend sandboxes)
                                                   │
 orchestrator ─┐                                   │   ┌─ episode sandbox (one per episode) ──┐
 grader ───────┼─▶ openenvd daemon :8100           │   │  worker  (env factory + actions)     │
 observer ─────┘     surfaces · tokens · runtime   │──▶│  workload processes (harness, ...)   │
 agent ──────────▶   /mcp (agent policy)           │   │  127.0.0.1:8000/mcp  agent-only      │
                     private dir (0700):           │   │  /sandbox/workspace  (seed copy)     │
                       seed, assets, snapshots     │   └──────────────────────────────────────┘
                                                   │   ┌─ privileged sandbox (per run) ───────┐
                                                   │──▶│  oracle / privileged process         │
                                                   │   │  workspace snapshot + assets         │
                                                   │   └──────────────────────────────────────┘
```

- **The daemon is the only holder of privilege.** Principal tokens, backend credentials (for
  example OpenShell gateway credentials), privileged assets and the pristine seed stay in its
  private directory. None of them enter the episode sandbox.
- **The workspace is a seed, never a mount.** Each episode gets a new sandbox with a copy of the
  seed at `/sandbox/workspace`. Grading reads downloaded snapshots.
- **Everything in one sandbox shares one trust level.** The worker, a harness and helper
  processes share one filesystem policy and one identity, so they can see each other. Anything
  that must be tamper-proof, such as oracle replay or final grading, runs on the daemon or in a
  privileged sandbox working from a snapshot.

### The contract

```yaml
# openenv.yaml
openenvd:
  enabled: true
  enforcement:
    backend: openshell            # or: local
    require: [asset_isolation, egress_control, privilege_drop, control_plane_isolation]
  openshell:                      # options for the openshell backend
    image: registry.example/my-env:openshell
    gateway: local
  surfaces:
    orchestrator: { allow_lifecycle: true }
    agent:        { tools: [env.*] }
    grader:
      tools: [env.*, grader.*]
      fs_read: [/workspace/**, /openenvd/assets/**]
      allow_privileged_exec: true
    observer:     { stream: [harness_events, fs_diff, process] }
  privileged_assets:              # relative to the daemon's --asset-root
    oracle: grade.sh
    solution: solution.txt
  workload:                       # backend-neutral filesystem
    read_only:  [/bin, /usr, /lib, /lib64, /etc, /proc, /opt, /dev/urandom]
    read_write: [/sandbox, /tmp, /dev/null]
  egress:                         # backend-neutral network
    mode: allowlist
    allow:
      - { host: api.anthropic.com, binaries: [/usr/local/bin/node] }
  processes:
    harness:
      trust: workload             # started in the episode sandbox after the worker
      argv: [/usr/local/bin/my-harness, --mcp, http://127.0.0.1:8000/mcp]
    rubric:
      trust: privileged           # run on demand in its own sandbox with assets
      asset: rubric
```

Static checks, applied whichever backend runs:

- The agent surface never matches lifecycle names or `grader.*`. Only graders may read files or
  run privileged processes. Observers carry no tools.
- Asset sources are relative paths inside the asset root and are never uploaded to the episode
  sandbox.
- Writable workload paths stay within `/sandbox`, `/tmp`, or exactly `/dev/null`. The worker's
  interpreter must not be writable.
- `egress` defaults to `none`. Allowlist rules may be scoped to the binaries allowed to use
  them, for example only the harness may reach its model API.
- A backend's native policy override and the neutral `workload`/`egress` declarations are
  mutually exclusive.

### Guarantees

| Guarantee | Meaning | `openshell` | `local` |
|-----------|---------|:-----------:|:-------:|
| `asset_isolation` | Privileged assets are unreachable from the workload's filesystem | ✓ | ✗ |
| `egress_control` | Workload egress is limited to the declared policy | ✓ | ✗ |
| `privilege_drop` | Workload runs unprivileged and cannot escalate | ✓ | ✗ |
| `control_plane_isolation` | Workload cannot reach orchestrator, grader or observer surfaces | ✓ | ✗ |

`enforcement.require` lists what the environment depends on. At startup the daemon calls
`backend.ensure(require)`:

1. If any required guarantee is not in the backend's set, it raises `EnforcementUnavailable`
   naming the missing guarantees.
2. Otherwise it runs `backend.probe()`, which checks host prerequisites. If they are absent,
   it raises `EnforcementUnavailable` naming the prerequisite.

Neither case falls back to another backend.

### Backend interface

```python
class Sandbox(Protocol):
    id: str | None
    async def start(self, seed: Path, directory: Path) -> None: ...  # create, verify, upload seed
    async def spawn(self, argv, env) -> asyncio.subprocess.Process: ...  # stdio-piped, env -i
    async def download(self, destination: Path) -> None: ...  # validated workspace export
    async def close(self) -> None: ...  # stop everything; confirm deletion

class EnforcementBackend(ABC):
    name: str
    guarantees: frozenset[Guarantee]
    async def ensure(self, required) -> None: ...  # refuse: missing guarantee or prerequisite
    async def probe(self) -> None: ...
    def sandbox(self, timeout_s: float) -> Sandbox: ...
    python: str  # interpreter for Python workers
    def workload_env(self) -> dict[str, str]: ...
```

Backends register by name (`register_backend`), so gVisor, Firecracker or cloud sandboxes can
be added without changes to core.

### The `openshell` backend

- **Policy.** Rendered from `workload` and `egress`, or taken from a native `openshell.policy`
  that must keep these invariants:
  - `landlock.compatibility: hard_requirement`
  - positive numeric `run_as_user`/`run_as_group`
  - `include_workdir: false`
  - confined writable paths
- **Probe.** The `openshell` CLI is stable `>=0.1.2,<0.2`, OpenSSH is installed, and
  `openshell status` reaches the configured gateway.
- **Start.**
  - `sandbox create --no-keep --no-auto-providers --label openenv-session=<uuid>` with a
    finite `/bin/sleep` main process, as a crash backstop.
  - Then `sandbox get` must report the expected identity, `phase=Ready`,
    `policy_source=sandbox`, and the requested policy. OpenShell's documented baseline paths
    are tolerated only when network rules are present.
  - Then the seed is uploaded and the SSH config saved with mode `0600`.
- **Spawn.** Over the sandbox's SSH transport: `cd /sandbox/workspace && exec /usr/bin/env -i
  <env> <argv>`. Gateway credentials are never inherited.
- **Download.** A bounded uncompressed tar (64 MiB, 4096 entries) is fully validated before any
  host write. Symlinks, sparse files, `..` paths, duplicates and conflicts are rejected.
- **Close.** Terminate every process, `sandbox stop`, then `sandbox delete`, and poll the
  label-scoped inventory until deletion is confirmed. Unconfirmed deletion blocks the next
  episode.

`control_plane_isolation` holds because the daemon's surfaces are outside the sandbox and
OpenShell never authorizes loopback or link-local destinations. Don't add the daemon's own
address to the egress allowlist.

### The `local` backend

A private directory holding a copy of the seed, with processes run as host subprocesses. It
lets authors develop environments, graders and harness wiring without a gateway. It provides no
guarantees, so any manifest that requires one refuses to start under it.

### Processes

| Trust | Where it runs | When | Example |
|-------|---------------|------|---------|
| `workload` | The episode sandbox, with the worker | Started after the worker on every reset; stopped with the sandbox | Harness, browser, language server |
| `privileged` | A fresh sandbox holding a workspace snapshot and the assets | On demand, through `grader.run_process(name)` by a grader with `allow_privileged_exec` | Oracle, rubric, test suite |

The worker itself is implicit. An `oracle` asset is an implicit privileged process, kept for
`grader.run_oracle`. Workload processes reach the environment through the agent-only listener at
`127.0.0.1:8000/mcp`, which has no privileged routes. Their output is drained, not retained.
A process that exits does not restart; the orchestrator decides whether to reset.

### Episode lifecycle

1. **Startup.**
   - Validate the contract.
   - Call `backend.ensure(require)`.
   - Capture the seed and copy the assets into the daemon's private directory.
   - Start the episode sandbox and the worker.
2. **Reset** (orchestrator only).
   - Stop the current sandbox and confirm its deletion.
   - Start a new sandbox from the captured seed.
   - Spawn the worker, which receives `{factory, action_class, agent_policy}`.
   - Spawn the workload processes.
   - Forward `reset`.
3. **Step / MCP.** The daemon serializes requests to the worker under one episode lock. A
   worker exit, invalid frame or timeout tears down the sandbox.
4. **Grading.**
   - `grader.fs_diff` and `grader.read_file` work on downloaded snapshots.
   - `grader.run_process` and `grader.run_oracle` stage the snapshot plus assets into a
     privileged sandbox, run the declared executable, return its output, and delete the
     sandbox.
   - Nothing is copied back to the episode.
5. **Observation.** Harness events are reported by the workload. Process events cover the
   worker, workload processes and sandbox lifecycle. `fs_diff` samples snapshots. `network` and
   `resource` streams are rejected until backend telemetry is integrated.

### Where guarantees are available

| Deployment | `openshell` | `local` |
|------------|-------------|---------|
| Linux host, Docker or Podman, kernel 6.2+ with Landlock ABI ≥ 3 | All four | None |
| macOS with Docker Desktop (Apple Silicon) | All four | None |
| Kubernetes gateway | All four | None |
| HF Spaces | Unavailable: a Space can't run a gateway, so a manifest requiring guarantees refuses to start | None |

### Backward compatibility

- With no `openenvd:` block, or `enabled: false`, the manifest's app runs unchanged.
- The agent's MCP surface is unchanged. openenvd adds no tools, headers or metadata that would
  let a trained policy detect it.
- `EnvClient` and `GenericEnvClient` gain optional request headers for bearer tokens.

## Review questions on #1053

- **Who is calling a surface?** Each privileged principal gets its own bearer token, supplied
  through deployment configuration. The agent surface is unauthenticated but holds only agent
  tools. Deployments exposing a non-loopback address supply TLS and network access controls.
- **What does reset reset?** One sandbox per episode, recreated from the seed. Deletion is
  confirmed before the next episode starts. Concurrent episodes run under separate daemons.
- **Which privileges, and who grants them?** The backend's own infrastructure, outside the
  workload: for OpenShell, the gateway and its driver. The daemon needs no root, no
  `CAP_SYS_ADMIN` and no writable cgroups. Where no backend can grant what the manifest
  requires, startup fails.

## Implementation

Two stacked PRs:

1. **Contract and enforcement:**
   - `openenv._openenvd_config` for the contract.
   - `openenv.core.openenvd.backends` for the `Sandbox` and `EnforcementBackend` interfaces,
     the registry, and the `openshell` and `local` backends.
2. **Daemon:**
   - The runtime, surfaces and worker protocol.
   - Grader tools, observation, processes and clients.
   - The `openenvd` CLI.
   - `EnvClient` header support and RFC 008 manifest preservation.

## Future Work

- **Backend telemetry:** `network` observer events from OpenShell's OCSF allow and deny
  logs, and resource sampling.
- **Credentials:** OpenShell providers, so harness model credentials are injected at the
  egress proxy instead of passed in the environment, which would add a `credential_isolation`
  guarantee.
- **RFC 008:** run the contract checks and OpenShell's policy prover in `openenv validate`.
- **Mid-episode grader policy:** for example, a grader may connect only after `done`.
- **More backends:** gVisor, Firecracker, and cloud sandbox providers.
