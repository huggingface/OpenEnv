# openenvd

openenvd is the control plane of an environment unit (RFC 009, [#1053](https://github.com/huggingface/OpenEnv/issues/1053)).
It runs inside the unit as PID 1, in the parent namespaces, and builds three
zones of containers in child namespaces:

| Zone | Containers | Can reach |
|------|------------|-----------|
| `agent` | `env` (the unmodified env server), optional `harness` | Only its own socket directory: recording relays for MCP, the model and hidden services |
| `services` | hidden simulators, for example `airbnb-sim` | Nothing. The `env` container calls them through a recording relay |
| `observers` | grader, judge, metrics | Read-only views of the workspace, assets, service state and the sealed trace |

The orchestrator talks to one listener (`:8100`): `/ws` (Gym, token), `/mcp`
(agent, unchanged), `/info`, `/reset_unit`, `/inspect`, `/resume`, `/end_episode`
(token) and `/observe` (observer token).

## Modules

| Module | Role |
|--------|------|
| `contract.py` | The `openenvd:` manifest block: zones, per-zone policy ceilings, phases, guarantees |
| `phases.py` | Canonical phases and who may cause each transition |
| `unit.py` | Builds an episode: overlays, socket dirs, relays, containers, sealing, grading |
| `probes.py` | Detects what the runtime permits; picks the tier; refuses unmet requirements |
| `cgroups.py` / `proctree.py` | Zone subtrees: cgroup v2 when writable, process trees with a watchdog otherwise |
| `oci.py`, `seccomp.py`, `runtime.py` | OCI bundles and the `crun` driver (`containers` tier) |
| `shim.py`, `landlock.py`, `forwarder.py` | Container PID 1: Landlock, a socket-family filter, loopback forwarders |
| `relays.py`, `surfaces.py`, `peercred.py` | Recording relays and the external listener |
| `trace.py`, `custody.py` | Hash-chained, sealed trace; validated copies of agent output for graders |
| `check.py` | `openenv check enforcement`: probe containers try what each zone must not do |

## Tiers

| Tier | When | Zones are |
|------|------|-----------|
| `containers` | user namespaces, a writable cgroup root and an OCI runtime (VM, Sysbox, userns pod, privileged dev container) | `crun` containers with their own user, PID, mount, network, IPC, UTS and cgroup namespaces |
| `landlock` | Landlock but no nesting (default Docker, HF Spaces) | Process trees in the unit's namespaces, confined by Landlock (files and TCP ports) and seccomp |
| `none` | neither | Plain processes; every guarantee is `not_supported` |

`enforcement.require` names the minimum strength per guarantee. openenvd refuses
to start below it and reports what it got at `GET /info`.

## Running

```bash
python -m openenv.core.openenvd --manifest openenv.yaml --tokens tokens.json
```

Container paths are also exposed as `OPENENVD_WORKSPACE`, `OPENENVD_OUT`,
`OPENENVD_INPUTS`, `OPENENVD_ASSETS`, `OPENENVD_SERVICES` and
`OPENENVD_SOCKETS`, so observers and harnesses work in both tiers. The env
container gets `OPENENVD_SERVICE_<NAME>` for each hidden service.

In the `containers` tier, the image must provide empty mountpoints at
`/workspace`, `/assets`, `/inputs`, `/services`, `/out` and `/run/openenvd`, and
each service's `state` path. List any extra code directories a zone needs under
`isolation.read_only`. Anything else outside the system directories is invisible
to the zone's Landlock domain.

## Tests

```bash
PYTHONPATH=src:envs uv run pytest tests/core/test_openenvd
# Live, in Docker: both tiers, real namespaces, cgroups, Landlock and seccomp
tests/core/test_openenvd/integration/run.sh
```
