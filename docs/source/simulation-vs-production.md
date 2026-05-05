# Simulation vs Production Mode

OpenEnv has two related but different ideas of "mode":

- **Simulation mode** is for training, evaluation, and any workflow where the orchestrator controls episode boundaries.
- **Production mode** is for exposing tools directly to clients over MCP without the training loop controlling `reset()`, `step()`, or `state()`.

This guide explains when to use each mode and how they interact with MCP tools.

## The Short Version

Use **simulation mode** when you need:

- `reset()`, `step()`, and `state()`
- rewards and `done` signals
- one action per step in a controlled trajectory
- training or evaluation loops

Use **production mode** when you need:

- direct MCP tool access
- no simulation control routes
- an agent or client talking to tools as a live service
- the environment to get out of the way and expose the tool interface directly

## Why OpenEnv Has Two Modes

This split follows the core OpenEnv design principles:

- training and evaluation need a controlled step loop
- production integrations need direct tool access
- the same environment should support both without inventing separate environment implementations

In practice, simulation mode models **trajectory time** and production mode models **service time**.

## Simulation Mode

Simulation mode is the default environment-control model.

In simulation mode, the orchestrator owns the episode:

1. Call `reset()` to start an episode.
2. Call `step()` for each action.
3. Read `reward`, `done`, and `state()` as part of the rollout.

For environments with MCP tools, the canonical simulation-mode pattern is still `step()`.

```python
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

obs = env.step(ListToolsAction())

obs = env.step(
    CallToolAction(
        tool_name="echo_message",
        arguments={"message": "Hello from simulation mode"},
    )
)
```

That pattern matters because the training loop can then:

- count tool usage as actions
- assign rewards to tool interactions
- record a full trajectory
- preserve the same `reset`/`step` contract across environments

### Simulation-Mode Routes

When an `HTTPEnvServer` registers routes in simulation mode, it exposes the full control surface:

- `/ws`
- `/mcp`
- `/reset`
- `/step`
- `/state`

This is the right mode for RL training infrastructure and for most environment testing.

## Production Mode

Production mode is for exposing tools directly.

In production mode, clients should interact with MCP tools as a service instead of driving the environment through `reset()` and `step()` as a trajectory loop.

Production mode keeps the MCP surface and removes the HTTP simulation control routes.

### Production-Mode Routes

When an `HTTPEnvServer` registers routes in production mode, OpenEnv does **not** expose:

- `/reset`
- `/step`
- `/state`

It still registers `/ws`, because the WebSocket transport remains part of the infrastructure boundary.

That does **not** mean `/ws` should be exposed to agents.

- `/ws` is for orchestration and simulation control
- `/mcp` is the agent-facing boundary
- production deployments should restrict `/ws` at the network, auth, or gateway layer if agents can reach the service directly

In other words, production mode removes the HTTP simulation endpoints, but operators must still treat `/ws` as infrastructure-only.

This is the right mode when:

- you are serving a tool-backed environment to external clients
- you do not want callers controlling episode boundaries
- the MCP interface is the product surface

## How MCP Fits Into Both Modes

MCP is available in both modes, but the role is different.

### In simulation mode

MCP tools are part of the environment action space.

- tool discovery can still happen
- tool calls are modeled as actions
- rewards and episode control remain in the OpenEnv loop

This is why examples such as `examples/echo_mcp_demo.py` use `ListToolsAction` and `CallToolAction` through `step()`.

### In production mode

MCP is the primary interface.

- clients call tools directly
- OpenEnv does not present simulation-control endpoints
- the service behaves like a live MCP endpoint, not an RL rollout loop

## Server-Side Configuration

The server-side switch happens when routes are registered.

```python
from fastapi import FastAPI

from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.types import ServerMode

app = FastAPI()
server = HTTPEnvServer(env=MyEnv, action_cls=MyAction, observation_cls=MyObservation)

# Training / evaluation
server.register_routes(app, mode=ServerMode.SIMULATION)

# Direct MCP serving
server.register_routes(app, mode=ServerMode.PRODUCTION)
```

`ServerMode.SIMULATION` is the default route-registration mode.

## Client-Side Patterns

For simulation-style interaction, use a client that participates in the OpenEnv control loop.

Examples:

- an environment-specific `EnvClient[...]` subclass
- `GenericEnvClient(base_url=..., mode="simulation")`
- `step(ListToolsAction())` and `step(CallToolAction(...))` for MCP-backed environments

For direct MCP access, use an MCP-oriented client.

Examples:

- `MCPToolClient(base_url=...)`
- environment-specific clients built on top of `MCPToolClient`

`MCPToolClient` defaults to production mode and rejects `mode="simulation"`.

## Mode-Aware Tools

`MCPEnvironment` supports mode-aware tool registration, so you can expose different tools depending on how the environment is being used.

```python
class MyEnv(MCPEnvironment):
    def __init__(self):
        @self.tool(mode="simulation")
        def score_candidate(answer: str) -> str:
            return "Used inside the training loop"

        @self.tool(mode="production")
        def lookup_docs(query: str) -> str:
            return "Used by live MCP clients"
```

This lets one environment preserve the training contract while still serving a cleaner production surface.

## Choosing the Right Mode

Choose **simulation mode** if the caller needs to control trajectories.

Typical cases:

- RL training
- policy evaluation
- benchmarking with rewards
- environments where tool calls should count as agent actions

Choose **production mode** if the caller needs direct tool access.

Typical cases:

- agent runtimes that speak MCP directly
- demos and hosted services
- integrations where `reset()` and `step()` should not be public

## Common Mistake

The most common confusion is assuming that "MCP environment" automatically means "production mode only".

That is not the model OpenEnv uses.

- An MCP-backed environment can still run in **simulation mode**.
- In simulation mode, MCP tool interactions are represented through the OpenEnv step loop.
- Production mode changes the public control surface, not the underlying environment concept.

## Related Reading

- [Core API](core.md)
- [Getting Started Tutorials](auto_getting_started/index)
- [RFC 002: Environment Spec](https://github.com/meta-pytorch/OpenEnv/blob/main/rfcs/002-env-spec.md)
- [RFC 005: Agentic Harnesses](https://github.com/meta-pytorch/OpenEnv/blob/main/rfcs/005-agentic-harnesses.md)
