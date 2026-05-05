# MCP Environment Lifecycle

This guide explains how MCP-backed environments work end to end in OpenEnv.

It exists to answer a common question: if an environment exposes MCP tools, when does `step()` run, when does `step_async()` run, and when should you use `call_tool()` versus `step(CallToolAction(...))`?

## The Short Answer

MCP environments in OpenEnv can be used in two layers:

- **Simulation layer**: the OpenEnv training loop controls `reset()`, `step()`, and `state()`.
- **Tool layer**: MCP tools are exposed through `ListToolsAction`, `CallToolAction`, `list_tools()`, and `call_tool()`.

If you are training or evaluating with episode control, the canonical pattern is still the OpenEnv step loop.

If you are serving tools to an external client, the MCP layer is the interface the agent should see.

## The Two Boundaries

OpenEnv keeps a strict API split:

- **Infrastructure boundary**: Gym-like control over `/ws`, `reset()`, `step()`, and `state()`
- **Agent boundary**: MCP tools over `/mcp`

This means:

- agents should use MCP tools
- orchestration and training infrastructure use the simulation control loop
- `/ws` is not an agent-facing interface, even if it is available on the server

## How MCP Environments Handle Actions

`MCPEnvironment` is still an OpenEnv environment.

It does not replace the step loop. Instead, it maps MCP actions into the step loop.

In simulation mode, MCP tool usage is represented as normal environment actions:

```python
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

obs = env.step(ListToolsAction())

obs = env.step(
    CallToolAction(
        tool_name="echo_message",
        arguments={"message": "Hello"},
    )
)
```

That is why an MCP-backed environment can still participate in:

- rewards
- `done` handling
- step counts
- trajectory logging

## Why `step()` May Look Like It Is Not Running

This is the main source of confusion.

On the server side, the WebSocket handler checks whether the environment overrides `step_async()`.

- if `step_async()` is overridden, the WebSocket path calls `step_async()`
- otherwise, it falls back to `step()`

That means an async client using the WebSocket session path may execute `step_async()` without hitting your synchronous `step()` instrumentation.

So if you add debug prints only to `step()` and use an async MCP client, it can look like "step is not being invoked" even though the action is being processed normally.

For debugging, check both:

- `step()`
- `step_async()`

The same rule applies to `reset()` and `reset_async()`.

## What `list_tools()` and `call_tool()` Actually Do

Environment-specific MCP clients such as `EchoEnv` and `FinQAEnv` inherit from `MCPToolClient`.

Those clients expose convenience methods:

- `list_tools()`
- `call_tool()`

These are helpers, not a separate environment lifecycle.

### Default behavior

By default, the convenience methods still go through the OpenEnv session path.

- `list_tools()` wraps `step(ListToolsAction())`
- `call_tool()` wraps `step(CallToolAction(...))`

This preserves:

- episode context
- rewards
- step counting
- trajectory semantics

### Direct MCP behavior

When production MCP access is explicitly enabled on the client, the same convenience methods use the HTTP `/mcp` JSON-RPC endpoint directly.

That path is for tool-serving behavior, not the training loop.

## Which Pattern Should You Use?

Use `step(CallToolAction(...))` when you need the full OpenEnv result object:

- `reward`
- `done`
- observation metadata
- trajectory-compatible behavior

Use `call_tool()` when you only want the tool result and do not need to manually inspect the full `StepResult`.

In other words:

- `step(...)` is the canonical simulation pattern
- `call_tool()` is a convenience wrapper

## Concrete Examples

Two good references in this repo are:

- [Echo environment](environments/echo.md)
- [FinQA environment](environments/finqa.md)

For a minimal simulation-mode example, see:

- `examples/echo_mcp_demo.py`

Echo is useful because it shows the MCP mechanics with almost no domain logic.

FinQA is useful because it shows an MCP environment where tool calls also participate in episode progression, rewards, and terminal submission.

## Recommended Mental Model

Think about MCP environments in OpenEnv like this:

1. The environment is still an OpenEnv environment.
2. MCP tools are one kind of action the environment knows how to handle.
3. In simulation mode, tool calls are part of the step loop.
4. In production mode, MCP becomes the agent-facing boundary.
5. The WebSocket simulation interface remains infrastructure-only and must not be given directly to agents.

## Debugging Checklist

If an MCP environment "doesn't call step", check these first:

1. Are you using an async client path that triggers `step_async()`?
2. Did you instrument both `step()` and `step_async()`?
3. Are you using `call_tool()` and assuming it bypasses the step loop?
4. Are you expecting the MCP tool layer to behave like a separate environment lifecycle?

Usually the action is flowing correctly, but through the async WebSocket path rather than the synchronous method you were watching.

## Related Reading

- [Core API](core.md)
- [Echo environment](environments/echo.md)
- [FinQA environment](environments/finqa.md)
