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

Those clients target a running environment server and expose convenience methods:

- `list_tools()`
- `call_tool()` — **async**, must be awaited

These are helpers, not a separate environment lifecycle.

### Client behavior

`MCPToolClient` and its base `MCPClientBase` only support `mode="production"`; construction raises `ValueError` for other modes. For direct in-process training or eval code, instantiate the environment class and call `env.step(CallToolAction(...))` instead of using `MCPToolClient`.

- `list_tools()` wraps `step(ListToolsAction())` and returns the `list[Tool]` directly.
- `call_tool(name, **kwargs)` returns the **unwrapped tool return value** directly — not the `CallToolObservation`, and not the runtime result object you would get from `obs.result`.
- `call_tool()` raises `RuntimeError` on any tool error. Use `step(CallToolAction(...))` when you need to inspect `ToolError.error_type` or continue after a failed tool call.

Remote `step(CallToolAction(...))` calls return a `StepResult`; the full observation is on `result.observation`, while `result.reward` carries the serialized reward field.

### Direct MCP behavior

When production MCP access is explicitly enabled on the client, the same convenience methods use the HTTP `/mcp` JSON-RPC endpoint directly.

That path is for tool-serving behavior, not the training loop. It bypasses reward computation, step counts, trajectory tracking, and `done` handling.

## Which Pattern Should You Use?

Use `step(CallToolAction(...))` when you need the full `CallToolObservation`:

- `reward`
- `done`
- observation metadata
- `obs.result`, a runtime result object typed as `Any`; FastMCP commonly returns `fastmcp.client.client.CallToolResult` with `.data`, `.content`, and `.structured_content`, but serialized clients or custom envs may surface a dict or plain value
- trajectory-compatible behavior

Use `await env.call_tool(name, **kwargs)` when you only want the tool's raw return value and do not need to inspect the full observation. It is async and unwraps the result for you.

In other words:

- `step(...)` is the canonical simulation pattern
- `call_tool()` is an async convenience wrapper that returns the unwrapped tool output

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
