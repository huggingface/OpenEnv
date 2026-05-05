# Core Concepts

:::{note}
Coming Soon
    This page is under construction.

Understanding

Understanding OpenEnv's core abstractions helps you work effectively with environments and build your own.

## The OpenEnv Model

OpenEnv follows a **client-server architecture** inspired by Gymnasium's simple API:

```
┌─────────────────┐     HTTP/WebSocket     ┌─────────────────┐
│   Your Agent    │ ◄──────────────────► │   Environment   │
│   (Client)      │    step/reset/state    │    (Server)     │
└─────────────────┘                        └─────────────────┘
```

## Key Abstractions

### Environment

An **Environment** is an isolated execution context where your agent can take actions and receive observations. Environments run as servers (typically in Docker containers) and expose a standard API.

### Action

An **Action** is a structured command that your agent sends to the environment. Each environment defines its own action schema.

```python
from coding_env import CodeAction

action = CodeAction(code="print('Hello!')")
```

### Observation

An **Observation** is the response from the environment after taking an action. It contains the current state visible to your agent.

```python
result = client.step(action)
print(result.observation.stdout)  # "Hello!"
```

### StepResult

A **StepResult** bundles together everything returned from a step:

- `observation`: What the agent can see
- `reward`: Numeric reward signal (for RL training)
- `terminated`: Whether the episode has ended
- `truncated`: Whether the episode was cut short
- `info`: Additional metadata

### Client

A **Client** is how you connect to and interact with an environment. OpenEnv provides both async and sync clients.

```python
from openenv import AutoEnv

env = AutoEnv.from_env("coding")

# Async (recommended for production)
async with env as client:
    result = await client.reset()
    result = await client.step(action)

# Sync (convenient for scripts)
with env.sync() as client:
    result = client.reset()
    result = client.step(action)
```

## The Step Loop

The core interaction pattern is the **step loop**:

```python
with env.sync() as client:
    # 1. Reset to get initial state
    result = client.reset()

    while not result.terminated:
        # 2. Observe current state
        obs = result.observation

        # 3. Decide on action (your agent logic)
        action = decide_action(obs)

        # 4. Take action, get new state
        result = client.step(action)

        # 5. Learn from reward (for RL)
        learn(result.reward)
```

## Connection Methods

OpenEnv supports multiple ways to connect to environments:

| Method | Use Case | Example |
|--------|----------|---------|
| **HTTP URL** | Remote servers, HF Spaces | `AutoEnv.from_env("user/space")` |
| **Docker** | Local development | `AutoEnv.from_docker_image("env:latest")` |
| **Direct** | Testing, embedded | `EnvClient(base_url="http://localhost:8000")` |

## Next Steps

- [Quick Start](quickstart.md) - Try these concepts hands-on
- [Auto-Discovery](guides/auto-discovery.md) - How to discover and load environments
- [Your First Environment](guides/first-environment.md) - Build your own environment
:::
