# Concepts

OpenEnv follows a client-server model inspired by Gymnasium's simple API.
Agents send structured actions to isolated environments and receive
observations, rewards, and episode status in return.

```
+-----------------+     HTTP/WebSocket     +-----------------+
|   Your Agent    | <--------------------> |   Environment   |
|   (Client)      |    step/reset/state    |    (Server)     |
+-----------------+                        +-----------------+
```

## Key Abstractions

### Environment

An **Environment** is an isolated execution context where your agent can take
actions and receive observations. Environments usually run as servers and expose
a standard API.

### Action

An **Action** is a structured command that your agent sends to the environment.
Each environment defines its own action schema.

```python
from coding_env import CodeAction

action = CodeAction(code="print('Hello!')")
```

### Observation

An **Observation** is the response from the environment after taking an action.
It contains the current state visible to your agent.

```python
result = client.step(action)
print(result.observation.stdout)  # "Hello!"
```

### StepResult

A **StepResult** bundles together everything returned from a step:

- `observation`: what the agent can see
- `reward`: numeric reward signal for training
- `terminated`: whether the episode has ended
- `truncated`: whether the episode was cut short
- `info`: additional metadata

### Rubric

A **Rubric** is a composable unit of reward computation that lives inside the
environment. Rubrics can be combined with `WeightedSum`, `Gate`, and
`Sequential`; use LLM judges for subjective criteria; and handle delayed rewards
with `TrajectoryRubric`. See the [Rubrics tutorial](../tutorials/rubrics.md)
for the full API.

### Client

A **Client** is how you connect to and interact with an environment. OpenEnv
provides both async and sync clients.

```python
from openenv import AutoEnv

env = AutoEnv.from_env("coding")

async with env as client:
    result = await client.reset()
    result = await client.step(action)

with env.sync() as client:
    result = client.reset()
    result = client.step(action)
```

## The Step Loop

```python
with env.sync() as client:
    result = client.reset()

    while not result.terminated:
        obs = result.observation
        action = decide_action(obs)
        result = client.step(action)
        learn(result.reward)
```

## Connection Methods

| Method | Use Case | Example |
|--------|----------|---------|
| HTTP URL | Remote servers, Hugging Face Spaces | `EnvClient(base_url="https://...")` |
| Docker | Local development | `EnvClient.from_docker_image("env:latest")` |
| Auto-discovery | Installed packages or known environments | `AutoEnv.from_env("echo")` |

## Next Steps

- [Getting Started](../getting-started.md)
- [Auto-discovery](auto-discovery.md)
- [Your first environment](first-environment.md)
