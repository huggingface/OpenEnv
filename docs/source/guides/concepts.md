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
- `done`: whether the episode has ended
- `metadata`: additional metadata returned alongside the observation

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

# Async
async with AutoEnv.from_env("coding") as client:
    result = await client.reset()
    result = await client.step(action)

# Sync, with its own client (a client is locked to the mode it is first used in)
with AutoEnv.from_env("coding").sync() as client:
    result = client.reset()
    result = client.step(action)
```

## The Step Loop

```python
with env.sync() as client:
    result = client.reset()

    while not result.done:
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
| Cloud / custom runtime | Run the server on a cloud sandbox | `EnvClient.from_docker_image("env:latest", provider=DaytonaProvider())` |
| Auto-discovery | Installed packages or known environments | `AutoEnv.from_env("echo")` |

See the [Runtime Providers guide](runtime-providers.md) for the available providers and how to pick one.

## Environment Anatomy

Every OpenEnv environment consists of:

```
my_env/
├── openenv.yaml               # Manifest file
├── __init__.py                # Exports the client and models
├── client.py                  # Client class
├── models.py                  # Action and Observation types
├── pyproject.toml             # Package metadata
├── README.md                  # Documentation (and the Space card)
└── server/
    ├── app.py                 # FastAPI app built with create_app
    ├── my_env_environment.py  # The Environment: reset() and step()
    └── Dockerfile             # Container definition
```

`openenv init my_env` generates this layout.

### The Manifest (openenv.yaml)

```yaml
spec_version: 1
name: my_env
version: 0.1.0
type: space
runtime: fastapi
app: server.app:app
port: 8000
validation:
  reward:
    range: [0.0, 1.0]
  capabilities:
    verifier:
      kind: reward_channel
```

`openenv init` writes the full manifest, including the resource limits checked by `openenv validate`. The client, action and observation classes are found by naming convention (`MyEnv`, `MyAction`, `MyObservation`), so the manifest doesn't list them.

### Models (Pydantic)

Custom `Action`, `Observation`, and `State` types subclass the base classes from `openenv.core.env_server.types` — not `pydantic.BaseModel` directly. The base `Observation` already carries `done` and `reward` fields, which `step()` populates; `Action` and `State` add metadata plumbing used by the server.

```python
from openenv.core.env_server.types import Action, Observation, State


class MyAction(Action):
    command: str
    args: list[str] = []


class MyObservation(Observation):
    output: str
    success: bool


class MyState(State):
    history: list[str] = []
```

### Environment Class

Environments subclass the abstract `Environment[ActT, ObsT, StateT]` base and implement `reset`, `step`, and the `state` property. Reward and termination are carried on the returned observation — they are **not** a tuple return value.

```python
from openenv.core.env_server.interfaces import Environment


class MyEnvironment(Environment[MyAction, MyObservation, MyState]):
    def reset(self, seed=None, episode_id=None, **kwargs) -> MyObservation:
        ...

    def step(self, action: MyAction, timeout_s=None, **kwargs) -> MyObservation:
        ...

    @property
    def state(self) -> MyState:
        ...
```

### Server (FastAPI)

Use `create_app` from `openenv.core.env_server` to wrap the environment as a FastAPI application. Pass the environment **class** (used as a factory so each WebSocket session gets its own instance) along with the action and observation types:

```python
from openenv.core.env_server import create_app

app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="my_env",
)
```

This is what the environment's `server/app.py` entry point typically does — see `envs/echo_env/server/app.py` for a minimal real example.

### Rewards via the Rubric

Rewards are computed **inside the environment**, not by external code. The base `Environment` accepts an optional `rubric` on `__init__` — pass it to `super().__init__(rubric=...)`, call `self._reset_rubric()` from `reset`, and `self._apply_rubric(action, observation)` from `step` (or `_apply_rubric_async` from `step_async`). The [Rubrics tutorial](../tutorials/rubrics.md) covers the composable API end-to-end.

## Next Steps

- [Getting Started](../getting-started.md)
- [Auto-discovery](auto-discovery.md)
- [Your first environment](first-environment.md)
