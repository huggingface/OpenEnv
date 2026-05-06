# Environment Anatomy

A deep dive into the structure of OpenEnv environments.

## Components

Every OpenEnv environment consists of:

```
my_env/
├── openenv.yaml          # Manifest file
├── my_env/
│   ├── __init__.py
│   ├── client.py         # Client classes
│   ├── server.py         # Server/Environment
│   └── models.py         # Pydantic models
├── Dockerfile            # Container definition
├── pyproject.toml        # Package metadata
└── README.md             # Documentation
```

## The Manifest (openenv.yaml)

```yaml
name: my_env
version: 0.1.0
description: My custom environment

client:
  class_name: MyEnvClient
  module: my_env.client

action:
  class_name: MyAction
  module: my_env.models

observation:
  class_name: MyObservation
  module: my_env.models

default_image: my-env:latest
spec_version: 1
```

## Models (Pydantic)

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

## Environment Class

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

## Server (FastAPI)

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

## Next Steps

- [Deployment](deployment.md) - Deploy your environment
- [Your First Environment](first-environment.md) - Build step by step
