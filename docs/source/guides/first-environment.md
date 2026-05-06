# Your First Environment

:::{note}
This page is a condensed preview. For the end-to-end walk-through — including Docker packaging, `openenv.yaml`, and Hugging Face Space deployment — see the [full environment builder guide](../auto_getting_started/environment-builder.md).
:::

## Overview

Building an OpenEnv environment involves:

1. **Define your models** - `Action`, `Observation`, and `State` types
2. **Implement the environment** - Core logic in a Python class
3. **Create the server** - FastAPI wrapper for HTTP access
4. **Package for deployment** - Docker container and manifest

## Quick Example

Here's a minimal environment that echoes back messages. Reward and `done` are fields on the `Observation` — `step` returns an observation, not a tuple.

```python
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State


class EchoAction(Action):
    message: str


class EchoObservation(Observation):
    echo: str


class EchoState(State):
    last_message: str = ""


class EchoEnvironment(Environment[EchoAction, EchoObservation, EchoState]):
    def reset(self, seed=None, episode_id=None, **kwargs) -> EchoObservation:
        self._state = EchoState()
        return EchoObservation(echo="Ready!")

    def step(self, action: EchoAction, timeout_s=None, **kwargs) -> EchoObservation:
        self._state.last_message = action.message
        return EchoObservation(echo=action.message, reward=0.0, done=False)

    @property
    def state(self) -> EchoState:
        return self._state
```

## Next Steps

- [Environment Anatomy](environment-anatomy.md) - Deep dive into structure
- [Deployment](deployment.md) - Deploy to Docker and HF Spaces
- [Full Guide](../auto_getting_started/environment-builder.md) - Complete documentation
