# Your First Environment

!!! note "Coming Soon"
    This page will contain a streamlined guide to building your first OpenEnv environment. For now, see the [environment anatomy guide](environment-anatomy.md).

## Overview

Building an OpenEnv environment involves:

1. **Define your models** - Action, Observation, and State types
2. **Implement the environment** - Core logic in a Python class
3. **Create the server** - FastAPI wrapper for HTTP access
4. **Package for deployment** - Docker container and manifest

## Quick Example

Here's a minimal environment that echoes back messages:

```python
from pydantic import BaseModel
from openenv.core import Environment

class EchoAction(BaseModel):
    message: str

class EchoObservation(BaseModel):
    echo: str

class EchoState(BaseModel):
    last_message: str = ""

class EchoEnvironment(Environment[EchoAction, EchoObservation, EchoState]):
    def reset(self) -> EchoObservation:
        self.state = EchoState()
        return EchoObservation(echo="Ready!")

    def step(self, action: EchoAction) -> tuple[EchoObservation, float, bool]:
        self.state.last_message = action.message
        return EchoObservation(echo=action.message), 0.0, False
```

## Next Steps

- [Environment Anatomy](environment-anatomy.md) - Deep dive into structure
- [Deployment](deployment.md) - Deploy to Docker and HF Spaces
- [Contributing Environments](../contributing.md) - Share your environment with the community
