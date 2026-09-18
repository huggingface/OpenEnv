# Building Your Own Environment

This guide shows you how to create a custom environment using the EnvTorch framework.

## Overview

Creating an environment involves five main steps:
1. Define your models (Action, Observation, State)
2. Implement the environment APIs: step, reset, state
3. Create the FastAPI server
4. Build a Docker image and push it to a public docker repo for community to access it
5. Subclass HTTPEnvclient and implement the parsing methods for result and state.

## Step-by-Step Guide

### 1. Define Models

Create your action, observation, and state models using Python dataclasses:

```python
# models.py
from dataclasses import dataclass
from openenv.core.env_server import Action, Observation, State


@dataclass
class MyAction(Action):
    """Your custom action."""

    command: str
    parameters: dict


@dataclass
class MyObservation(Observation):
    """Your custom observation."""

    result: str
    success: bool


@dataclass
class MyState(State):
    """Custom state fields."""

    custom_field: int = 0
```

### 2. Implement Environment

Implement the three core methods: `reset()`, `step()`, and `state`:

```python
# server/my_environment.py
import uuid
from openenv.core.env_server import Environment
from ..models import MyAction, MyObservation, MyState


class MyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = MyState()

    def reset(self) -> MyObservation:
        self._state = MyState(episode_id=str(uuid.uuid4()))
        return MyObservation(result="Ready", success=True)

    def step(self, action: MyAction) -> MyObservation:
        # Implement your logic here
        self._state.step_count += 1
        result = self._execute_command(action.command)
        return MyObservation(result=result, success=True)

    @property
    def state(self) -> MyState:
        return self._state
```

### 3. Create FastAPI Server

Use the `create_fastapi_app` helper to create your HTTP server:

```python
# server/app.py
from openenv.core.env_server import create_fastapi_app
from ..models import MyAction, MyObservation
from .my_environment import MyEnvironment

env = MyEnvironment()
app = create_fastapi_app(env, MyAction, MyObservation)
```

### 4. Define Dependencies

**For Python-only dependencies (most common case):**

Create `envs/my_env/server/requirements.txt`:
```txt
your-package>=1.0.0
another-package
```

**For complex setup (optional, only if needed):**

If you need additional setup beyond pip install, create `envs/my_env/server/install_deps.sh`:
```bash
#!/bin/bash
set -e

# Install Python dependencies
pip install --no-cache-dir -r /tmp/requirements.txt

# Additional setup commands (only if needed)
mkdir -p /some/directory
# ... other setup steps
```

### 5. Create Dockerfile

Build your Docker image from the openenv-base. Place this at `envs/my_env/server/Dockerfile`:

**Simple case (just requirements.txt):**
```dockerfile
# Accept base image as build argument for CI/CD flexibility
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install dependencies
COPY envs/my_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy environment code
COPY src/openenv/core/ /app/src/openenv/core/
COPY envs/my_env/ /app/envs/my_env/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "envs.my_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Complex case (requirements.txt + install_deps.sh):**
```dockerfile
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install dependencies and run setup
COPY envs/my_env/server/requirements.txt /tmp/requirements.txt
COPY envs/my_env/server/install_deps.sh /tmp/install_deps.sh
RUN chmod +x /tmp/install_deps.sh && \
    /tmp/install_deps.sh && \
    rm /tmp/install_deps.sh /tmp/requirements.txt

# Copy environment code
COPY src/openenv/core/ /app/src/openenv/core/
COPY envs/my_env/ /app/envs/my_env/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "envs.my_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5. Update GitHub Actions Workflow

**Important**: To enable automatic Docker image builds on GitHub, add your environment to the workflow matrix.

Edit `.github/workflows/docker-build.yml` and add your environment to the matrix:

```yaml
strategy:
  matrix:
    image:
      - name: echo-env
        dockerfile: envs/echo_env/server/Dockerfile
      - name: chat-env
        dockerfile: envs/chat_env/server/Dockerfile
      - name: coding-env
        dockerfile: envs/coding_env/server/Dockerfile
      - name: my-env  # Add your environment here
        dockerfile: envs/my_env/server/Dockerfile
```

Once added, every push to `main` will automatically:
- Build your Docker image
- Push it to GitHub Container Registry as `ghcr.io/YOUR_USERNAME/openenv-my-env:latest`

### 6. Implement Client

Create a client that extends `EnvClient`:

```python
# client.py
from openenv.core import EnvClient, StepResult
from .models import MyAction, MyObservation, MyState


class MyEnv(EnvClient[MyAction, MyObservation, MyState]):
    def _step_payload(self, action: MyAction) -> dict:
        return {"command": action.command, "parameters": action.parameters}

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        obs = MyObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> MyState:
        return MyState(**payload)
```

## Building and Using Your Environment

### Build Docker Images

```bash
# First, build the base image (if not already built)
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Then build your environment image
docker build -t my-env:latest -f envs/my_env/server/Dockerfile .
```

### Use Your Environment

```python
from envs.my_env import MyAction, MyEnv

# Create environment from Docker image
client = MyEnv.from_docker_image("my-env:latest")

# Reset
result = client.reset()
print(result.observation.result)  # "Ready"

# Execute actions
result = client.step(MyAction(command="test", parameters={}))
print(result.observation.result)
print(result.observation.success)

# Get state
state = client.state()
print(state.episode_id)
print(state.step_count)

# Cleanup
client.close()
```

## Project Structure

Organize your environment following this structure:

```
envs/my_env/
├── __init__.py           # Export MyAction, MyObservation, MyState, MyEnv
├── models.py             # Action, Observation, State definitions
├── client.py             # MyEnv client implementation
├── README.md             # Environment documentation
└── server/
    ├── __init__.py
    ├── my_environment.py # Environment logic
    ├── app.py            # FastAPI application
    └── Dockerfile        # Docker image definition
```

## Example Environments

Study these examples to see the patterns in action:

### Echo Environment
Location: `envs/echo_env/`

A minimal environment that echoes messages back. Great for:
- Learning the basics
- Testing infrastructure
- Reference implementation

See: [`echo_env/README.md`](echo_env/README.md)

### Coding Environment
Location: `envs/coding_env/`

Executes Python code in a sandboxed environment. Demonstrates:
- Complex environment logic
- Error handling
- External tool integration (smolagents)

See: [`coding_env/README.md`](coding_env/README.md)

## Best Practices

### 1. Type Safety
Always use typed dataclasses for actions, observations, and state:
```python
@dataclass
class MyAction(Action):
    command: str  # Use explicit types
    count: int = 0  # Provide defaults when appropriate
```

### 2. Error Handling
Handle errors gracefully in your environment:
```python
def step(self, action: MyAction) -> MyObservation:
    try:
        result = self._process(action)
        return MyObservation(result=result, success=True)
    except Exception as e:
        return MyObservation(result="", success=False, error=str(e))
```

### 3. State Management
Track all relevant episode state:
```python
@dataclass
class MyState(State):
    # Add custom fields
    accumulated_reward: float = 0.0
    last_action: str = ""
```

### 4. Documentation
Provide comprehensive README for your environment:
- Overview and purpose
- Quick start example
- Action/Observation specifications
- Build instructions
- Usage examples

### 5. Testing
Test your environment before containerization:
```python
# test_my_environment.py
from envs.my_env.server.my_environment import MyEnvironment
from envs.my_env.models import MyAction


def test_environment():
    env = MyEnvironment()

    # Test reset
    obs = env.reset()
    assert obs.success

    # Test step
    action = MyAction(command="test", parameters={})
    obs = env.step(action)
    assert obs.success

    # Test state
    assert env.state.step_count == 1
```

## Advanced Topics

### Custom Transforms
Apply transformations to observations:

```python
from openenv.core.env_server import Transform


class MyTransform(Transform):
    def __call__(self, observation: Observation) -> Observation:
        # Transform observation
        return modified_observation


# Use in environment
env = MyEnvironment(transform=MyTransform())
```

### Additional Dependencies
Install environment-specific packages in Dockerfile:

```dockerfile
FROM openenv-base:latest

# Install specific versions
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    pandas==2.0.0 \
    your-custom-package==1.0.0
```
