# Auto-Discovery

OpenEnv provides a HuggingFace-style auto-discovery API that makes it easy to work with environments without manual imports.

## Overview

The auto-discovery system provides two main classes:

- **`AutoEnv`**: Automatically loads and instantiates environment clients
- **`AutoAction`**: Automatically loads action classes for environments

Both classes work with:

- **Local packages**: Installed via `pip install openenv-<env-name>`
- **HuggingFace Hub**: Environments hosted on HuggingFace Spaces

## Quick Start

### Basic Usage

Instead of manually importing specific environment classes:

```python
# Old way - requires knowing the module path
from coding_env import CodingEnv, CodeAction
```

You can now use the auto-discovery API:

```python
from openenv import AutoEnv, AutoAction

# Create environment (returns async client)
env = AutoEnv.from_env("coding-env")

# Get action class
CodeAction = AutoAction.from_env("coding-env")

# Use with sync wrapper for simple scripts
with env.sync() as client:
    result = client.reset()
    action = CodeAction(code="print('Hello, OpenEnv!')")
    step_result = client.step(action)
```

## AutoEnv API

### `AutoEnv.from_env(name, **kwargs)`

Create an environment client from a name or HuggingFace Hub repository.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `name` | Environment name or Hub repo ID. Local: `"coding"`, `"coding-env"`, `"coding_env"`. Hub: `"meta-pytorch/coding-env"`, `"username/env-name"` |
| `base_url` | Optional base URL for HTTP connection |
| `docker_image` | Optional Docker image name (overrides default) |
| `container_provider` | Optional container provider |
| `wait_timeout` | Timeout for container startup (default: 30s) |
| `env_vars` | Optional environment variables for the container |
| `**kwargs` | Additional arguments passed to the client class |

**Returns:** Instance of the environment client class

**Examples:**

```python
from openenv import AutoEnv

# From installed package
env = AutoEnv.from_env("coding-env")

# From HuggingFace Hub
env = AutoEnv.from_env("meta-pytorch/coding-env")

# With custom configuration
env = AutoEnv.from_env(
    "coding",
    docker_image="my-coding-env:v2",
    wait_timeout=60.0,
    env_vars={"DEBUG": "1"}
)
```

### `AutoEnv.list_environments()`

List all available environments.

```python
from openenv import AutoEnv

AutoEnv.list_environments()
# Output:
# Available Environments:
# ----------------------------------------------------------------------
# coding         : Coding environment for OpenEnv (v0.1.0)
# echo           : echo_env environment (v0.1.0)
# browsergym     : BrowserGym environment (v0.1.0)
# ...
```

### `AutoEnv.get_env_info(name)`

Get detailed information about an environment.

```python
from openenv import AutoEnv

info = AutoEnv.get_env_info("coding")
print(f"Description: {info['description']}")
print(f"Version: {info['version']}")
print(f"Docker Image: {info['default_image']}")
print(f"Client Class: {info['env_class']}")
print(f"Action Class: {info['action_class']}")
```

### `AutoEnv.get_env_class(name)`

Get the environment class (not an instance).

```python
from openenv import AutoEnv

CodingEnv = AutoEnv.get_env_class("coding")
# Now you can instantiate it yourself with custom parameters
env = CodingEnv.from_docker_image("coding-env:latest", wait_timeout=60.0)
```

## AutoAction API

### `AutoAction.from_env(name)`

Get the Action class from an environment name or HuggingFace Hub repository.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `name` | Environment name or Hub repo ID |

**Returns:** Action class (not an instance!)

**Examples:**

```python
from openenv import AutoAction

# From installed package
CodeAction = AutoAction.from_env("coding-env")
action = CodeAction(code="print('Hello!')")

# From HuggingFace Hub
CodeAction = AutoAction.from_env("meta-pytorch/coding-env")

# Different name formats work
EchoAction = AutoAction.from_env("echo")
EchoAction = AutoAction.from_env("echo-env")
EchoAction = AutoAction.from_env("echo_env")
```

### `AutoAction.from_hub(env_name)`

Alias for `from_env()` for backward compatibility.

```python
from openenv import AutoAction

CodeAction = AutoAction.from_env("coding")
action = CodeAction(code="x = 5 + 3")
```

### `AutoAction.list_actions()`

List all available action classes.

```python
from openenv import AutoAction

AutoAction.list_actions()
# Output:
# Available Action Classes:
# ----------------------------------------------------------------------
# coding         : CodeAction
# echo           : EchoAction
# browsergym     : BrowsergymAction
# ...
```

### `AutoAction.get_action_info(name)`

Get detailed information about an action class.

```python
from openenv import AutoAction

info = AutoAction.get_action_info("coding")
print(f"Action Class: {info['action_class']}")
print(f"Module: {info['module']}")
```

## HuggingFace Hub Integration

### Loading from HuggingFace Spaces

AutoEnv can automatically connect to environments running on HuggingFace Spaces:

```python
from openenv import AutoEnv, AutoAction

# Load from HuggingFace Space
env = AutoEnv.from_env("username/coding-env-test")

# Get action class
CodeAction = AutoAction.from_env("username/coding-env-test")

# Use with sync wrapper
with env.sync() as client:
    result = client.reset()
    action = CodeAction(code="print('Hello from HF Space!')")
    step_result = client.step(action)
    print(f"Output: {step_result.observation.stdout}")
```

The system automatically:
1. Detects HuggingFace repo IDs (format: `username/repo-name`)
2. Resolves the Space URL (e.g., `https://username-repo-name.hf.space`)
3. Checks if the Space is running and accessible
4. Installs the environment package using `git+` URL (prompts for confirmation)
5. Connects to the running Space

### Security: Remote Code Installation

When loading environments from HuggingFace Hub, AutoEnv needs to install Python code from the remote repository. Since this executes code from the internet, AutoEnv will prompt for confirmation before installing:

```
============================================================
SECURITY WARNING: Remote Code Installation
============================================================
You are about to install code from a remote repository:
  Repository: username/coding-env-test
  Source: https://huggingface.co/spaces/username/coding-env-test

This will execute code from the internet on your machine.
Only proceed if you trust the source.
============================================================

Do you want to proceed? [y/N]:
```

To skip the confirmation prompt, you can either:

1. **Use the `trust_remote_code` parameter:**
   ```python
   env = AutoEnv.from_env("username/coding-env", trust_remote_code=True)
   ```

2. **Set the environment variable:**
   ```bash
   export OPENENV_TRUST_REMOTE_CODE=1
   python your_script.py
   ```

### Package Installation

AutoEnv uses `uv pip` if available, otherwise falls back to standard `pip`. This ensures compatibility with different Python environments:

```bash
# If uv is installed, AutoEnv uses:
uv pip install git+https://huggingface.co/spaces/username/coding-env

# Otherwise, it uses:
pip install git+https://huggingface.co/spaces/username/coding-env
```

## Complete Workflow Example

Here's a complete example showing the auto-discovery workflow:

```python
from openenv import AutoEnv, AutoAction

# 1. List available environments
print("Available environments:")
AutoEnv.list_environments()

# 2. Create environment and get action class
env = AutoEnv.from_env("coding-env")
CodeAction = AutoAction.from_env("coding-env")

# 3. Use with sync wrapper for simple scripts
with env.sync() as client:
    # Reset environment
    result = client.reset()
    print(f"Environment ready: {result.observation}")

    # Execute actions
    action = CodeAction(code="""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(f"Fibonacci(10) = {fibonacci(10)}")
""")

    step_result = client.step(action)
    print(f"Output:\n{step_result.observation.stdout}")
```

For async usage (recommended for production):

```python
import asyncio
from coding_env import CodingEnv, CodeAction

async def main():
    async with CodingEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        result = await client.step(CodeAction(code="print('async!')"))
        print(result.observation.stdout)

asyncio.run(main())
```

## Error Handling

The auto-discovery API provides helpful error messages:

```python
from openenv import AutoEnv

try:
    env = AutoEnv.from_env("nonexistent-env")
except ValueError as e:
    print(e)
    # Output:
    # Unknown environment 'nonexistent'.
    # Did you mean: coding?
    # Available environments: atari, browsergym, chat, coding, ...
```

For typos, it suggests similar environment names:

```python
try:
    env = AutoEnv.from_env("cooding-env")  # Typo
except ValueError as e:
    print(e)
    # Output:
    # Unknown environment 'cooding'.
    # Did you mean: coding?
    # Available environments: ...
```

## Flexible Name Formats

AutoEnv accepts multiple name formats:

```python
from openenv import AutoEnv

# All of these work and refer to the same environment:
env = AutoEnv.from_env("coding")           # Simple name
env = AutoEnv.from_env("coding-env")       # With suffix
env = AutoEnv.from_env("coding_env")       # With underscore
env = AutoEnv.from_env("coding-env:latest") # With tag (ignored)
```

## How It Works

The auto-discovery system works by:

1. **Package Discovery**: Uses `importlib.metadata` to find installed `openenv-*` packages
2. **Manifest Loading**: Reads `openenv.yaml` files from package resources
3. **Caching**: Caches discovery results for performance
4. **Lazy Loading**: Only imports classes when actually needed
5. **Hub Support**: Downloads and installs packages from HuggingFace Hub on-demand

### Environment Packages

Environments are distributed as installable Python packages:

```bash
# Install an environment
pip install openenv-coding-env

# Now it's automatically discoverable
python -c "from openenv import AutoEnv; AutoEnv.list_environments()"
```

Each environment package includes:

- Client classes (e.g., `CodingEnv`)
- Action/Observation models (e.g., `CodeAction`, `CodeObservation`)
- Server Docker image
- `openenv.yaml` manifest describing the environment

### Manifest Format

Each environment includes an `openenv.yaml` file:

```yaml
name: coding_env
version: 0.1.0
description: Coding environment for OpenEnv

client:
  class_name: CodingEnv
  module: coding_env.client

action:
  class_name: CodeAction
  module: coding_env.client

observation:
  class_name: CodeObservation
  module: coding_env.client

default_image: coding-env:latest
spec_version: 1
```

## Benefits

| Benefit | Description |
|---------|-------------|
| **Simple** | No need to know which module to import from |
| **Flexible** | Works with local packages and HuggingFace Hub |
| **Discoverable** | List and explore available environments |
| **Type-Safe** | Returns properly typed environment classes |
| **HuggingFace-style** | Familiar API for ML practitioners |
| **Performant** | Caching and lazy loading for efficiency |

## See Also

- [Your First Environment](first-environment.md) - How to create your own environments
- [Core API Documentation](../core.md) - Low-level API details
- [HuggingFace Hub](https://huggingface.co/meta-pytorch) - Pre-built environments
