# Connecting to Servers

:::{note}
Coming Soon
    This page is under construction.

Learn how to connect

Learn how to connect to OpenEnv environments via different methods.

## Connection Methods

### HTTP URL (Remote Servers)

Connect to environments running on remote servers or Hugging Face Spaces:

```python
from openenv import AutoEnv

# HuggingFace Space
env = AutoEnv.from_env("openenv/echo_env")

# Direct URL
env = AutoEnv.from_env("echo", base_url="http://your-server:8000")
```

### Docker (Local Development)

Run environments locally using Docker:

```python
from openenv import AutoEnv

env = AutoEnv.from_env(
    "coding",
    docker_image="coding-env:latest",
    wait_timeout=60.0
)
```

### Hugging Face Spaces

OpenEnv environments can be hosted on Hugging Face Spaces for easy sharing:

```python
env = AutoEnv.from_env("username/my-environment")
```

## Next Steps

- [Auto-Discovery](auto-discovery.md) - Discover available environments
- [Async vs Sync Usage](async-sync.md) - Choose the right client mode
:::
