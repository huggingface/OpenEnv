# Quick Start

Get up and running with OpenEnv in under 5 minutes.

## Install

```bash
pip install openenv-core
```

## Try an Environment

```python
from openenv import AutoEnv, AutoAction

# Load the echo environment
env = AutoEnv.from_env("echo")
EchoAction = AutoAction.from_env("echo")

# Use it
with env.sync() as client:
    result = client.reset()
    print(result.observation.echoed_message)  # "Echo environment ready!"

    result = client.step(EchoAction(message="Hello, OpenEnv!"))
    print(result.observation.echoed_message)  # "Hello, OpenEnv!"
```

## What's Next?

- [Installation](installation.md) - Detailed setup options and usage patterns
- [Core Concepts](concepts.md) - Understand the key abstractions
- [Explore Environments](environments.md) - See all available environments
- [Build Your Own](guides/first-environment.md) - Create a custom environment
