# Getting Started

Install OpenEnv, load an environment, and run your first step.

## Install OpenEnv

```bash
pip install openenv
```

> [!NOTE]
> This installs the `openenv` CLI and the `openenv.core` runtime. Environment
> projects can depend on `openenv[core]` when they only need the server and
> client libraries.

## Try an Environment

Use `AutoEnv` and `AutoAction` when you want OpenEnv to find the matching client
and action classes for an installed or discoverable environment.

```python
from openenv import AutoAction, AutoEnv

env = AutoEnv.from_env("echo")
EchoAction = AutoAction.from_env("echo")

with env.sync() as client:
    result = client.reset()
    print(result.observation.echoed_message)  # "Echo environment ready!"

    result = client.step(EchoAction(message="Hello, OpenEnv!"))
    print(result.observation.echoed_message)  # "Hello, OpenEnv!"
```

`AutoEnv.from_env()` accepts the common name forms:

```python
AutoEnv.from_env("echo")
AutoEnv.from_env("echo-env")
AutoEnv.from_env("echo_env")
```

## Connect to a Running Environment

OpenEnv clients are async by default. Use the async client for production code,
parallel environment runs, and integrations with async frameworks.

```python
import asyncio

from echo_env import EchoAction, EchoEnv


async def main():
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
        result = await client.reset()
        print(result.observation.echoed_message)

        result = await client.step(EchoAction(message="Hello, World!"))
        print(result.reward)


asyncio.run(main())
```

For scripts and notebooks, use `.sync()`:

```python
from echo_env import EchoAction, EchoEnv

with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello, World!"))
    print(result.observation.echoed_message)
```

## Use Containers or Local Servers

You can run an environment from a Docker image:

```python
import asyncio

from echo_env import EchoEnv


async def main():
    client = await EchoEnv.from_docker_image(
        "registry.hf.space/openenv-echo-env:latest"
    )
    async with client:
        result = await client.reset()
        print(result.observation)


asyncio.run(main())
```

Or connect to a local server:

```bash
cd path/to/echo-env
uv venv
source .venv/bin/activate
uv pip install -e .
uv run server --host 0.0.0.0 --port 8000
```

```python
from echo_env import EchoEnv

with EchoEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
```

## Next Steps

- [Concepts](guides/concepts.md)
- [Explore environments](environments.md)
- [Build your first environment](guides/first-environment.md)
