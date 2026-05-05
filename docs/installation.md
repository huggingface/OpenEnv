# Installation

On this page we will walk you through the process of installing and using OpenEnv environments. If you want to build your own environment, please see the [Building an Environment](guides/first-environment.md) page.

## Install OpenEnv

To install the OpenEnv package, you can use the following command:

```bash
pip install openenv-core
```

!!! note
    This installs both the `openenv` CLI and the `openenv.core` runtime. Environment projects can depend on `openenv-core[core]` if they only need the server/client libraries.

### Using the Echo Environment (Example)

Let's start by using the Echo Environment. This is a simple environment that echoes back messages.

Install the echo environment client package:

```bash
pip install git+https://huggingface.co/spaces/openenv/echo-env
```

Then you can use the environment. The client is **async by default**:

```python
import asyncio
from echo_env import EchoAction, EchoEnv

async def main():
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
        # Reset the environment
        result = await client.reset()
        print(result.observation.echoed_message)  # "Echo environment ready!"

        # Send messages
        result = await client.step(EchoAction(message="Hello, World!"))
        print(result.observation.echoed_message)  # "Hello, World!"
        print(result.reward)  # 1.3 (based on message length)

asyncio.run(main())
```

For **synchronous usage**, use the `.sync()` wrapper:

```python
from echo_env import EchoAction, EchoEnv

with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello, World!"))
    print(result.observation.echoed_message)
```

### Using environments from Hugging Face

You can also use environments from Hugging Face. To do this, you can use the `from_env` method of the environment class.

```python
import asyncio
from echo_env import EchoEnv

async def main():
    # Pulls from Hugging Face and starts a container
    client = await EchoEnv.from_env("openenv/echo_env")
    async with client:
        result = await client.reset()
        print(result.observation)

asyncio.run(main())
```

In the background, the environment will be pulled from Hugging Face and a container will be started on your local machine.

You can also connect to the remote space on Hugging Face by passing the base URL to the environment class.

```python
async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
    result = await client.reset()
```

### Using Docker containers

You can also use environments from Docker containers. To do this, you can use the `from_docker_image` method of the environment class.

```python
import asyncio
from echo_env import EchoEnv

async def main():
    client = await EchoEnv.from_docker_image("registry.hf.space/openenv-echo-env:latest")
    async with client:
        result = await client.reset()
        print(result.observation)

asyncio.run(main())
```

In the background, the environment will be pulled from Docker Hub and a container will be started on your local machine.

As above, you can also connect to the Docker container by passing the base URL to the environment class.

```sh
docker run -p 8000:8000 registry.hf.space/openenv-echo-env:latest
```

Then you can use the environment via its HTTP interface.

```python
# Async
async with EchoEnv(base_url="http://localhost:8000") as client:
    result = await client.reset()

# Or sync
with EchoEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
```

### Using AutoEnv and AutoAction (Recommended)

The `AutoEnv` and `AutoAction` classes provide a HuggingFace-style auto-discovery API that automatically selects and instantiates the correct environment client and action classes without manual imports.

!!! note
    `AutoEnv.from_env()` returns a synchronous client by default for convenience. For async usage, use the client class directly.

```python
from openenv import AutoEnv, AutoAction

# Load environment from installed package (returns sync client)
env = AutoEnv.from_env("echo-env")

# Get the action class
EchoAction = AutoAction.from_env("echo-env")

# Use them together (sync API)
with env.sync() as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello!"))
    print(result.observation.echoed_message)  # "Hello!"
```

AutoEnv supports multiple name formats - all of these work:

```python
env = AutoEnv.from_env("echo")       # Short name
env = AutoEnv.from_env("echo-env")   # With suffix
env = AutoEnv.from_env("echo_env")   # Underscore variant
```

You can also load environments directly from HuggingFace Hub:

```python
# From Hub repo ID - auto-downloads and installs if needed
env = AutoEnv.from_env("meta-pytorch/coding-env")
CodeAction = AutoAction.from_env("meta-pytorch/coding-env")

# If the Space is running, connects directly without local Docker
# If not, falls back to local Docker mode
```

To see all available environments:

```python
AutoEnv.list_environments()
AutoAction.list_actions()
```

### Using environments from a local directory

You can also use environments from a local directory. To do this, navigate to the directory of the environment and start the server.

```bash
cd path/to/echo-env

# manage dependencies with uv
uv venv
source .venv/bin/activate
uv pip install -e .

# start the server
uv run server --host 0.0.0.0 --port 8000
# or
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then you can use the environment via its HTTP interface.

```python
from echo_env import EchoEnv

# Async (recommended)
async with EchoEnv(base_url="http://localhost:8000") as client:
    result = await client.reset()

# Or sync
with EchoEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
```

## Async vs Sync: When to Use Each

OpenEnv clients are **async by default** to support efficient concurrent operations. Use:

- **Async (`async with`, `await`)**: Best for production, parallel environments, and integration with async frameworks
- **Sync (`.sync()` wrapper)**: Convenient for scripts, notebooks, and synchronous codebases

```python
# Async - parallel environment interactions
async def run_parallel():
    async with EchoEnv(base_url="...") as env1, EchoEnv(base_url="...") as env2:
        # Run in parallel
        result1, result2 = await asyncio.gather(
            env1.step(action1),
            env2.step(action2)
        )

# Sync - simple sequential usage
with EchoEnv(base_url="...").sync() as env:
    result = env.step(action)
```

## Nice work! You've now used an OpenEnv environment.

Your next steps are to:

- [Check out the environments](environments/index.md)
- [Try out the end-to-end tutorial](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb)
- [Build your own environment](guides/first-environment.md)
