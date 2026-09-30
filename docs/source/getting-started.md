# Getting Started

Install OpenEnv, load an environment, and run your first step.

## Install OpenEnv

```bash
pip install openenv
```

> [!NOTE]
> This installs the full OpenEnv runtime: the environment server, the client,
> the `openenv` CLI, the web interface, and MCP support. Environments depend on
> `openenv` directly.

### Optional dependencies

A few integrations ship as optional extras. Install them with
`pip install openenv[<extra>]`:

| Extra | Pulls in |
|-------|----------|
| `inspect` | The Inspect AI evaluation harness |
| `daytona`, `aca`, `modal`, `novita`, `tensorlake` | Cloud sandbox providers (see the Core API reference) |

## Try an Environment

Use `AutoEnv` to load the Echo client from its Hugging Face Space. It asks for
confirmation before installing the environment package, then connects to the
running Space. Echo exposes its actions as MCP tools.

```python
from openenv import AutoEnv

env = AutoEnv.from_env(
    "openenv/echo_env",
    base_url="https://openenv-echo-env.hf.space",
)

with env.sync() as client:
    result = client.reset()
    print(result.observation.metadata["message"])  # "Echo environment ready!"

    message = client.call_tool("echo_message", message="Hello, OpenEnv!")
    print(message)  # "Hello, OpenEnv!"
```

For installed environment packages, `AutoEnv.from_env()` also accepts the common
name forms below. Without a `base_url`, these start a local Docker container and
require Docker and the environment image:

```python
from openenv import AutoEnv

AutoEnv.from_env("echo")
AutoEnv.from_env("echo-env")
AutoEnv.from_env("echo_env")
```

## Connect to a Running Environment

OpenEnv clients are async by default. Use the async client for production code,
parallel environment runs, and integrations with async frameworks.

```python
import asyncio

from echo_env import CallToolAction, EchoEnv


async def main():
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
        await client.reset()

        result = await client.step(
            CallToolAction(
                tool_name="echo_message",
                arguments={"message": "Hello, World!"},
            )
        )
        print(result.reward)


asyncio.run(main())
```

For scripts and notebooks, use `.sync()`:

```python
from echo_env import CallToolAction, EchoEnv

with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as client:
    client.reset()
    result = client.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello, World!"},
        )
    )
    print(result.observation.result)
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

If you run the Hugging Face Space image yourself with `docker run`, expose port
`7860` and connect to that port:

```bash
docker run -it -p 7860:7860 --platform=linux/amd64 \
    registry.hf.space/openenv-echo-env:latest
```

```python
from echo_env import EchoEnv

with EchoEnv(base_url="http://localhost:7860").sync() as client:
    result = client.reset()
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

Cloud sandbox providers implement the same `ContainerProvider` contract as
local Docker: they start an isolated environment server and return a `base_url`
that an `EnvClient` can connect to directly, while keeping provider-specific
control-plane concepts (sandbox groups, projects, signed URLs, snapshots, egress
policy) inside the provider. Because the contract is provider-neutral, any hosted
runtime can implement it without changing the client/server protocol.

Providers shipped today: `LocalDockerProvider`, `DockerSwarmProvider`,
`UVProvider`, `DaytonaProvider`, `ACASandboxProvider` (Azure Container Apps
Sandboxes), `ModalProvider`, `NovitaSandboxProvider`, and `TensorlakeProvider`. A `KubernetesProvider`
is planned.

See the [Runtime Providers guide](guides/runtime-providers.md) for the full list,
install extras, and how to select a provider, and the
[Core API reference](reference/core.md) for each provider's API. The
provider-neutral invariants are described in the Cloud Sandbox Providers amendment
proposed in RFC 002 (env-spec).

## Next Steps

- [Concepts](guides/concepts.md)
- [Explore environments](environments.md)
- [Build your first environment](guides/first-environment.md)
