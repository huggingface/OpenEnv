# Runtime Providers

A runtime provider starts an environment server and returns a `base_url` that an
`EnvClient` connects to. Container providers implement the same
`ContainerProvider` contract, so switching from local Docker to a cloud sandbox
is a one-line change.

## Available providers

| Provider | Backend | Install | Status |
|----------|---------|---------|--------|
| `LocalDockerProvider` | Local Docker daemon | core | ✅ |
| `DockerSwarmProvider` | Docker Swarm cluster | core | ✅ |
| `UVProvider` | Local process via `uv` (no container) | core | ✅ |
| `DaytonaProvider` | Daytona cloud sandboxes | `pip install openenv[daytona]` | ✅ |
| `ACASandboxProvider` | Azure Container Apps Sandboxes | `pip install openenv[aca]` | ✅ |
| `ModalProvider` | Modal sandboxes | `pip install openenv[modal]` | ✅ |
| `NovitaSandboxProvider` | Novita AI sandboxes | `pip install openenv[novita]` | ✅ |
| `KubernetesProvider` | Kubernetes cluster | `pip install openenv[kubernetes]` | ✅ |

Cloud-provider SDKs are optional extras, imported lazily, so installing core
OpenEnv pulls in no cloud SDK. The core providers (`LocalDockerProvider`,
`DockerSwarmProvider`, `UVProvider`) are re-exported from the runtime package;
cloud providers are imported from their module:

```python
from openenv.core.containers.runtime import LocalDockerProvider  # core
from openenv.core.containers.runtime.daytona_provider import DaytonaProvider  # cloud
```

See the [Core API reference](../reference/core.md#container-providers) for each
provider's full API.

## Lifecycle

Container providers that store their source image on the provider can be owned
by the client. In this form, the client starts the provider on first connect,
waits for readiness, and stops the provider when the client closes:

```python
image = DaytonaProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
provider = DaytonaProvider(image=image)

async with MyEnv(provider=provider) as env:
    result = await env.reset()
    ...
```

`ModalProvider`, `DaytonaProvider`, and `ACASandboxProvider` support this
provider-owned flow. Providers that require an explicit image at
`start_container()` time, such as `LocalDockerProvider` and
`DockerSwarmProvider`, should still be started manually and passed in with the
returned `base_url`:

```python
base_url = provider.start_container(image)
provider.wait_for_ready(base_url, timeout_s=180)
try:
    async with MyEnv(base_url=base_url, provider=provider) as env:
        result = await env.reset()
        ...
finally:
    provider.stop_container()
```

`UVProvider` is not a container provider: it runs the server as a local process
and exposes `.start()` / `.wait_for_ready()` / `.stop()` instead.

## Reusing one server for multiple sessions

After a client has connected, call `new_session()` to open another independent
environment session against the same running server:

```python
async with MyEnv(provider=provider) as env:
    first = await env.reset()
    child = await env.new_session()
    second = await child.reset()
```

Child sessions are owned by the parent client: closing the parent also closes
any children it created. You can still close a child earlier when you no longer
need it. Server capacity limits still apply, so `new_session()` can fail while
opening the child WebSocket when the server has reached `MAX_CONCURRENT_ENVS`.

## Running many environments in parallel

Scaling out (for example, many concurrent RL rollouts) is a main reason cloud
providers exist. The model is one provider and one client per environment: each
provider starts its own isolated sandbox, so they run independently. Launch them
concurrently with `asyncio.gather`, wrapping the blocking provider calls in
`asyncio.to_thread` since most cloud SDKs are synchronous:

```python
async def run_one(env_id: int, image) -> str:
    provider = DaytonaProvider()
    base_url = await asyncio.to_thread(provider.start_container, image)
    try:
        await asyncio.to_thread(provider.wait_for_ready, base_url, 300)
        async with MyEnv(base_url=base_url, provider=provider) as env:
            result = await env.reset()
            return result.observation.text
    finally:
        await asyncio.to_thread(provider.stop_container)

image = DaytonaProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
results = await asyncio.gather(*(run_one(i, image) for i in range(20)))
```

Full example: [`examples/daytona_tbench2_concurrent.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/daytona_tbench2_concurrent.py)
spins up N sandboxes concurrently and reports per-stage timing.

## Per-provider setup

### ACASandboxProvider

Runs the server in an Azure Container Apps Sandbox. Install with
`pip install openenv[aca]`. Requires Azure credentials (`credential=None`
falls back to `DefaultAzureCredential`).

```python
from openenv.core.containers.runtime.aca_provider import ACASandboxProvider

provider = ACASandboxProvider(
    image="disk:my-env",
    subscription_id="<subscription-id>",
    resource_group="<resource-group>",
    sandbox_group="<sandbox-group>",
    region="eastus",   # used to derive the endpoint when endpoint=None
    endpoint=None,
    credential=None,   # defaults to DefaultAzureCredential()
    sdk_kwargs={},
)
```

### DaytonaProvider

Runs the server in a Daytona cloud sandbox. Install with
`pip install openenv[daytona]`. Requires the `DAYTONA_API_KEY` environment
variable.

```python
from openenv.core.containers.runtime.daytona_provider import DaytonaProvider

image = DaytonaProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
provider = DaytonaProvider(image=image)
```

Full examples: [`examples/daytona_tbench2_simple.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/daytona_tbench2_simple.py)
and [`examples/daytona_tbench2_concurrent.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/daytona_tbench2_concurrent.py).

### DockerSwarmProvider

Deploys the server as a service on a Docker Swarm cluster. Initializes Swarm
automatically when it is not already active.

```python
from openenv.core.containers.runtime import DockerSwarmProvider

provider = DockerSwarmProvider()
```

### KubernetesProvider

Runs one environment as one Pod and one Service in an existing namespace.
Install with `pip install openenv[kubernetes]`. The returned URL is the
in-cluster Service DNS name over `http://`, which is the connectivity choice
for this provider. `ModalProvider` and `ACASandboxProvider` still require
`https://`. Use this provider from a trainer in the same cluster; it does not
create a namespace, Deployment, Ingress, or port-forward.

```python
from openenv.core.containers.runtime.kubernetes_provider import KubernetesProvider

provider = KubernetesProvider(namespace="openenv")
base_url = provider.start_container("echo-env:latest")
provider.wait_for_ready(base_url)
provider.stop_container()
```

### LocalDockerProvider

Runs the server on the local Docker daemon. This is the default for
`from_docker_image`, so you rarely construct it explicitly.

```python
from openenv.core.containers.runtime import LocalDockerProvider

provider = LocalDockerProvider()
```

### ModalProvider

Runs the server in a Modal sandbox over an encrypted tunnel. Install with
`pip install openenv[modal]`. Requires a configured Modal account
(`modal setup`).

```python
from openenv.core.containers.runtime.modal_provider import ModalProvider

image = ModalProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
provider = ModalProvider(app_name="openenv", image=image)
```

Full example: [`examples/modal_echo_env.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/modal_echo_env.py).

### NovitaSandboxProvider

Runs the server in a Novita AI sandbox. Install with
`pip install openenv[novita]`. Requires the `NOVITA_API_KEY` environment
variable (and optionally `NOVITA_DOMAIN` to select a region — the default is
`us-phx-1`).

```python
from openenv.core.containers.runtime.novita_provider import NovitaSandboxProvider

provider = NovitaSandboxProvider(image="ghcr.io/org/echo-env:latest")
```

A local Dockerfile works too, and builds a Novita template on first use:

```python
image = NovitaSandboxProvider.image_from_dockerfile(
    "envs/echo_env/server/Dockerfile"
)
provider = NovitaSandboxProvider(image=image)
```

Novita's template parser does not accept multi-stage build definitions, which is
the layout every in-repo environment uses, so `image_from_dockerfile` rewrites
the Dockerfile before handing it over: BuildKit `--mount` flags are stripped,
`ARG`/`--platform` in `FROM` lines are resolved, and a two-stage build whose
stages share one base image is replayed as a single stage. A Dockerfile that
does not fit those rules raises with the registry route as the alternative —
build with `openenv build`, push with `openenv push --registry`, and pass the
resulting registry reference.

Two things the provider pins that the image does not: the sandbox runs as
`root` (Novita's parser otherwise rewrites USER to a non-root `user`, which
cannot write the root-owned `/app` the server installs into), and the template's
start command is a keepalive rather than the image's `CMD`, which leaves port
8000 free for the server the provider launches itself. That launch writes a PID
file, so `wait_for_ready` can report a crashed server immediately instead of
waiting out the full timeout.

### UVProvider

Runs the server as a local process via `uv`, without a container. Useful for
developing an environment from a checkout.

```python
from openenv.core.containers.runtime import UVProvider

provider = UVProvider(project_path="path/to/env")
base_url = provider.start()
provider.wait_for_ready()
```
