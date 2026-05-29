# <img width="35" height="35" alt="image" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> OpenEnv: Agentic Execution Environments

An e2e framework for creating, deploying and using isolated execution environments for agentic RL training, built using Gymnasium style simple APIs.

[![PyPI](https://img.shields.io/pypi/v/openenv-core?color=blue)](https://pypi.org/project/openenv-core/) 
[![Discord](https://img.shields.io/badge/Discord-OpenEnv-7289da?style=flat&logo=discord&logoColor=white)](https://discord.gg/YsTYBh6PD9)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb) 
[![Docs](https://img.shields.io/badge/Docs-Explore-blue?logo=readthedocs&logoColor=white)](https://meta-pytorch.org/OpenEnv/) 

---

**🚀 Featured Example:** Train LLMs to play BlackJack using [torchforge](https://github.com/meta-pytorch/torchforge) (PyTorch's agentic RL framework): [`examples/grpo_blackjack/`](examples/grpo_blackjack/)

**🔥 Zero to Hero Tutorial:** End to end tutorial from our [GPU Mode](tutorial/README.md) lecture and other hackathons.

## Quick Start

Install the OpenEnv core package:

```bash
pip install openenv-core
```

Install an environment client (e.g., Echo):

```bash
pip install git+https://huggingface.co/spaces/openenv/echo_env
```

Then use the environment:

```python
import asyncio
from echo_env import CallToolAction, EchoEnv

async def main():
    # Connect to a running Space (async context manager)
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
        # Reset the environment
        result = await client.reset()
        print(result.observation.echoed_message)  # "Echo environment ready!"

        # Send messages
        result = await client.step(
            CallToolAction(
                tool_name="echo_message",
                arguments={"message": "Hello, World!"},
            )
        )
        print(result.observation.result)  # "Hello, World!"
        print(result.reward)

asyncio.run(main())
```

**Synchronous usage** is also supported via the `.sync()` wrapper:

```python
from echo_env import CallToolAction, EchoEnv

# Use .sync() for synchronous context manager
with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as client:
    result = client.reset()
    result = client.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello, World!"},
        )
    )
    print(result.observation.result)
```

For a detailed quick start, check out the [docs page](https://meta-pytorch.org/OpenEnv/auto_getting_started/index.html).

## OpenEnv on partner platforms:

- [Lightning AI Studio](https://lightning.ai/environments?section=featured)
- [TRL example](https://huggingface.co/docs/trl/openenv)
- [Unsloth Google Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb)
- [ART example](https://art.openpipe.ai/integrations/openenv-integration)
- [Oumi example](https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20OpenEnv%20GRPO%20with%20trl.ipynb)

## Overview

OpenEnv provides a standard for interacting with agentic execution environments via simple Gymnasium style APIs - `step()`, `reset()`, `state()`. Users of agentic execution environments can interact with the environment during RL training loops using these simple APIs.

In addition to making it easier for researchers and RL framework writers, we also provide tools for environment creators making it easier for them to create richer environments and make them available over familiar protocols like HTTP and packaged using canonical technologies like docker. Environment creators can use the OpenEnv framework to create environments that are isolated, secure, and easy to deploy and use.

The OpenEnv CLI (`openenv`) provides commands to initialize new environments and deploy them to Hugging Face Spaces.

> ⚠️ **Early Development Warning** OpenEnv is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

### RFCs

Below is a list of active and historical RFCs for OpenEnv. RFCs are proposals for major changes or features. Please review and contribute!

- [RFC 001: Baseline API and Interface Specifications](https://github.com/meta-pytorch/OpenEnv/pull/26)
- [RFC 002: Discoverability of environment tools by agents](https://github.com/meta-pytorch/OpenEnv/pull/32)
- [RFC 003: Add MCP (Model Context Protocol) support](https://github.com/meta-pytorch/OpenEnv/pull/224)
- [RFC 004: Add delayed rewards support for trajectory-based scoring](https://github.com/meta-pytorch/OpenEnv/pull/337)
- [RFC 005: Agentic Harness Integration](https://github.com/meta-pytorch/OpenEnv/pull/387)

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Client Application                   │
│  ┌────────────────┐              ┌──────────────────┐   │
│  │  EchoEnv       │              │  CodingEnv       │   │
│  │  (EnvClient)   │              │   (EnvClient)    │   │
│  └────────┬───────┘              └────────┬─────────┘   │
└───────────┼───────────────────────────────┼─────────────┘
            │ WebSocket                     │ WebSocket
            │ (reset, step, state)          │
┌───────────▼───────────────────────────────▼─────────────┐
│              Docker Containers (Isolated)               │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │ FastAPI Server       │    │ FastAPI Server       │   │
│  │   EchoEnvironment    │    │ PythonCodeActEnv     │   │
│  │ (Environment base)   │    │ (Environment base)   │   │
│  └──────────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Web Interface

OpenEnv includes a built-in web interface for interactive environment exploration and debugging. The web interface provides:

- **Two-Pane Layout**: HumanAgent interaction on the left, state observation on the right
- **Real-time Updates**: WebSocket-based live updates without page refresh
- **Dynamic Forms**: Automatically generated action forms based on environment Action types
- **Action History**: Complete log of all actions taken and their results

The web interface is **conditionally enabled** based on environment variables:

- **Local Development**: Disabled by default for lightweight development
- **Manual Override**: Enable with `ENABLE_WEB_INTERFACE=true`

To use the web interface:

```python
from openenv.core.env_server import create_web_interface_app
from your_env.models import YourAction, YourObservation
from your_env.server.your_environment import YourEnvironment

env = YourEnvironment()
app = create_web_interface_app(env, YourAction, YourObservation)
```

When enabled, open `http://localhost:8000/web` in your browser to interact with the environment.

#### 2. Environment (Server-Side)
Base class for implementing environment logic:
- **`reset()`**: Initialize a new episode, returns initial `Observation`
- **`step(action)`**: Execute an `Action`, returns resulting `Observation`
- **`state()`**: Access episode metadata (`State` with episode_id, step_count, etc.)

#### 3. EnvClient (Client-Side)
Base class for environment communication:
- **Async by default**: Use `async with` and `await` for all operations
- **Sync wrapper**: Call `.sync()` to get a `SyncEnvClient` for synchronous usage
- Handles WebSocket connections to environment server
- Contains a utility to spin up a docker container locally for the corresponding environment
- Type-safe action/observation parsing

#### 4. Container Providers
Manage container deployment:
- `LocalDockerProvider`: Run containers on local Docker daemon
- `KubernetesProvider`: Deploy to K8s clusters (future)

#### 5. Models
Type-safe data structures:
- `Action`: Base class for environment actions
- `Observation`: Base class for environment observations
- `State`: Episode state tracking
- `StepResult`: Combines observation, reward, done flag

## Project Structure

### For Environment Creators

Use the CLI to quickly scaffold a new environment:

```bash
openenv init my_env
```

This creates the following structure:

```
my_env/
├── .dockerignore        # Docker build exclusions
├── __init__.py           # Export YourAction, YourObservation, YourEnv
├── models.py             # Define Action, Observation, State dataclasses
├── client.py             # Implement YourEnv(EnvClient)
├── README.md             # Document your environment
├── openenv.yaml          # Environment manifest
├── pyproject.toml        # Dependencies and package configuration
├── outputs/              # Runtime outputs (logs, evals) - gitignored
│   ├── logs/
│   └── evals/
└── server/
    ├── your_environment.py  # Implement YourEnvironment(Environment)
    ├── app.py               # Create FastAPI app
    ├── requirements.txt     # Dependencies for Docker (can be generated)
    └── Dockerfile           # Define container image
```

#### Dependency Management

OpenEnv uses `pyproject.toml` as the primary dependency specification:

- **Environment-level `pyproject.toml`**: Each environment defines its own dependencies
- **Root-level `pyproject.toml`**: Contains shared core dependencies (fastapi, pydantic, uvicorn)
- **Server `requirements.txt`**: Can be auto-generated from `pyproject.toml` for Docker builds

**Development Workflow:**

```bash
# Install environment in editable mode
cd my_env
pip install -e .

# Or using uv (faster)
uv pip install -e .

# Run server locally without Docker
uv run server --host 0.0.0.0 --port 8000
```

**Benefits:**
- ✅ **Client-side extensions**: Modify client classes locally without repo changes
- ✅ **Better dependency management**: Clear separation between environments
- ✅ **Flexible workflows**: Use pip, uv, or Docker for different scenarios
- ✅ **CI/CD ready**: Automated dependency generation and validation

See [`envs/README.md`](envs/README.md) for a complete guide on building environments.

### For Environment Users

To use an environment:
1. Install the client: `pip install git+https://huggingface.co/spaces/openenv/echo-env`
2. Import: `from echo_env import CallToolAction, EchoEnv`
3. Use async (recommended) or sync API:

**Async (recommended):**
```python
async with EchoEnv(base_url="...") as client:
    result = await client.reset()
    result = await client.step(action)
```

**Sync (via `.sync()` wrapper):**
```python
with EchoEnv(base_url="...").sync() as client:
    result = client.reset()
    result = client.step(action)
```

See example scripts in `examples/` directory.

## CLI Commands

The OpenEnv CLI provides commands to manage environments:

- **`openenv init <env_name>`** - Initialize a new environment from template
- **`openenv push [--repo-id <repo>] [--private]`** - Deploy environment to Hugging Face Spaces

### Quick Start

```bash
# Create a new environment
openenv init my_game_env

# Deploy to Hugging Face (will prompt for login if needed)
cd my_game_env
openenv push
```

For detailed options: `openenv init --help` and `openenv push --help`.

## Design Principles

1. **Separation of Concerns**: Clear client-server boundaries
2. **Type Safety**: Strongly-typed actions, observations, and state
3. **Container Isolation**: Each environment runs in its own container
4. **Simple APIs**: Minimal, intuitive interfaces

## Development

### Installation

```bash
# Clone the repository
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv

# Install core package in editable mode
pip install -e .
# Or using uv (faster)
uv pip install -e .
```

### Running Tests

OpenEnv uses a modular dependency structure: the core package is minimal, and each environment has its own dependencies. This means some tests require environment-specific packages.

```bash
# Install pytest (required for running tests)
uv pip install pytest

# Run all tests (skips tests requiring uninstalled dependencies)
PYTHONPATH=src:envs uv run pytest tests/ -v --tb=short

# Run a specific test file
PYTHONPATH=src:envs uv run pytest tests/envs/test_echo_environment.py -v
```

**To run environment-specific tests**, install that environment's dependencies:

```bash
# Example: Install coding_env with dev dependencies (includes smolagents + pytest)
uv pip install -e "envs/coding_env[dev]"

# Then run coding_env tests
PYTHONPATH=src:envs uv run pytest tests/envs/test_python_codeact_rewards.py -v
```

Tests will be automatically skipped if their required dependencies aren't installed.

## Requirements

- Python 3.10+
- Docker Desktop or Docker Engine
- FastAPI >= 0.104.0
- Uvicorn >= 0.24.0
- Requests >= 2.25.0
- Environment-specific dependencies (e.g., smolagents for coding_env)

## Supported RL Tools
The goal of this project is to support a broad set of open and closed tools to help standardize the agentic RL community. If you have a project that supports OpenEnv environments, please put up a PR to add your tool name along with a link to your documentation.

### torchforge
See GRPO BlackJack training example: [`examples/grpo_blackjack/`](examples/grpo_blackjack/)

### TRL
See the [TRL example](https://huggingface.co/docs/trl/openenv) on how to integrate OpenEnv environments with GRPO training.

### Unsloth
See the 2048 game example based on gpt-oss: [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb)

### SkyRL
See the [SkyRL example](https://skyrl.readthedocs.io/en/latest/examples/openenv.html) on how to train on OpenEnv environments with SkyRL.

### ART
See the [ART example](https://art.openpipe.ai/integrations/openenv-integration) on how OpenEnv environments can be used to train models with ART.

### Oumi
See the [Oumi example](https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20OpenEnv%20GRPO%20with%20trl.ipynb) on how OpenEnv environments can be used to train models with Oumi.

## Example Environments

| Environment | Description |
|---|---|
| [Echo Environment](envs/echo_env/README.md) | Echoes back messages with metadata. Ideal for testing HTTP server infrastructure, learning framework basics, and verifying container deployment. |
| [Coding Environment](envs/coding_env/README.md) | Sandboxed Python code execution via smolagents. Captures stdout/stderr/exit codes, supports persistent episode context, and provides detailed error handling. |
| [Chess Environment](envs/chess_env/README.md) | Chess RL environment with configurable opponents and full rules support. |
| [Atari Environment](envs/atari_env/README.md) | Classic Arcade Learning Environment tasks for RL benchmarking. |
| [FinRL Environment](envs/finrl_env/README.md) | Financial market simulations for algorithmic trading experiments. |

> Browse the full catalog of community environments at [meta-pytorch.org/OpenEnv/environments](https://meta-pytorch.org/OpenEnv/environments.html).

## Community Support & Acknowledgments
This is an open and community-centric project. If you would like to add your name here, please put up a pull request and tag @jspisak for review. Ty!!

Supporters include: Meta-PyTorch, Hugging Face, [Scaler AI Labs](https://scalerailabs.com), [Patronus AI](https://patronus.ai), [Surge AI](https://surgehq.ai), [LastMile AI](https://www.lastmileai.dev), Unsloth AI, Reflection AI, vLLM, SkyRL (UC-Berkeley), LightningAI, Axolotl AI, Stanford Scaling Intelligence Lab, Mithril, [OpenMined](https://openmined.org/), [Fleet AI](https://fleetai.com), [Halluminate](https://halluminate.ai/), [Turing](https://www.turing.com/), [Scale AI](https://scale.com/), [Scorecard](https://www.scorecard.io/) ..

And we'd also like to acknowledge the team at Farama Foundation as the OpenEnv API was heavily inspired by the work you all have done on Gymnasium. Cheers!

## License

BSD 3-Clause License (see [LICENSE](./LICENSE) file)
