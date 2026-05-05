# OpenEnv: Agentic Execution Environments

<div class="hero">
  <p class="hero__subtitle">
    A unified framework for building, deploying, and interacting with isolated execution environments for agentic reinforcement learning—powered by simple, Gymnasium-style APIs.
  </p>
  <div class="hero__actions">
    <a class="hero__button hero__button--primary" href="auto_getting_started/index.html">
      Getting Started Tutorials
    </a>
    <a class="hero__button" href="auto_getting_started/environment-builder.html">
      Build Your Own Environment
    </a>
    <a class="hero__button" href="environments.html">
      Explore Environments
    </a>
  </div>
  <div>
    <a href="https://discord.gg/YsTYBh6PD9"><img src="https://camo.githubusercontent.com/aa8bf380611b9abd47f42596a15b4842eaf01af84c86cc97001d2d5d166ef8c0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446973636f72642d4f70656e456e762d3732383964613f7374796c653d666c6174266c6f676f3d646973636f7264266c6f676f436f6c6f723d7768697465"></a>
    <a href="https://pypi.org/project/openenv-core/"><img src="https://img.shields.io/pypi/v/openenv-core?color=blue"></a>
    <a href="https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  </div>
</div>

## What is OpenEnv?

**OpenEnv** is an end-to-end framework designed to standardize how agents interact with execution environments during reinforcement learning (RL) training. At its core, OpenEnv provides a consistent, Gymnasium-compatible interface through three simple APIs: `step()`, `reset()`, and `state()`.

### Why OpenEnv?

Training RL agents—especially in agentic settings like code generation, web browsing, or game playing—requires environments that are:

- **Isolated**: Each agent instance runs in its own sandboxed environment, preventing interference and ensuring reproducibility.
- **Scalable**: Environments can be deployed as HTTP services or containerized with Docker, enabling distributed training across clusters.
- **Standardized**: A unified API means researchers and practitioners can switch between environments without rewriting integration code.

OpenEnv bridges the gap between environment creators and RL practitioners:

- **For Researchers & Framework Authors**: Interact with any OpenEnv-compatible environment using familiar Gymnasium-style APIs—no need to learn environment-specific protocols.
- **For Environment Creators**: Build rich, production-ready environments with built-in support for HTTP deployment, Docker packaging, and security isolation.

### Key Features

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🎮 Gymnasium-Style APIs
Familiar `step()`, `reset()`, and `state()` interface for seamless integration with existing RL frameworks.
:::

:::{grid-item-card} 🐳 Docker-First Design
Package environments as containers for consistent, reproducible deployments across any infrastructure.
:::

:::{grid-item-card} 🌐 HTTP-Native
Deploy environments as HTTP services for distributed training and remote execution.
:::

:::{grid-item-card} 🔒 Secure Isolation
Run untrusted agent code safely with sandboxed execution environments.
:::

:::{grid-item-card} 📦 Rich Environment Library
Pre-built environments for games, coding, web browsing, and more.
:::

:::{grid-item-card} 🛠️ CLI Tools
Powerful command-line interface for environment management and deployment.
:::
::::

## Getting Started

New to OpenEnv? Follow our recommended learning path:

1. **[Getting Started Tutorials](auto_getting_started/index)** — A hands-on, 3-part series covering what OpenEnv is, how to use existing environments, and how to build your own.

2. **[Build Your Own Environment](auto_getting_started/environment-builder)** — The complete reference guide for creating, packaging, and deploying custom environments with Docker and Hugging Face Hub.

3. **[Simulation vs Production Mode](simulation-vs-production)** — Understand when to use the training loop, when to expose MCP directly, and how tools behave in each mode.

4. **[MCP Environment Lifecycle](mcp-environment-lifecycle)** — Understand how MCP tools fit into the OpenEnv step loop, when `step_async()` is used, and when to use `call_tool()` versus `step(...)`.

5. **[Explore Environments](environments)** — Browse pre-built environments for games, coding, web browsing, and more.

## How Can I Contribute?

We welcome contributions from the community! If you find a bug, have a feature request, or want to contribute a new environment, please open an issue or submit a pull request. The repository is hosted on GitHub at [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv).

```{warning}
OpenEnv is currently in an experimental stage. You should expect bugs, incomplete features, and APIs that may change in future versions. The project welcomes bug fixes, but to ensure coordination, please discuss significant changes before starting work. Signal your intention to contribute in the issue tracker by filing a new issue or claiming an existing one.
```

```{toctree}
:maxdepth: 2
:caption: Get Started
:hidden:

quickstart
installation
concepts
auto_getting_started/index
```

```{toctree}
:maxdepth: 2
:caption: Guides
:hidden:

guides/index
guides/auto-discovery
guides/connecting
guides/async-sync
simulation-vs-production
mcp-environment-lifecycle
guides/first-environment
guides/environment-anatomy
guides/deployment
guides/rl-integration
guides/rewards
customizing-web-ui
```

```{toctree}
:maxdepth: 2
:caption: Tutorials
:hidden:

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: Environments
:hidden:

environments
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

reference/index
core
cli
auto_discovery
```

```{toctree}
:maxdepth: 2
:caption: Community
:hidden:

contributing
auto_getting_started/contributing-envs
```
