# OpenEnv: Agentic Execution Environments

<div class="hero">
  <p class="hero__subtitle">
    A unified framework for building, deploying, and interacting with isolated execution environments for agentic reinforcement learning—powered by simple, Gymnasium-style APIs.
  </p>
</div>

Training RL agents—especially in agentic settings like code generation, web browsing, or game playing—requires environments that are:

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🎮 Gymnasium-Style APIs
Familiar `step()`, `reset()`, and `state()` interface for seamless integration with existing RL frameworks.
:::

:::{grid-item-card} Container-First Design
Package environments as containers for consistent, reproducible deployments across any infrastructure.
:::

:::{grid-item-card} HTTP-Native
Deploy environments as HTTP services for distributed training and remote execution.
:::

:::{grid-item-card} Secure Isolation
Run untrusted agent code safely with sandboxed execution environments.
:::

:::{grid-item-card} Rich Environment Library
Pre-built environments for games, coding, web browsing, and more.
:::

:::{grid-item-card} CLI Tools
Powerful command-line interface for environment management and deployment.
:::
::::

## Getting Started

New to OpenEnv? Follow our recommended learning path:

1. **[Getting Started Series](tutorials/index)** — A 5-part series covering what OpenEnv is, how to use and build environments, and how to contribute. No GPU required.

2. **[Build Your Own Environment](auto_getting_started/environment-builder)** — The complete reference guide for creating, packaging, and deploying custom environments with Docker and Hugging Face Hub.

3. **[Simulation vs Production Mode](guides/simulation-vs-production)** — Understand when to use the training loop, when to expose MCP directly, and how tools behave in each mode.

4. **[MCP Environment Lifecycle](guides/mcp-environment-lifecycle)** — Understand how MCP tools fit into the OpenEnv step loop, when `step_async()` is used, and when to use `call_tool()` versus `step(...)`.

5. **[Explore Environments](environments)** — Browse pre-built environments for games, coding, web browsing, and more.

## How Can I Contribute?

We welcome contributions from the community! If you find a bug, have a feature request, or want to contribute a new environment, please open an issue or submit a pull request. The repository is hosted on GitHub at [huggingface/OpenEnv](https://github.com/huggingface/OpenEnv).

```{warning}
OpenEnv is currently in an experimental stage. You should expect bugs, incomplete features, and APIs that may change in future versions. The project welcomes bug fixes, but to ensure coordination, please discuss significant changes before starting work. Signal your intention to contribute in the issue tracker by filing a new issue or claiming an existing one.
```

```{toctree}
:maxdepth: 2
:caption: Get Started
:hidden:

getting-started
```

```{toctree}
:maxdepth: 2
:caption: Guides
:hidden:

guides/index
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
```
