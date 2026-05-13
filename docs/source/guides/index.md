# Guides

Practical how-to guides for working with OpenEnv. These guides are task-oriented and help you accomplish specific goals.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🔌 Using Environments
Learn how to connect to and interact with OpenEnv environments.

[Concepts →](concepts.md)
:::

:::{grid-item-card} 🛠️ Building Environments
Create your own custom environments for agentic training.

[Your First Environment →](first-environment.md)
:::

:::{grid-item-card} 🧠 Training
Integrate OpenEnv with RL frameworks for agent training.

[RL Integration →](rl-integration.md)
:::
::::

## Using Environments

- [**Concepts**](concepts.md) - Understand the core environment, action, observation, and reward abstractions
- [**Auto-Discovery (AutoEnv)**](auto-discovery.md) - Automatically discover and load environments
- [**Connecting to Servers**](connecting.md) - Connect via HTTP, Docker, or Hugging Face Spaces
- [**Async vs Sync Usage**](async-sync.md) - When and how to use async vs sync clients

## Building Environments

- [**Your First Environment**](first-environment.md) - Build a simple environment from scratch
- [**Environment Anatomy**](environment-anatomy.md) - Deep dive into environment structure
- [**Deployment**](deployment.md) - Deploy to Docker, Hugging Face Spaces, and registries
- [**Customizing the Web UI**](customizing-web-ui.md) - Customize your environment's built-in web interface

## Training

- [**RL Framework Integration**](rl-integration.md) - Use OpenEnv with TRL, torchforge, and more
- [**Reward Design**](rewards.md) - Design effective reward functions for your agents
- [**Simulation vs Production Mode**](simulation-vs-production.md) - When to use the training loop vs direct MCP access
- [**MCP Environment Lifecycle**](mcp-environment-lifecycle.md) - How MCP tools fit into the step loop

```{toctree}
:hidden:
:maxdepth: 1

concepts
auto-discovery
connecting
async-sync
simulation-vs-production
mcp-environment-lifecycle
first-environment
environment-anatomy
deployment
rl-integration
rewards
customizing-web-ui
```
