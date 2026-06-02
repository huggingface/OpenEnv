# Guides

Practical how-to guides for working with OpenEnv. These guides are task-oriented and help you accomplish specific goals.

<div class="mt-6">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-4">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-md" href="concepts">
      <div class="font-bold mb-2">🔌 Using Environments</div>
      <p>Learn how to connect to and interact with OpenEnv environments.</p>
      <p class="text-sm font-medium">Concepts →</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-md" href="first-environment">
      <div class="font-bold mb-2">🛠️ Building Environments</div>
      <p>Create your own custom environments for agentic training.</p>
      <p class="text-sm font-medium">Your First Environment →</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-md" href="rl-integration">
      <div class="font-bold mb-2">🧠 Training</div>
      <p>Integrate OpenEnv with RL frameworks for agent training.</p>
      <p class="text-sm font-medium">RL Integration →</p>
    </a>
  </div>
</div>

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

