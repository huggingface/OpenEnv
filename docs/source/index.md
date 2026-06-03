# OpenEnv: Agentic Execution Environments

<div class="hero">
  <p class="hero__subtitle">
    A unified framework for building, deploying, and interacting with isolated execution environments for agentic reinforcement learning—powered by simple, Gymnasium-style APIs.
  </p>
</div>

Training RL agents—especially in agentic settings like code generation, web browsing, or game playing—requires environments that are:

<div class="mt-6">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-4">
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Gymnasium-Style APIs</div>
      <p>Familiar <code>step()</code>, <code>reset()</code>, and <code>state()</code> interface for seamless integration with existing RL frameworks.</p>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Container-First Design</div>
      <p>Package environments as containers for consistent, reproducible deployments across any infrastructure.</p>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">HTTP-Native</div>
      <p>Deploy environments as HTTP services for distributed training and remote execution.</p>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Secure Isolation</div>
      <p>Run untrusted agent code safely with sandboxed execution environments.</p>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Rich Environment Library</div>
      <p>Pre-built environments for games, coding, web browsing, and more.</p>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">CLI Tools</div>
      <p>Powerful command-line interface for environment management and deployment.</p>
    </div>
  </div>
</div>

## Getting Started

New to OpenEnv? Follow our recommended learning path:

1. **[Getting Started Series](tutorials/index)** — A 5-part series covering what OpenEnv is, how to use and build environments, and how to contribute. No GPU required.

2. **[Build Your Own Environment](getting_started/environment-builder)** — The complete reference guide for creating, packaging, and deploying custom environments with Docker and Hugging Face Hub.

3. **[Simulation vs Production Mode](guides/simulation-vs-production)** — Understand when to use the training loop, when to expose MCP directly, and how tools behave in each mode.

4. **[MCP Environment Lifecycle](guides/mcp-environment-lifecycle)** — Understand how MCP tools fit into the OpenEnv step loop, when `step_async()` is used, and when to use `call_tool()` versus `step(...)`.

5. **[Explore Environments](environments)** — Browse pre-built environments for games, coding, web browsing, and more.

## How Can I Contribute?

We welcome contributions from the community! OpenEnv is openly governed by a technical committee that includes Hugging Face, Unsloth, Reflection, and Meta PyTorch. The committee coordinates project direction, major technical decisions, RFCs, and release planning through the public repository.

If you find a bug, have a feature request, or want to contribute a new environment, please open an issue or submit a pull request. The repository is hosted on GitHub at [huggingface/OpenEnv](https://github.com/huggingface/OpenEnv).

> [!WARNING]
> OpenEnv is currently in an experimental stage. You should expect bugs, incomplete features, and APIs that may change in future versions. The project welcomes bug fixes, but significant changes should be discussed before implementation so the technical committee and community can coordinate scope, compatibility, and release timing. Signal your intention to contribute in the issue tracker by filing a new issue or claiming an existing one.

