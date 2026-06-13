# Environments

> [!NOTE]
> The environments listed here may not reflect the latest additions. For the official OpenEnv collection, see the [OpenEnv organization on Hugging Face](https://huggingface.co/openenv). You may also find additional community environments tagged `agent-environment` on [Hugging Face Spaces](https://huggingface.co/spaces?category=agent-environment). The environments highlighted below are a curated selection.

The OpenEnv community has built a catalog of ready-to-run environments that cover deterministic smoke tests, full developer workflows, and multi-step reasoning challenges. Explore the surface area below and jump directly into the guides for each environment.

<div class="mt-6">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-4">
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Echo</div>
      <p class="text-sm">Minimal observation/action loop for verifying client integrations, CI pipelines, and onboarding flows in seconds.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/echo" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/openenv/echo_env" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Coding</div>
      <p class="text-sm">Secure sandbox with filesystem access and evaluation hooks for executing generated code and building autonomous dev workflows.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/coding" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/openenv/coding_env" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Jupyter</div>
      <p class="text-sm">Notebook-style coding environment backed by E2B with setup/verify hooks and a web UI for interactive runs.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/jupyter" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Terminus</div>
      <p class="text-sm">Terminal-first coding environment with high-contrast shell output and session controls for execute/verify/close flows.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/terminus" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Coding Tools</div>
      <p class="text-sm">SETA-style multi-tool coding environment with shell, file editing, search, todos, and submit verification.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/coding_tools" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Chat</div>
      <p class="text-sm">Message-driven loop tailored for conversational agents that need structured turns, safety rails, and message attribution.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/chat" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/openenv/chat_env" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Atari</div>
      <p class="text-sm">Classic Arcade Learning Environment tasks packaged for fast benchmarking of reinforcement-learning style agents.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/atari" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/openenv/atari_env" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">OpenSpiel</div>
      <p class="text-sm">Multi-agent, game-theory workloads powered by DeepMind's OpenSpiel suite, ideal for search and self-play experiments.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/openspiel" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/openenv/openspiel_env" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">SUMO-RL</div>
      <p class="text-sm">Traffic control scenarios with SUMO simulators for agents that reason about continuous control and scheduling.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/sumo" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">FinRL</div>
      <p class="text-sm">Financial market simulations with portfolio APIs, perfect for RLHF strategies and algorithmic trading experiments.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/finrl" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">TextArena</div>
      <p class="text-sm">Multi-task text arena for language-game competitions such as Wordle, reasoning puzzles, and program synthesis.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/textarena" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/burtenshaw/textarena_env" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Git</div>
      <p class="text-sm">Teaches agents to navigate repositories, inspect diffs, and land changes via Git-native operations.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/git" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">DIPG Safety</div>
      <p class="text-sm">Safety-critical diagnostics from the DIPG benchmark, highlighting guardrails, adversarial prompts, and risk scoring.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/dipg" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/surfiniaburger/dipg-gym" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Snake</div>
      <p class="text-sm">Classic snake game environment for RL research with configurable grids, partial observability, and customizable rewards.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/snake" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/Crashbandicoote2/snake_env" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Web Search</div>
      <p class="text-sm">Web search environment for RL research with configurable grids, partial observability, and customizable rewards.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/websearch" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/lawhy/web_search" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">BrowserGym</div>
      <p class="text-sm">Browser automation environment for web agents with DOM interaction, navigation, and multi-step task completion.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/browsergym" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/burtenshaw/browsergym-v2" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">KernRL</div>
      <p class="text-sm">RL environment for GPU kernel optimization. Train LLM agents to write fast CUDA/Triton kernels that beat baseline implementations.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/kernrl" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Calendar</div>
      <p class="text-sm">Calendar tool-use environment exposing a Calendar Gym through the OpenEnv reset/step/state interface for scheduling agents.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/calendar" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">CARLA</div>
      <p class="text-sm">Embodied evaluation environment for testing LLM decision-making in a full 3D driving simulator with irreversible consequences and ethical trolley scenarios.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/carla" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/sergiopaniego/carla-env" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Chess</div>
      <p class="text-sm">Chess RL environment powered by the moonfish engine with configurable opponents, position evaluation, and full chess rules.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/chess" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Connect4</div>
      <p class="text-sm">Classic Connect Four board game environment for training agents on turn-based strategy with a 6×7 grid.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/connect4" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">DM Control</div>
      <p class="text-sm">Generic OpenEnv wrapper for dm_control.suite, providing access to all MuJoCo-based continuous control tasks like cartpole, walker, and humanoid.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/dm_control" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">FinQA</div>
      <p class="text-sm">Financial question-answering environment that evaluates LLMs on complex financial questions using tool calls on SEC 10-K filing data.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/finqa" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Grid World</div>
      <p class="text-sm">Simple 5×5 grid world RL testbed and step-by-step guide for building new OpenEnv environments from scratch.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/grid_world" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/yuvrajpant56/grid_world_env" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Julia</div>
      <p class="text-sm">Julia code execution environment with test result tracking and reward calculation for RL training on Julia programming tasks.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/julia" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Maze</div>
      <p class="text-sm">Gridworld maze where agents navigate from start to exit while avoiding walls, with configurable 8×8 layouts.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/maze" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">OpenApp</div>
      <p class="text-sm">Web application simulation wrapping the OpenApps framework and BrowserGym for training UI agents on calendar, todo, messenger, and maps apps.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/openapp" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Reasoning Gym</div>
      <p class="text-sm">Integrates the Reasoning Gym library to provide single-step reasoning tasks with configurable datasets and scoring.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/reasoning_gym" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">REPL</div>
      <p class="text-sm">Python REPL environment for code execution tasks based on the Recursive Language Models paradigm with sandboxed execution and context loading.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/repl" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">TB2</div>
      <p class="text-sm">OpenEnv wrapper for Terminal-Bench 2 tasks with local and Docker execution modes for terminal-based agent evaluation.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/tbench2" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Unity</div>
      <p class="text-sm">OpenEnv wrapper for Unity ML-Agents environments, providing access to Unity's RL environments through HTTP/WebSocket interfaces.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/unity" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Wildfire</div>
      <p class="text-sm">Autonomous wildfire-control simulation where agents contain spreading fires using water, firebreaks, and timing under dynamic conditions.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/wildfire" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Agent World Model</div>
      <p class="text-sm">AgentWorldModel-1K — 1,000 synthetic MCP tool-use environments with 10,000 tasks for large-scale agentic RL training.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/agent_world_model" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/ChilleD/agent_world_model_env" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">OpenCode</div>
      <p class="text-sm"><code>opencode_env</code> runs the OpenCode coding agent inside an isolated E2B sandbox against any OpenAI-compatible LLM endpoint, optionally capturing per-token logprobs.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/opencode" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">Sophistry Bench Sprint</div>
      <p class="text-sm"><code>sophistry_bench_sprint_env</code> is a single-turn advocacy reward-hacking environment on QuALITY passages: the policy defends an assigned answer and the reward proxy peaks at 8 <code>&lt;claim&gt;</code> tags, with four weight-0 canaries that detect format hacking.</p>
      <div class="flex gap-2 mt-3">
        <a href="environments/sophistry_bench_sprint" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
      </div>
    </div>
  </div>
</div>

> [!TIP]
> Want to publish your own environment? Head over to the [Build Your Own Environment](getting_started/environment-builder.md) guide for a step-by-step walkthrough.

## Community Environments

<div class="mt-6">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-3 md:gap-4">
    <div class="border dark:border-gray-700 p-5 rounded-lg shadow">
      <div class="font-bold mb-2">RLVE Gym</div>
      <p class="text-sm">A suite of 400 environments that procedurally generate reasoning problems for LM training with configurable difficulty.</p>
      <div class="flex gap-2 mt-3">
        <a href="https://huggingface.co/spaces/ZhiyuanZeng/RLVE_Gym/blob/main/README.md" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">📄 Docs</a>
        <a href="https://huggingface.co/spaces/ZhiyuanZeng/RLVE_Gym" class="!no-underline border dark:border-gray-700 px-3 py-1 rounded text-sm hover:shadow">🤗 HF</a>
      </div>
    </div>
  </div>
</div>
