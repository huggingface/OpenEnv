# Environments

The OpenEnv community has built a catalog of ready-to-run environments that cover deterministic smoke tests, full developer workflows, and multi-step reasoning challenges. Each environment follows the standard OpenEnv API (`step()`, `reset()`, `state()`) and can be deployed via Docker or connected to directly.

## Quick Start

```python
from openenv import AutoEnv

# Load any environment by name
env = AutoEnv("echo")  # Simple echo environment for testing
env = AutoEnv("chess")  # Chess game environment
env = AutoEnv("coding")  # Code execution environment
```

## Community Environments

<div class="environment-grid">
  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Echo</span>
      <p class="environment-card__description">
        Minimal observation/action loop for verifying client integrations, CI pipelines, and onboarding flows in seconds.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🎮 Games</span>
      <a href="echo/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/openenv/echo_env">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Snake</span>
      <p class="environment-card__description">
        Classic snake game environment for RL research with configurable grids, partial observability, and customizable rewards.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🎮 Games</span>
      <a href="snake/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/Crashbandicoote2/snake_env">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Chess</span>
      <p class="environment-card__description">
        Chess RL environment powered by the moonfish engine with configurable opponents, PSQT evaluation, and full rules support.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🎮 Games</span>
      <a href="chess/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/luccabb/moonfish_chess">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Atari</span>
      <p class="environment-card__description">
        Classic Arcade Learning Environment tasks packaged for fast benchmarking of reinforcement-learning style agents.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🎮 Games</span>
      <a href="atari/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/openenv/atari_env">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">OpenSpiel</span>
      <p class="environment-card__description">
        Multi-agent, game-theory workloads powered by DeepMind's OpenSpiel suite, ideal for search and self-play experiments.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🎮 Games</span>
      <a href="openspiel/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/openenv/openspiel_env">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">TextArena</span>
      <p class="environment-card__description">
        Multi-task text arena for language-game competitions such as Wordle, reasoning puzzles, and program synthesis.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🎮 Games</span>
      <a href="textarena/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/burtenshaw/textarena_env">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Coding</span>
      <p class="environment-card__description">
        Secure sandbox with filesystem access and evaluation hooks for executing generated code and building autonomous dev workflows.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">💻 Code</span>
      <a href="coding/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/openenv/coding_env">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">REPL</span>
      <p class="environment-card__description">
        Python REPL environment for training language models on code execution tasks. Based on the Recursive Language Models (RLM) paradigm.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">💻 Code</span>
      <a href="repl/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/openenv/repl">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Git</span>
      <p class="environment-card__description">
        Teaches agents to navigate repositories, inspect diffs, and land changes via Git-native operations.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">💻 Code</span>
      <a href="git/">📄 Docs</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Terminal-Bench 2</span>
      <p class="environment-card__description">
        OpenEnv wrapper for Terminal-Bench 2 tasks.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">💻 Code</span>
      <a href="tbench2/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/openenv/tbench2">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">BrowserGym</span>
      <p class="environment-card__description">
        Browser automation environment for web agents with DOM interaction, navigation, and multi-step task completion.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🌐 Web</span>
      <a href="browsergym/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/burtenshaw/browsergym-v2">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">OpenApp</span>
      <p class="environment-card__description">
        A web application simulation environment for OpenEnv that wraps the OpenApps framework and BrowserGym.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🌐 Web</span>
      <a href="openapp/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/Crashbandicoote2/Openapp_env">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Web Search</span>
      <p class="environment-card__description">
        Web search environment for RL research with search APIs, result parsing, and customizable rewards.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🌐 Web</span>
      <a href="websearch/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/lawhy/web_search">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Unity</span>
      <p class="environment-card__description">
        A wrapper for Unity environments to bring different graphical simulation environments from Unity that support ML-Agents.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🤖 Simulation</span>
      <a href="unity/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/Crashbandicoote2/unity_env">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">SUMO-RL</span>
      <p class="environment-card__description">
        Traffic control scenarios with SUMO simulators for agents that reason about continuous control and scheduling.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🤖 Simulation</span>
      <a href="sumo/">📄 Docs</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">FinRL</span>
      <p class="environment-card__description">
        Financial market simulations with portfolio APIs, perfect for RLHF strategies and algorithmic trading experiments.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🤖 Simulation</span>
      <a href="finrl/">📄 Docs</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Agent World Model</span>
      <p class="environment-card__description">
        AgentWorldModel-1K: 1,000 synthetic MCP tool-use environments with 10,000 tasks for large-scale agentic RL training. Each scenario spawns an isolated MCP environment with its own SQLite-backed state and task-specific verifier.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">🤖 Simulation</span>
      <a href="agent_world_model/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/ChilleD/agent_world_model_env">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">Chat</span>
      <p class="environment-card__description">
        Message-driven loop tailored for conversational agents that need structured turns, safety rails, and message attribution.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">💬 Conversational</span>
      <a href="chat/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/openenv/chat_env">🤗 Hugging Face</a>
    </div>
  </div>

  <div class="environment-card">
    <div class="environment-card__body">
      <span class="environment-card__tag">DIPG Safety</span>
      <p class="environment-card__description">
        Safety-critical diagnostics from the DIPG benchmark, highlighting guardrails, adversarial prompts, and risk scoring.
      </p>
    </div>
    <div class="environment-card__links">
      <span class="environment-card__category">💬 Conversational</span>
      <a href="dipg/">📄 Docs</a>
      <a href="https://huggingface.co/spaces/surfiniaburger/dipg-gym">🤗 Hugging Face</a>
    </div>
  </div>

</div>

> Want to publish your own environment? Head over to the [Build Your Own Environment](../guides/first-environment.md) guide for a step-by-step walkthrough.
