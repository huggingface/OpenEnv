# Environments

```{note}
The environments listed here may not reflect the latest additions. For the official OpenEnv collection, see the [OpenEnv organization on Hugging Face](https://huggingface.co/openenv). You may also find additional community environments tagged `agent-environment` on [Hugging Face Spaces](https://huggingface.co/spaces?category=agent-environment). The environments highlighted below are a curated selection.
```

The OpenEnv community has built a catalog of ready-to-run environments that cover deterministic smoke tests, full developer workflows, and multi-step reasoning challenges. Explore the surface area below and jump directly into the guides for each environment.

`````{grid} 1 2 3 3
:gutter: 3

````{grid-item-card} Echo
:class-card: sd-border-1

Minimal observation/action loop for verifying client integrations, CI pipelines, and onboarding flows in seconds.

+++
```{button-link} environments/echo.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/openenv/echo_env
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} Coding
:class-card: sd-border-1

Secure sandbox with filesystem access and evaluation hooks for executing generated code and building autonomous dev workflows.

+++
```{button-link} environments/coding.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/openenv/coding_env
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} Chat
:class-card: sd-border-1

Message-driven loop tailored for conversational agents that need structured turns, safety rails, and message attribution.

+++
```{button-link} environments/chat.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/openenv/chat_env
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} Atari
:class-card: sd-border-1

Classic Arcade Learning Environment tasks packaged for fast benchmarking of reinforcement-learning style agents.

+++
```{button-link} environments/atari.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/openenv/atari_env
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} OpenSpiel
:class-card: sd-border-1

Multi-agent, game-theory workloads powered by DeepMind's OpenSpiel suite, ideal for search and self-play experiments.

+++
```{button-link} environments/openspiel.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/openenv/openspiel_env
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} SUMO-RL
:class-card: sd-border-1

Traffic control scenarios with SUMO simulators for agents that reason about continuous control and scheduling.

+++
```{button-link} environments/sumo.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} FinRL
:class-card: sd-border-1

Financial market simulations with portfolio APIs, perfect for RLHF strategies and algorithmic trading experiments.

+++
```{button-link} environments/finrl.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} TextArena
:class-card: sd-border-1

Multi-task text arena for language-game competitions such as Wordle, reasoning puzzles, and program synthesis.

+++
```{button-link} environments/textarena.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/burtenshaw/textarena_env
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} Git
:class-card: sd-border-1

Teaches agents to navigate repositories, inspect diffs, and land changes via Git-native operations.

+++
```{button-link} environments/git.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} DIPG Safety
:class-card: sd-border-1

Safety-critical diagnostics from the DIPG benchmark, highlighting guardrails, adversarial prompts, and risk scoring.

+++
```{button-link} environments/dipg.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/surfiniaburger/dipg-gym
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} Snake
:class-card: sd-border-1

Classic snake game environment for RL research with configurable grids, partial observability, and customizable rewards.

+++
```{button-link} environments/snake.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/Crashbandicoote2/snake_env
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} Web Search
:class-card: sd-border-1

Web search environment for RL research with configurable grids, partial observability, and customizable rewards.

+++
```{button-link} environments/websearch.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/lawhy/web_search
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} BrowserGym
:class-card: sd-border-1

Browser automation environment for web agents with DOM interaction, navigation, and multi-step task completion.

+++
```{button-link} environments/browsergym.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/burtenshaw/browsergym-v2
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} KernRL
:class-card: sd-border-1

RL environment for GPU kernel optimization. Train LLM agents to write fast CUDA/Triton kernels that beat baseline implementations.

+++
```{button-link} environments/kernrl.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} Calendar
:class-card: sd-border-1

Calendar tool-use environment exposing a Calendar Gym through the OpenEnv reset/step/state interface for scheduling agents.

+++
```{button-link} environments/calendar.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} CARLA
:class-card: sd-border-1

Embodied evaluation environment for testing LLM decision-making in a full 3D driving simulator with irreversible consequences and ethical trolley scenarios.

+++
```{button-link} environments/carla.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/sergiopaniego/carla-env
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} Chess
:class-card: sd-border-1

Chess RL environment powered by the moonfish engine with configurable opponents, position evaluation, and full chess rules.

+++
```{button-link} environments/chess.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} Connect4
:class-card: sd-border-1

Classic Connect Four board game environment for training agents on turn-based strategy with a 6×7 grid.

+++
```{button-link} environments/connect4.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} DM Control
:class-card: sd-border-1

Generic OpenEnv wrapper for dm_control.suite, providing access to all MuJoCo-based continuous control tasks like cartpole, walker, and humanoid.

+++
```{button-link} environments/dm_control.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} FinQA
:class-card: sd-border-1

Financial question-answering environment that evaluates LLMs on complex financial questions using tool calls on SEC 10-K filing data.

+++
```{button-link} environments/finqa.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} Grid World
:class-card: sd-border-1

Simple 5×5 grid world RL testbed and step-by-step guide for building new OpenEnv environments from scratch.

+++
```{button-link} environments/grid_world.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/yuvrajpant56/grid_world_env
:color: warning
:outline:

🤗 Hugging Face
```
````

````{grid-item-card} Julia
:class-card: sd-border-1

Julia code execution environment with test result tracking and reward calculation for RL training on Julia programming tasks.

+++
```{button-link} environments/julia.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} Maze
:class-card: sd-border-1

Gridworld maze where agents navigate from start to exit while avoiding walls, with configurable 8×8 layouts.

+++
```{button-link} environments/maze.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} OpenApp
:class-card: sd-border-1

Web application simulation wrapping the OpenApps framework and BrowserGym for training UI agents on calendar, todo, messenger, and maps apps.

+++
```{button-link} environments/openapp.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} Reasoning Gym
:class-card: sd-border-1

Integrates the Reasoning Gym library to provide single-step reasoning tasks with configurable datasets and scoring.

+++
```{button-link} environments/reasoning_gym.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} REPL
:class-card: sd-border-1

Python REPL environment for code execution tasks based on the Recursive Language Models paradigm with sandboxed execution and context loading.

+++
```{button-link} environments/repl.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} TB2
:class-card: sd-border-1

OpenEnv wrapper for Terminal-Bench 2 tasks with local and Docker execution modes for terminal-based agent evaluation.

+++
```{button-link} environments/tbench2.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} Unity
:class-card: sd-border-1

OpenEnv wrapper for Unity ML-Agents environments, providing access to Unity's RL environments through HTTP/WebSocket interfaces.

+++
```{button-link} environments/unity.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} Wildfire
:class-card: sd-border-1

Autonomous wildfire-control simulation where agents contain spreading fires using water, firebreaks, and timing under dynamic conditions.

+++
```{button-link} environments/wildfire.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
````

````{grid-item-card} Agent World Model
:class-card: sd-border-1

AgentWorldModel-1K — 1,000 synthetic MCP tool-use environments with 10,000 tasks for large-scale agentic RL training.

+++
```{button-link} environments/agent_world_model.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/ChilleD/agent_world_model_env
:color: warning
:outline:

🤗 Hugging Face
```
````

`````

```{tip}
Want to publish your own environment? Head over to the [Build Your Own Environment](auto_getting_started/environment-builder.md) guide for a step-by-step walkthrough.
```

## Community Environments

`````{grid} 1 2 3 3
:gutter: 3

````{grid-item-card} RLVE Gym
:class-card: sd-border-1

A suite of 400 environments that procedurally generate reasoning problems for LM training with configurable difficulty.

+++
```{button-link} https://huggingface.co/spaces/ZhiyuanZeng/RLVE_Gym/blob/main/README.md
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/ZhiyuanZeng/RLVE_Gym
:color: warning
:outline:

🤗 Hugging Face
```
````

`````

```{toctree}
:hidden:
:maxdepth: 1

environments/echo
environments/coding
environments/chat
environments/atari
environments/openspiel
environments/sumo
environments/finrl
environments/textarena
environments/git
environments/dipg
environments/snake
environments/websearch
environments/browsergym
environments/repl
environments/calendar
environments/carla
environments/chess
environments/connect4
environments/dm_control
environments/finqa
environments/grid_world
environments/julia
environments/kernrl
environments/maze
environments/openapp
environments/reasoning_gym
environments/tbench2
environments/unity
environments/wildfire
environments/agent_world_model
```
