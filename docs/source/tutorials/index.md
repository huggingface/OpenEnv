# Learn

Choose a learning path from the sidebar:

- **Basics:** start with [Hello World](openenv-tutorial) and build [Your First Environment](../guides/first-environment).
- **Training:** [the ways to train and the supported frameworks](../guides/training), then train a [reasoning model](end-to-end-walkthrough), use [TRL](wordle-grpo) or [Unsloth](rl-training-2048), or collect rollouts for [SFT](sft-warmup).
- **Harnesses:** [pick a path](harnesses): train with the trainer's own loop on [BrowserGym](browsergym-harness) (white-box), train real agents through [Harbor](harbor-harness) (black-box), or [evaluate Claude Code inside an environment](claude-code-harness). The [OpenCode](opencode-agent-grpo) and [Pi](pi-agent-grpo) tutorials are deprecated.
- **Evals:** follow [Evaluating with Environments](evaluation-inspect).

For [MCP environments](mcp-environment) and [rubrics](rubrics), see Concepts in the sidebar.

## New to OpenEnv? Start Here

The Getting Started Series walks you from zero to deploying your own environment in five short parts. No GPU required.

| Part | What it covers | Notebook |
|------|---------------|----------|
| 1 — Introduction & Quick Start | What OpenEnv is, why it exists, and your first environment in under 10 minutes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/openenv_introduction_quickstart.ipynb) |
| 2 — Using Environments | Connect to environments, create policies, run evaluations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/openenv_using_environments.ipynb) |
| 3 — Building Environments | Create a custom environment from scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/openenv_building_environments.ipynb) |
| [4 — Packaging & Deploying](../getting_started/environment-builder) | Package with Docker and deploy to Hugging Face | — |
| [5 — Contributing to Hugging Face](../getting_started/contributing-envs) | Publish, fork, and share environments on the Hub | — |

## Topic Tutorials

Already familiar with the basics? These tutorials cover specific workflows in depth.

| Tutorial | What it covers | GPU | Notebook |
|----------|---------------|-----|----------|
| [OpenEnv Tutorial](openenv-tutorial) | Full introduction to OpenEnv: install, connect to a hosted environment, step through an episode, define a reward function, and run a basic training loop. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb) |
| [End-to-end walkthrough](end-to-end-walkthrough) | The full pipeline: connect to `reasoning_gym`, wire it into TRL via `environment_factory`, fine-tune with GRPO, and push the checkpoint to the Hub. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/end_to_end_walkthrough.ipynb) |
| [Building and using MCP environments](mcp-environment) | Consume and build MCP-backed environments: list and call tools through `step()`, register Python functions as tools with FastMCP. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/mcp_environment.ipynb) |
| [Rubrics](rubrics) | Compose reward functions from reusable pieces using `Gate`, `WeightedSum`, `LLMJudge`, and `TrajectoryRubric`. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/rubrics.ipynb) |
| [Wordle GRPO](wordle-grpo) | Train an agent to play Wordle using GRPO via TRL's `environment_factory`. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/grpo_wordle/grpo_wordle.ipynb) |
| [RL Training with 2048](rl-training-2048) | Train a language model to play 2048 using GRPO. Covers game-state representation and reward shaping. | Yes | — |
| [Evaluating with Environments](evaluation-inspect) | Wrap an OpenEnv environment in an Inspect AI `Task`, run it via `InspectAIHarness`, and get a structured `EvalResult`. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/evaluation_inspect.ipynb) |
| [White-box: Train a Web Agent on BrowserGym](browsergym-harness) | Train a vision-language model on BrowserGym web tasks with TRL's `GRPOTrainer` and `environment_factory`, so TRL runs the multi-turn tool loop. | Yes | — |
| [Black-box: Train Real Agents with Harbor](harbor-harness) | Run real agents (OpenCode, Claude Code, Codex, …) on Harbor tasks with their token ids and logprobs captured, and train the model behind them with TRL's `AsyncGRPOTrainer`. | Yes | — |
| [Evaluate Claude Code in an Environment](claude-code-harness) | Run Claude Code inside an OpenEnv environment with `HarnessEnvironment` (RFC 005), inject the environment's tools over MCP, and evaluate it on τ²-bench against a simulated customer. Also serves it in production mode. | No | — |
| [Training a Real Coding Agent (deprecated)](opencode-agent-grpo) | Deprecated, removed in OpenEnv 0.9.0: use [Harbor](https://huggingface.co/docs/openenv/environments/harbor) with `harness="opencode"`. Train the actual OpenCode agent (black-box, loop-owning) with TRL's `AsyncGRPOTrainer`: a transparent proxy captures each turn's token ids and logprobs while the agent runs its own tool loop. | Yes | — |
| [Collecting rollouts for supervised training](sft-warmup) | Run a teacher model to collect reward-labeled rollouts, filter them, and fine-tune a student with TRL's `SFTTrainer` as a warm-start for GRPO. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/sft_warmup.ipynb) |
| [Training a Real Coding Agent (Pi, deprecated)](pi-agent-grpo) | Deprecated, removed in OpenEnv 0.9.0: use [Harbor](https://huggingface.co/docs/openenv/environments/harbor) with `harness="pi"`. Train the actual Pi agent (black-box, loop-owning) with TRL's AsyncGRPOTrainer: a transparent proxy captures each turn's token ids and logprobs while the agent runs its own tool loop. | Yes | — |
