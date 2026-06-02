# Tutorials

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
| [OpenEnv Tutorial](openenv-tutorial.md) | Full introduction to OpenEnv: install, connect to a hosted environment, step through an episode, define a reward function, and run a basic training loop. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb) |
| [End-to-end walkthrough](end-to-end-walkthrough.md) | The full pipeline: connect to `reasoning_gym`, wire it into TRL via `environment_factory`, fine-tune with GRPO, and push the checkpoint to the Hub. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/end_to_end_walkthrough.ipynb) |
| [Building and using MCP environments](mcp-environment.md) | Consume and build MCP-backed environments: list and call tools through `step()`, register Python functions as tools with FastMCP. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/mcp_environment.ipynb) |
| [Rubrics](rubrics.md) | Compose reward functions from reusable pieces using `Gate`, `WeightedSum`, `LLMJudge`, and `TrajectoryRubric`. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/rubrics.ipynb) |
| [Wordle GRPO](wordle-grpo.md) | Train an agent to play Wordle using GRPO via TRL's `environment_factory`. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb) |
| [RL Training with 2048](rl-training-2048.md) | Train a language model to play 2048 using GRPO. Covers game-state representation and reward shaping. | Yes | — |
| [Evaluating agents with Inspect AI](evaluation-inspect.md) | Wrap an OpenEnv environment in an Inspect AI `Task`, run it via `InspectAIHarness`, and get a structured `EvalResult`. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/evaluation_inspect.ipynb) |
| [BrowserGym Harness Rollouts](browsergym-harness.md) | Drive BrowserGym through the OpenEnv harness runtime when a trainer needs token sampling, logprobs, and reward assignment inside the training loop. | Yes | — |
| [Collecting rollouts for supervised training](sft-warmup.md) | Run a teacher model to collect reward-labeled rollouts, filter them, and fine-tune a student with TRL's `SFTTrainer` as a warm-start for GRPO. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/sft_warmup.ipynb) |

