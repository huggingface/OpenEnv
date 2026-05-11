# Tutorials

## New to OpenEnv? Start Here

The Getting Started Series walks you from zero to deploying your own environment in five short parts. No GPU required.

| Part | What it covers |
|------|---------------|
| [1 — Introduction & Quick Start](../auto_getting_started/plot_01_introduction_quickstart) | What OpenEnv is, why it exists, and your first environment in under 10 minutes |
| [2 — Using Environments](../auto_getting_started/plot_02_using_environments) | Connect to environments, create policies, run evaluations |
| [3 — Building Environments](../auto_getting_started/plot_03_building_environments) | Create a custom environment from scratch |
| [4 — Packaging & Deploying](../auto_getting_started/environment-builder) | Package with Docker and deploy to Hugging Face |
| [5 — Contributing to Hugging Face](../auto_getting_started/contributing-envs) | Publish, fork, and share environments on the Hub |

## Topic Tutorials

Already familiar with the basics? These tutorials cover specific workflows in depth.

| Tutorial | What it covers | GPU | Notebook |
|----------|---------------|-----|----------|
| [OpenEnv Tutorial](openenv-tutorial.md) | Full introduction to OpenEnv: install, connect to a hosted environment, step through an episode, define a reward function, and run a basic training loop. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb) |
| [End-to-end walkthrough](end-to-end-walkthrough.md) | The full pipeline: connect to `reasoning_gym`, wire it into TRL via `environment_factory`, fine-tune with GRPO, and push the checkpoint to the Hub. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/end_to_end_walkthrough.ipynb) |
| [Building and using MCP environments](mcp-environment.md) | Consume and build MCP-backed environments: list and call tools through `step()`, register Python functions as tools with FastMCP. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/mcp_environment.ipynb) |
| [Rubrics](rubrics.md) | Compose reward functions from reusable pieces using `Gate`, `WeightedSum`, `LLMJudge`, and `TrajectoryRubric`. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/rubrics.ipynb) |
| [Wordle GRPO](wordle-grpo.md) | Train an agent to play Wordle using GRPO via TRL's `environment_factory`. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb) |
| [RL Training with 2048](rl-training-2048.md) | Train a language model to play 2048 using GRPO. Covers game-state representation and reward shaping. | Yes | — |
| [Evaluating agents with Inspect AI](evaluation-inspect.md) | Wrap an OpenEnv environment in an Inspect AI `Task`, run it via `InspectAIHarness`, and get a structured `EvalResult`. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/evaluation_inspect.ipynb) |

```{toctree}
:maxdepth: 1
:hidden:

../auto_getting_started/index
openenv-tutorial
end-to-end-walkthrough
mcp-environment
rubrics
wordle-grpo
browsergym-harness
rl-training-2048
evaluation-inspect
```
