# Tutorials

Welcome to the OpenEnv tutorials! These guides will help you get started with using and building environments with OpenEnv.

## Getting Started

If you're new to OpenEnv, we recommend starting with the [Getting Started](/auto_getting_started/index) series to understand the core concepts and basic usage patterns.

## Available Tutorials

| Tutorial | What it covers | GPU | Notebook |
|----------|---------------|-----|----------|
| [OpenEnv Tutorial](openenv-tutorial.md) | Full introduction to OpenEnv: install, connect to a hosted environment, step through an episode, define a reward function, and run a basic training loop. Start here if you're new. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb) |
| [End-to-end walkthrough](end-to-end-walkthrough.md) | The full pipeline in one page: connect to `reasoning_gym`, wire it into TRL via `environment_factory`, fine-tune with GRPO on `chain_sum`, read the reward delta from training logs, and push the checkpoint to the Hub. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/end_to_end_walkthrough.ipynb) |
| [Building and using MCP environments](mcp-environment.md) | Consume and build MCP-backed environments: list and call tools through `step()`, register Python functions as tools with FastMCP, and understand the dual API boundary. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/mcp_environment.ipynb) |
| [Wordle GRPO](wordle-grpo.md) | Train an agent to play Wordle using GRPO via TRL's `environment_factory`. Shows the multi-turn tool-calling loop: the model guesses a word each turn and receives letter-position feedback until it wins or the episode ends. | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb) |
| [RL Training with 2048](rl-training-2048.md) | Train a language model to play the 2048 tile-sliding game with GRPO. Covers game-state representation and reward shaping for a puzzle environment. | Yes | — |
| [Evaluating agents with Inspect AI](evaluation-inspect.md) | Wrap an OpenEnv environment in an Inspect AI `Task` (dataset + solver + scorer), run it via `InspectAIHarness`, and get a structured `EvalResult` with accuracy scores. No training required — useful as a standalone eval pass on any checkpoint. | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/evaluation_inspect.ipynb) |

```{toctree}
:maxdepth: 1
:hidden:
openenv-tutorial
end-to-end-walkthrough
mcp-environment
wordle-grpo
rl-training-2048
evaluation-inspect
```
