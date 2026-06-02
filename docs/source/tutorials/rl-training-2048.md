# RL Training with OpenEnv: 2048 Game

This tutorial covers training a language model to play the 2048 game using
reinforcement learning with GRPO (Group Relative Policy Optimization).

```{note}
**Time**: ~45 minutes | **Difficulty**: Advanced | **GPU Required**: Yes (T4 or better)
```

## What You'll Learn

- **Model Setup**: Load and configure LLMs with Unsloth for efficient RL
- **Environment Connection**: Connect to the 2048 OpenEnv environment
- **Reward Design**: Create effective reward functions
- **GRPO Training**: Train models with reinforcement learning
- **Deployment**: Save and deploy trained models

## Prerequisites

Before starting this tutorial, you should have completed the
[Getting Started](/auto_getting_started/index) series to understand:

- How OpenEnv environments work
- The reset/step/state API pattern
- How to connect to environments

You'll also need:

- A GPU (free T4 on Google Colab works)
- Basic understanding of PyTorch
- ~30 minutes for training

## Part 1: Environment Setup

### Installation

```bash
# Install required packages
!pip install -q unsloth openenv trl

# For Google Colab, also run:
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Imports

```python
import torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import random

# Check GPU availability
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Part 2: Model Configuration

We use Unsloth for memory-efficient training with LoRA adapters.

### Configuration Classes

```python
@dataclass
class ModelConfig:
    """Configuration for loading LLM models."""
    model_name: str = "unsloth/Qwen2.5-1.5B"
    max_seq_length: int = 768
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # Auto-detect


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.0

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
```

### Loading the Model

```python
from unsloth import FastLanguageModel

# Create configurations
model_config = ModelConfig()
lora_config = LoRAConfig()

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_config.model_name,
    max_seq_length=model_config.max_seq_length,
    load_in_4bit=model_config.load_in_4bit,
    dtype=model_config.dtype,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_config.r,
    target_modules=lora_config.target_modules,
    lora_alpha=lora_config.lora_alpha,
    lora_dropout=lora_config.lora_dropout,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Check parameter counts
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
```

## Part 3: The 2048 Environment

### Game Overview

2048 is a sliding puzzle game where you combine tiles to reach 2048.

**Actions:**
- `0` = UP
- `1` = RIGHT
- `2` = DOWN
- `3` = LEFT

**Goal:** Create a tile with value 2048 (or higher!)

### Connecting to the Environment

```python
from envs.openspiel_env import OpenSpielEnv, OpenSpielAction

# Connect to 2048 environment
# Option 1: From Hub
env = OpenSpielEnv.from_hub("openenv/openspiel-env")

# Option 2: From running server
# env = OpenSpielEnv(base_url="http://localhost:8000")

# Test connection
with env:
    result = env.reset()
    print(f"Game started!")
    print(f"Legal actions: {result.observation.legal_actions}")

    # Take a test action
    action = OpenSpielAction(action_id=0, game_name="2048")
    result = env.step(action)
    print(f"After UP: reward={result.reward}, done={result.done}")
```

### Board Utilities

```python
import numpy as np
from typing import List

def info_state_to_board(info_state: List[int], size: int = 4) -> List[List[int]]:
    """Convert flat info_state to 2D board."""
    return np.array(info_state, dtype=int).reshape(size, size).tolist()

def render_board(board: List[List[int]]) -> str:
    """Render board as ASCII string."""
    lines = ["+------" * len(board[0]) + "+"]
    for row in board:
        cells = [f"{v:5d}" if v > 0 else "    ." for v in row]
        lines.append("|" + " |".join(cells) + " |")
        lines.append("+------" * len(row) + "+")
    return "\n".join(lines)

def get_max_tile(board: List[List[int]]) -> int:
    """Get highest tile value."""
    return max(cell for row in board for cell in row)
```

## Part 4: Reward Function Design

The reward function is crucial for RL. We consider:

1. **Success**: Did we reach 2048?
2. **Progress**: What's the highest tile achieved?
3. **Code Quality**: Did the generated code execute correctly?

### Reward Implementation

```python
import math

def calculate_reward(
    max_tile: int,
    success: bool,
    code_error: bool = False
) -> float:
    """
    Calculate reward for a 2048 game outcome.

    Args:
        max_tile: Highest tile achieved (2, 4, 8, ..., 2048)
        success: Whether we reached 2048
        code_error: Whether generated code had errors

    Returns:
        Float reward value
    """
    if code_error:
        return -0.5  # Penalty for invalid code

    if success:
        return 1.0  # Full reward for winning

    # Progress reward: log scale from 0 to 0.9
    if max_tile > 0:
        progress = math.log2(max_tile) / math.log2(2048)
        return min(0.9, progress)

    return 0.0

# Test reward function
test_cases = [
    (2048, True, False, "Won!"),
    (1024, False, False, "Got to 1024"),
    (512, False, False, "Got to 512"),
    (64, False, False, "Early game"),
]

for max_tile, success, error, desc in test_cases:
    reward = calculate_reward(max_tile, success, error)
    print(f"{desc:20s} -> Reward: {reward:+.3f}")
```

## Part 5: Strategy Generation

We'll train the model to generate Python strategy functions.

### Prompt Template

```python
SYSTEM_PROMPT = """You are an expert at playing 2048. Generate a Python function
that takes a board state and returns the best action (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT).

The board is a 4x4 list of integers. Empty cells are 0.
Your function should analyze the board and return an optimal move.
"""

def create_prompt(board: List[List[int]]) -> str:
    """Create prompt for strategy generation."""
    board_str = "\n".join(str(row) for row in board)
    return f"""{SYSTEM_PROMPT}

Current board:
{board_str}

Generate a strategy function:
```python
def strategy(board):
    # Your code here
    return action  # 0, 1, 2, or 3
```"""
```

### Executing Generated Strategies

```python
import ast
from typing import Callable

def extract_and_execute_strategy(
    generated_code: str,
    board: List[List[int]],
    timeout: float = 5.0
) -> tuple[int, bool]:
    """
    Extract and execute a generated strategy function.

    Returns:
        (action, success): The action to take and whether execution succeeded
    """
    try:
        # Extract code block
        if "```python" in generated_code:
            code = generated_code.split("```python")[1].split("```")[0]
        else:
            code = generated_code

        # Parse and validate AST
        tree = ast.parse(code)

        # Execute in sandbox
        namespace = {"board": board}
        exec(compile(tree, "<strategy>", "exec"), namespace)

        # Call the strategy function
        if "strategy" in namespace:
            action = namespace["strategy"](board)
            if action in [0, 1, 2, 3]:
                return action, True

        return 0, False  # Default action on failure

    except Exception as e:
        print(f"Strategy execution error: {e}")
        return 0, False
```

## Part 6: GRPO Training

GRPO (Group Relative Policy Optimization) is optimized for language models.

### Training Configuration

```python
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    # Learning rate
    learning_rate=2e-6,

    # Batch sizes
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    # Training duration
    max_steps=200,

    # Memory optimization
    bf16=True,
    gradient_checkpointing=True,

    # Logging
    logging_steps=1,
    output_dir="./2048_grpo_output",
    report_to="none",
)
```

### Training Loop

```python
def train_2048_agent(
    model,
    tokenizer,
    env,
    config: GRPOConfig,
    num_episodes: int = 100,
):
    """
    Train the model to play 2048 using GRPO.
    """
    # Prepare model for training
    FastLanguageModel.for_training(model)

    training_data = []

    for episode in range(num_episodes):
        # Reset environment
        result = env.reset()
        board = info_state_to_board(result.observation.info_state)

        episode_reward = 0
        steps = 0

        while not result.done and steps < 1000:
            # Generate strategy
            prompt = create_prompt(board)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Execute strategy
            action, success = extract_and_execute_strategy(generated, board)

            # Take action in environment
            env_action = OpenSpielAction(action_id=action, game_name="2048")
            result = env.step(env_action)

            # Update board
            board = info_state_to_board(result.observation.info_state)
            episode_reward += result.reward if result.reward else 0
            steps += 1

        # Calculate final reward
        max_tile = get_max_tile(board)
        final_reward = calculate_reward(max_tile, max_tile >= 2048)

        # Store for training
        training_data.append({
            "prompt": prompt,
            "response": generated,
            "reward": final_reward,
        })

        if episode % 10 == 0:
            print(f"Episode {episode}: Max tile={max_tile}, Reward={final_reward:.3f}")

    return training_data
```

## Part 7: Deployment

After training, save and deploy your model.

### Saving the Model

```python
# Save LoRA adapters only
model.save_pretrained("./2048_strategy_model")
tokenizer.save_pretrained("./2048_strategy_model")

# Save merged model for inference
model.save_pretrained_merged(
    "./2048_strategy_model_merged",
    tokenizer,
    save_method="merged_16bit",
)
```

### Push to Hugging Face Hub

```python
# Push to Hub
model.push_to_hub(
    "your-username/2048-strategy-model",
    tokenizer,
    save_method="merged_16bit",
    private=False,
)

print("Model deployed to: huggingface.co/your-username/2048-strategy-model")
```

### Using the Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load trained model
model = AutoModelForCausalLM.from_pretrained("your-username/2048-strategy-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/2048-strategy-model")

# Generate strategy
def get_action(board: List[List[int]]) -> int:
    prompt = create_prompt(board)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    action, _ = extract_and_execute_strategy(generated, board)
    return action

# Play a game
with OpenSpielEnv.from_hub("openenv/openspiel-env") as env:
    result = env.reset()
    board = info_state_to_board(result.observation.info_state)

    while not result.done:
        action = get_action(board)
        result = env.step(OpenSpielAction(action_id=action, game_name="2048"))
        board = info_state_to_board(result.observation.info_state)

    print(f"Final max tile: {get_max_tile(board)}")
```

## Preventing Reward Hacking

Be aware of potential reward hacking strategies:

1. **Code that modifies rewards** - Run in sandboxed environment
2. **Infinite loops** - Set execution timeouts
3. **Memory exhaustion** - Limit resource usage

```python
import resource
import signal

def safe_execute(code: str, board: List[List[int]], timeout: float = 5.0) -> int:
    """Execute strategy with safety limits."""

    def handler(signum, frame):
        raise TimeoutError("Strategy timed out")

    # Set timeout
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(timeout))

    try:
        # Set memory limit (100MB)
        resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, -1))

        # Execute in restricted namespace
        namespace = {"board": board, "__builtins__": {"len": len, "max": max, "min": min}}
        exec(code, namespace)

        return namespace.get("strategy", lambda b: 0)(board)
    finally:
        signal.alarm(0)
```

## Summary

In this tutorial, you learned:

1. **Model Setup**: Loading LLMs with Unsloth and LoRA
2. **Environment Connection**: Using OpenEnv's 2048 environment
3. **Reward Design**: Creating balanced reward functions
4. **GRPO Training**: Training with reinforcement learning
5. **Deployment**: Saving and sharing trained models

## Next Steps

- Try different model architectures
- Experiment with reward function designs
- Train on other OpenEnv environments
- Share your trained models on Hugging Face Hub!

## Related Resources

- [OpenEnv Getting Started](../auto_getting_started/index)
- [Building Custom Environments](../auto_getting_started/plot_03_building_environments)
- [GRPO Documentation](https://huggingface.co/docs/trl/grpo_trainer)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
