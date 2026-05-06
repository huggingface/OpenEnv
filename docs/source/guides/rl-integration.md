# RL Framework Integration

:::{note}
This page is still being filled in. TRL integration is covered below; torchforge and SkyRL integrations are planned.
:::

Use OpenEnv with popular RL frameworks like TRL, torchforge, and SkyRL.

## Overview

OpenEnv environments are designed to integrate seamlessly with RL training frameworks. The standard `step()`, `reset()`, `state()` API makes it easy to use environments in training loops.

## TRL Integration

[TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl) is the recommended framework for training language models with RL.

```python
from trl import GRPOTrainer
from openenv import AutoEnv, AutoAction

env = AutoEnv.from_env("textarena")
TextAction = AutoAction.from_env("textarena")

# Use with TRL's GRPO trainer
trainer = GRPOTrainer(
    model=model,
    reward_model=reward_model,
    # ... TRL config
)
```

See the [Wordle with GRPO](../tutorials/wordle-grpo.md) tutorial for a complete example.

## Generic Training Loop

For custom training setups:

```python
from openenv import AutoEnv, AutoAction

env = AutoEnv.from_env("my-env")
Action = AutoAction.from_env("my-env")

with env.sync() as client:
    for episode in range(num_episodes):
        result = client.reset()

        while not result.terminated:
            # Get action from your policy
            action = policy(result.observation)

            # Take step
            result = client.step(action)

            # Update policy with reward
            policy.update(result.reward)
```

## Next Steps

- [Reward Design](rewards.md) - Design effective reward functions
- [Wordle with GRPO](../tutorials/wordle-grpo.md) - Complete TRL example
