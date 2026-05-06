# Reward Design

:::{note}
This page is still being filled in — it will be rewritten around the [Rubric system (RFC 004)](https://github.com/meta-pytorch/OpenEnv/blob/main/rfcs/004-rubrics.md) that lives in `openenv.core.rubrics`. The content below covers the basics of ad-hoc reward shaping until then.
:::

Learn how to design effective reward functions for your OpenEnv environments.

## Overview

Reward functions are critical for RL training. They signal to your agent what behaviors are desirable.

## Reward Design Principles

### 1. Start Simple

Begin with sparse rewards (success/failure) before adding shaped rewards:

```python
def compute_reward(observation, action, terminated):
    if terminated and observation.success:
        return 1.0
    elif terminated:
        return -1.0
    return 0.0
```

### 2. Shape Carefully

Add intermediate rewards to help learning, but avoid reward hacking:

```python
def compute_reward(observation, action, terminated):
    reward = 0.0

    # Progress reward
    reward += 0.1 * observation.progress_delta

    # Success bonus
    if terminated and observation.success:
        reward += 10.0

    return reward
```

### 3. Consider Density

Dense rewards (every step) speed learning but can cause local optima. Sparse rewards are cleaner but slower.

## Environment Examples

### Chess (Sparse)

```python
# Win: +1, Loss: -1, Draw: 0
reward = result.observation.game_result
```

### Coding (Dense)

```python
reward = 0.0
if observation.tests_passed > prev_tests_passed:
    reward += 0.5 * (observation.tests_passed - prev_tests_passed)
if observation.all_tests_passed:
    reward += 5.0
```

### TextArena (Mixed)

```python
# Per-turn progress + final outcome
reward = observation.score_delta + (10.0 if observation.won else 0.0)
```

## Common Pitfalls

1. **Reward hacking** - Agent finds unintended shortcuts
2. **Sparse rewards** - Agent never finds positive signal
3. **Conflicting signals** - Mixed incentives confuse learning

## Next Steps

- [RL Framework Integration](rl-integration.md) - Use rewards in training
- [Environment Anatomy](environment-anatomy.md) - Where to implement rewards
