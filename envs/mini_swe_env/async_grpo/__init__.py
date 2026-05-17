"""Async GRPO training for Mini SWE on HF Spaces.

This package provides:

- :class:`SWEToolEnv` — TRL ``environment_factory``-compatible environment
  with ``bash()`` and ``answer()`` tools backed by an HF/Docker sandbox.
- :func:`swe_reward` — Reward function that parses grading results from
  completion messages.

These plug directly into TRL's ``AsyncGRPOTrainer`` (or ``GRPOTrainer``).
No custom rollout worker, interception server, or control plane needed —
TRL handles tokenization, generation, logprobs, token IDs, weight sync,
and sample assembly.
"""

from .swe_tool_env import (
    SWEToolEnv,
    swe_reward,
)

__all__ = [
    "SWEToolEnv",
    "swe_reward",
]
