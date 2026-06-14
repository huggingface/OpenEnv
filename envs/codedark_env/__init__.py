"""
CodeDark - OpenEnv-Compatible Data Analytics Environment

Multi-turn RL environment for training AI agents on data analytics tasks.

Example usage:
    from codedark import CodeDarkEnv

    env = CodeDarkEnv("http://localhost:8000")
    obs = env.reset()
    print(f"Task: {obs['question']}")

    obs = env.run_python("result = df['y'].mean() * 100")
    obs = env.submit_answer(11.26)
    print(f"Reward: {obs['reward']}")
"""

from .client import CodeDarkEnv
from .models import (
    CodeDarkAction,
    CodeDarkObservation,
    CodeDarkState,
    ResetRequest,
    StepRequest,
    HealthResponse,
)

__all__ = [
    "CodeDarkEnv",
    "CodeDarkAction",
    "CodeDarkObservation",
    "CodeDarkState",
    "ResetRequest",
    "StepRequest",
    "HealthResponse",
]

__version__ = "0.1.0"
