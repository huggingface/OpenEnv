# Type definitions for EnvTorch
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

# Generic type for observations
ObsT = TypeVar("ObsT")
StateT = TypeVar("StateT")


@dataclass
class StepResult(Generic[ObsT]):
    """
    Represents the result of one environment step.

    Attributes:
        observation:
            The environment's observation after the action.
        reward (`float`, *optional*):
            Scalar reward for this step.
        done (`bool`, *optional*, defaults to `False`):
            Whether the episode is finished.
    """

    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
