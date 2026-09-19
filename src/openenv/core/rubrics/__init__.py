# SPDX-License-Identifier: BSD-3-Clause

"""Rubrics for reward computation.

See RFC 004 for full design: rfcs/004-rubrics.md
"""

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.components import (
    RewardComponent,
    RewardComponentType,
    aggregate_weighted_sum,
    serialize_reward_components,
)
from openenv.core.rubrics.containers import (
    Gate,
    RubricDict,
    RubricList,
    Sequential,
    WeightedSum,
)
from openenv.core.rubrics.llm_judge import LLMJudge
from openenv.core.rubrics.trajectory import (
    ExponentialDiscountingTrajectoryRubric,
    TrajectoryRubric,
)

__all__ = [
    # Base
    "Rubric",
    # Components
    "RewardComponent",
    "RewardComponentType",
    "aggregate_weighted_sum",
    "serialize_reward_components",
    # Containers
    "Sequential",
    "Gate",
    "WeightedSum",
    "RubricList",
    "RubricDict",
    # Trajectory
    "TrajectoryRubric",
    "ExponentialDiscountingTrajectoryRubric",
    # LLM Judge
    "LLMJudge",
]
