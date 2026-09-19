# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward component schema and helpers.

This module provides a standard representation for decomposed reward signals
while preserving a scalar optimization target.
"""

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class RewardComponentType(str, Enum):
    """Common reward component styles for training diagnostics."""

    BINARY = "binary"
    SPARSE = "sparse"
    DENSE = "dense"
    SHAPING = "shaping"
    PENALTY = "penalty"


class RewardComponent(BaseModel):
    """Structured reward component emitted by a rubric/environment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Stable component identifier")
    type: RewardComponentType = Field(..., description="Reward component style")
    value: float = Field(..., description="Raw component value before weighting")
    weight: float = Field(default=1.0, description="Aggregation weight")
    weighted_value: float | None = Field(
        default=None,
        description="Optional explicit weighted value (defaults to value * weight)",
    )
    terminal_only: bool = Field(
        default=False,
        description="Whether this component is meaningful only at terminal steps",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional component-specific diagnostics",
    )

    def effective_weighted_value(self) -> float:
        """Return weighted component value, respecting explicit overrides."""
        if self.weighted_value is not None:
            return self.weighted_value
        return self.value * self.weight


def aggregate_weighted_sum(components: List[RewardComponent]) -> float:
    """Aggregate reward components with weighted-sum semantics."""
    return float(sum(component.effective_weighted_value() for component in components))


def serialize_reward_components(components: List[RewardComponent]) -> List[Dict[str, Any]]:
    """Serialize reward components for observation metadata."""
    return [component.model_dump() for component in components]
