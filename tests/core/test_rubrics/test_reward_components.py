# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for reward component schema and helpers."""

from openenv.core.rubrics.components import (
    RewardComponent,
    RewardComponentType,
    aggregate_weighted_sum,
    serialize_reward_components,
)


class TestRewardComponent:
    def test_weighted_value_defaults_to_value_times_weight(self):
        component = RewardComponent(
            name="progress",
            type=RewardComponentType.DENSE,
            value=0.6,
            weight=0.5,
        )
        assert component.effective_weighted_value() == 0.3

    def test_explicit_weighted_value_takes_precedence(self):
        component = RewardComponent(
            name="safety_penalty",
            type=RewardComponentType.PENALTY,
            value=-1.0,
            weight=0.2,
            weighted_value=-0.5,
        )
        assert component.effective_weighted_value() == -0.5


class TestRewardComponentHelpers:
    def test_aggregate_weighted_sum(self):
        components = [
            RewardComponent(
                name="success",
                type=RewardComponentType.BINARY,
                value=1.0,
                weight=0.7,
            ),
            RewardComponent(
                name="format",
                type=RewardComponentType.SPARSE,
                value=1.0,
                weight=0.3,
            ),
        ]
        assert aggregate_weighted_sum(components) == 1.0

    def test_serialize_reward_components(self):
        components = [
            RewardComponent(
                name="step_quality",
                type=RewardComponentType.SHAPING,
                value=-0.05,
            )
        ]
        payload = serialize_reward_components(components)
        assert len(payload) == 1
        assert payload[0]["name"] == "step_quality"
        assert payload[0]["type"] == "shaping"
        assert payload[0]["value"] == -0.05
