# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for observation serialization, specifically metadata preservation.

Ensures that Observation.metadata survives the serialize -> deserialize
round-trip through serialize_observation() and GenericEnvClient._parse_result().
"""

import pytest
from openenv.core.env_server.serialization import serialize_observation
from openenv.core.env_server.types import Observation, ResetResponse, StepResponse
from openenv.core.generic_client import GenericEnvClient


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class CustomObservation(Observation):
    """Observation subclass with domain-specific fields."""

    ally_tree: str = ""
    task_instruction: str = ""


# ---------------------------------------------------------------------
# serialize_observation tests
# ---------------------------------------------------------------------


class TestSerializeObservation:
    """Tests for serialize_observation()."""

    def test_metadata_preserved_in_serialized_output(self):
        """Metadata must appear in the serialized dict, not be silently dropped."""
        obs = Observation(
            done=False,
            reward=0.5,
            metadata={"total_nodes": 5, "task_id": 1},
        )
        result = serialize_observation(obs)

        assert "metadata" in result
        assert result["metadata"]["total_nodes"] == 5
        assert result["metadata"]["task_id"] == 1

    def test_empty_metadata_omitted(self):
        """When metadata is empty, it should not clutter the response."""
        obs = Observation(done=False, reward=0.0, metadata={})
        result = serialize_observation(obs)

        assert "metadata" not in result

    def test_falsy_metadata_preserved(self):
        """Falsy metadata values must not be silently dropped."""
        obs = Observation(metadata={"active": False, "count": 0, "flag": ""})
        result = serialize_observation(obs)

        assert "metadata" in result
        assert result["metadata"] == {"active": False, "count": 0, "flag": ""}

    def test_reward_and_done_promoted(self):
        """reward and done must be top-level siblings, not inside observation."""
        obs = Observation(done=True, reward=1.0, metadata={"k": "v"})
        result = serialize_observation(obs)

        assert result["reward"] == 1.0
        assert result["done"] is True
        assert "reward" not in result["observation"]
        assert "done" not in result["observation"]

    def test_metadata_not_inside_observation(self):
        """metadata must be a top-level sibling, not nested inside observation."""
        obs = Observation(done=False, reward=0.0, metadata={"step": 3})
        result = serialize_observation(obs)

        assert "metadata" not in result["observation"]
        assert result["metadata"]["step"] == 3

    def test_custom_observation_fields_in_observation_dict(self):
        """Subclass fields must appear inside the observation dict."""
        obs = CustomObservation(
            ally_tree="[ref=btn_1 role=button]",
            task_instruction="Book a ticket",
            done=False,
            reward=0.2,
            metadata={"variant": "label_drift"},
        )
        result = serialize_observation(obs)

        assert result["observation"]["ally_tree"] == "[ref=btn_1 role=button]"
        assert result["observation"]["task_instruction"] == "Book a ticket"
        assert result["metadata"]["variant"] == "label_drift"

    def test_reset_metadata_preserved(self):
        """ResetResponse must preserve metadata from the observation."""
        obs = Observation(metadata={"reset_key": "val"})
        serialized = serialize_observation(obs)
        reset_response = ResetResponse(**serialized)
        assert reset_response.metadata == {"reset_key": "val"}

    def test_step_response_metadata_preserved(self):
        """StepResponse must preserve metadata from the observation."""
        obs = Observation(metadata={"step_key": "val"})
        serialized = serialize_observation(obs)
        step_response = StepResponse(**serialized)
        assert step_response.metadata == {"step_key": "val"}


# ---------------------------------------------------------------------
# Round-trip: serialize -> client parse
# ---------------------------------------------------------------------


class TestMetadataRoundTrip:
    """End-to-end: serialize on server, parse on client."""

    def test_generic_client_receives_metadata(self):
        """GenericEnvClient._parse_result must surface metadata from payload."""
        obs = Observation(
            done=False,
            reward=0.42,
            metadata={"total_nodes": 6, "completed": ["origin", "dest"]},
        )
        payload = serialize_observation(obs)

        client = GenericEnvClient.__new__(GenericEnvClient)
        step_result = client._parse_result(payload)

        assert step_result.reward == 0.42
        assert step_result.done is False
        assert step_result.metadata is not None
        assert step_result.metadata["total_nodes"] == 6
        assert step_result.metadata["completed"] == ["origin", "dest"]

    def test_generic_client_handles_missing_metadata(self):
        """When server sends no metadata, StepResult.metadata should be None."""
        payload = {"observation": {"text": "hello"}, "reward": 0.0, "done": False}

        client = GenericEnvClient.__new__(GenericEnvClient)
        step_result = client._parse_result(payload)

        assert step_result.metadata is None
