# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State
except ImportError:  # standalone import path
    from core.client_types import StepResult
    from core.env_client import EnvClient
    from core.env_server.types import State

from .models import AdvocacyAction, AdvocacyObservation


class SophistryBenchSprintEnv(EnvClient[AdvocacyAction, AdvocacyObservation, State]):
    """Typed client for the sophistry-bench sprint OpenEnv environment."""

    def step_text(self, text: str) -> StepResult[AdvocacyObservation]:
        """Convenience: submit a raw argument string as an AdvocacyAction."""
        return super().step(AdvocacyAction(text=text))

    def _step_payload(self, action: AdvocacyAction) -> dict:
        return action.model_dump()

    def _parse_result(self, data: dict) -> StepResult[AdvocacyObservation]:
        obs_data = dict(data["observation"])
        # The framework's HTTP layer strips the base ``metadata`` dict from the
        # serialized observation, so the reward components arrive in the declared
        # ``components`` field (and the diagnostic message in ``error``). Rebuild
        # ``metadata`` here so the public contract holds — ``observation.metadata``
        # carries the eight components — preferring any metadata that survived
        # (in-process callers), else the mirrored ``components``.
        wire_metadata = obs_data.pop("metadata", None)
        metadata = (
            dict(wire_metadata)
            if wire_metadata
            else dict(obs_data.get("components") or {})
        )
        error = obs_data.get("error") or ""
        if error and "error" not in metadata:
            metadata["error"] = error
        # Construct once with metadata set, rather than mutating the model after.
        observation = AdvocacyObservation(**obs_data, metadata=metadata)
        return StepResult(
            observation=observation,
            reward=data["reward"],
            done=data["done"],
        )

    def _parse_state(self, data: dict) -> State:
        return State(**data)
