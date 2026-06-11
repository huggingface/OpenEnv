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
        observation = AdvocacyObservation(**data["observation"])
        # The framework's HTTP layer strips the base ``metadata`` dict from the
        # serialized observation, so the reward components arrive in the declared
        # ``components`` field. Re-populate ``metadata`` to keep the public
        # contract (``observation.metadata`` carries the eight components).
        if not observation.metadata and observation.components:
            observation.metadata = dict(observation.components)
        return StepResult(
            observation=observation,
            reward=data["reward"],
            done=data["done"],
        )

    def _parse_state(self, data: dict) -> State:
        return State(**data)
