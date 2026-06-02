# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Maze Environment Client.

This module provides the client for connecting to a Maze Environment server
via WebSocket for persistent sessions.
"""

from typing import Dict

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State

    from .models import MazeAction, MazeObservation, MazeState
except ImportError:
    from models import MazeAction, MazeObservation, MazeState

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv_core.client_types import StepResult
    from openenv_core.env_client import EnvClient
    from openenv_core.env_server.types import State


class MazeEnv(EnvClient[MazeAction, MazeObservation, MazeState]):
    """
    Client for the Maze Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def _step_payload(self, action: MazeAction) -> Dict:
        """
        Convert MazeAction to JSON payload for step message.

        Args:
            action: MazeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MazeObservation]:
        """
        Parse server response into StepResult[MazeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MazeObservation
        """
        obs_data = payload.get("observation", {})
        observation = MazeObservation(
            legal_actions=obs_data.get("legal_actions", []),
            current_position=obs_data.get("current_position", []),
            previous_position=obs_data.get("previous_position", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> MazeState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return MazeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            done=payload.get("done", False),
            current_position=payload.get("current_position", []),
            exit_cell=payload.get("exit_cell", []),
            status=payload.get("status", "playing"),
        )
