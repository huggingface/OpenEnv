# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Chess Environment Client.

This module provides the client for connecting to a Chess Environment server
via WebSocket for persistent sessions.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import ChessAction, ChessObservation, ChessState


class ChessEnv(EnvClient[ChessAction, ChessObservation, ChessState]):
    """
    Client for Chess Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.

    Uses the moonfish chess engine for opponent moves and position evaluation.

    Example:
        >>> with ChessEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.fen)
        ...     print(result.observation.legal_moves)
        ...
        ...     result = client.step(ChessAction(move="e2e4"))
        ...     print(result.reward, result.done)
    """

    def _step_payload(self, action: ChessAction) -> Dict[str, Any]:
        """
        Convert ChessAction to JSON payload for step request.

        Args:
            action: ChessAction instance with UCI move string.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "move": action.move,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ChessObservation]:
        """
        Parse server response into StepResult[ChessObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with ChessObservation.
        """
        obs_data = payload.get("observation", {})

        observation = ChessObservation(
            fen=obs_data.get("fen", ""),
            legal_moves=obs_data.get("legal_moves", []),
            is_check=obs_data.get("is_check", False),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            result=obs_data.get("result"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ChessState:
        """
        Parse server response into ChessState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            ChessState object with environment state information.
        """
        return ChessState(
            episode_id=payload.get("episode_id", ""),
            fen=payload.get("fen", ""),
            current_player=payload.get("current_player", "white"),
            move_history=payload.get("move_history", []),
            step_count=payload.get("step_count", 0),
        )
