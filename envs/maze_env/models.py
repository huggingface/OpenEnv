# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Maze Environment.

The maze environment is a simple gridworld with walls, a start cell, and an exit.
"""

from typing import List, Optional

from pydantic import Field

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    try:
        # Standalone imports with the current package namespace.
        from openenv.core.env_server.types import Action, Observation, State
    except ImportError:
        # Backward-compatible standalone imports when only openenv_core is available.
        from openenv_core.env_server.types import Action, Observation, State


class MazeAction(Action):
    """Action for the Maze environment."""

    action: int


class MazeObservation(Observation):
    """
    Observation from the Maze environment.

    Attributes:
        maze: The maze layout as a 2D grid (0 = empty, 1 = wall).
        position: Agent position as [col, row].
        total_reward: Cumulative reward in the current episode.
        legal_actions: List of valid action indices.
    """

    legal_actions: List[int] = Field(default_factory=list)
    current_position: List[int] = Field(default_factory=list)
    previous_position: List[int] = Field(default_factory=list)


class MazeState(State):
    """State for Maze environment."""

    episode_id: Optional[str] = None
    step_count: int
    done: bool = False
    current_position: List[int] = Field(default_factory=list)
    exit_cell: List[int] = Field(default_factory=list)
    status: str = "playing"  # e.g., "playing", "win", "lose"
