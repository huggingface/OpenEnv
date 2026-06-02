# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree
import uuid

try:
    from openenv.core.env_server import Environment
    from openenv.core.env_server.types import State
except ImportError:
    from core.env_server import Environment
    from core.env_server.types import State
from ..models import GridWorldAction, GridWorldObservation, MoveAction


class GridWorldEnvironment(Environment):
    """
    A simple 5x5 Grid World environment.

    The agent starts at [0, 0] and must navigate to [4, 4].
    """

    def __init__(self):
        super().__init__()

        # ---Environment Configuration ---
        self.grid_size = 5
        self.goal_pos = [4, 4]

        # --- Internal State Variable ---
        self.agent_x = 0
        self.agent_y = 0

        # Initialize the base OpenEnv State Container
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

    def reset(self) -> GridWorldObservation:
        # Update State
        self.agent_x = 0
        self.agent_y = 0

        self._state.step_count = 0
        self._state.episode_id = str(uuid.uuid4())
        # Return initial observation (reward must be float 0.0, not None)
        return GridWorldObservation(
            x=self.agent_x,
            y=self.agent_y,
            message="Welcome to Grid World! Goal is at [4, 4].",
            reward=0.0,
            done=False,
        )

    def step(self, action: GridWorldAction) -> GridWorldObservation:
        # Increment step counter in the base State
        self._state.step_count += 1
        # =============================================
        move = action.action

        # self._state.episode_steps += 1

        # Use current state
        # current_x = self._state.agent_x
        # current_y = self._state.agent_y

        move = action.action

        if move == MoveAction.UP:
            self.agent_x -= 1
        elif move == MoveAction.DOWN:
            self.agent_x += 1
        elif move == MoveAction.LEFT:
            self.agent_y -= 1
        elif move == MoveAction.RIGHT:
            self.agent_y += 1

        # Clamp to boundaries
        self.agent_x = max(0, min(self.agent_x, self.grid_size - 1))
        self.agent_y = max(0, min(self.agent_y, self.grid_size - 1))

        # # Update State
        # self._state.agent_x = current_x
        # self._state.agent_y = current_y

        # Logic
        done = False
        message = "Keep going..."
        reward = -0.1

        if [self.agent_x, self.agent_y] == self.goal_pos:
            reward = 1.0
            done = True
            message = "You found the goal!"

        return GridWorldObservation(
            x=self.agent_x, y=self.agent_y, message=message, reward=reward, done=done
        )

    @property
    def state(self) -> State:
        return self._state
