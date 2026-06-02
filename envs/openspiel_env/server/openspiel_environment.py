# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenSpiel Environment Server Implementation.

This module wraps OpenSpiel's rl_environment.Environment and exposes it
via the OpenEnv Environment interface.
"""

import uuid
from typing import Any, Dict

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.interfaces import Environment

    from ..models import OpenSpielAction, OpenSpielObservation, OpenSpielState
    from .opponent_policies import get_opponent_policy, OpponentPolicy
except ImportError:
    from models import OpenSpielAction, OpenSpielObservation, OpenSpielState

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.interfaces import Environment
    from server.opponent_policies import get_opponent_policy, OpponentPolicy

# Import OpenSpiel
try:
    import pyspiel
    from open_spiel.python import rl_environment
except ImportError as e:
    raise ImportError(
        "OpenSpiel is not installed. "
        "Please install it following instructions at: "
        "https://github.com/google-deepmind/open_spiel"
    ) from e


class OpenSpielEnvironment(Environment):
    """
    OpenSpiel Environment wrapper for OpenEnv.

    This environment wraps OpenSpiel games and provides a single-agent interface.
    For multi-player games, the agent controls one player while opponent(s) use
    a fixed policy (e.g., random).

    Supported games:
    - Single-player: catch, cliff_walking, 2048, blackjack
    - Multi-player: tic_tac_toe, kuhn_poker

    Args:
        game_name: Name of the OpenSpiel game (e.g., "catch", "tic_tac_toe").
        agent_player: Which player ID the agent controls (default 0).
        opponent_policy: Policy for opponent players ("random", "first", etc.).
        game_params: Optional game-specific parameters.

    Example:
        >>> env = OpenSpielEnvironment("catch")
        >>> obs = env.reset()
        >>> print(obs.info_state)  # Agent's observation
        >>> obs = env.step(OpenSpielAction(action_id=1))
        >>> print(obs.reward)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        game_name: str = "catch",
        agent_player: int = 0,
        opponent_policy: str = "random",
        game_params: Dict[str, Any] | None = None,
    ):
        """Initialize OpenSpiel environment."""
        super().__init__()

        self.game_name = game_name
        self.agent_player = agent_player
        self.game_params = game_params or {}

        # Create OpenSpiel environment
        try:
            self._ospiel_env = rl_environment.Environment(game_name, **self.game_params)
        except Exception as e:
            raise ValueError(
                f"Failed to create OpenSpiel game '{game_name}': {e}"
            ) from e

        self.num_players = self._ospiel_env.num_players
        self.is_turn_based = self._ospiel_env.is_turn_based

        # Validate agent_player
        if agent_player >= self.num_players:
            raise ValueError(
                f"agent_player={agent_player} >= num_players={self.num_players}"
            )

        # Set up opponent policy for multi-player games
        self.opponent_policy_fn: OpponentPolicy | None = None
        if self.num_players > 1:
            self.opponent_policy_fn = get_opponent_policy(opponent_policy)

        # Initialize state
        self._state = OpenSpielState(
            game_name=game_name,
            agent_player=agent_player,
            opponent_policy=opponent_policy,
            game_params=self.game_params,
            num_players=self.num_players,
        )

        # Track last opponent action for learning
        self._last_opponent_action: int | None = None

    def reset(self) -> OpenSpielObservation:
        """
        Reset the environment and return initial observation.

        For multi-player games, this will autoplay opponent turns until
        it's the agent's turn (or terminal state).

        Returns:
            Initial observation for the agent.
        """
        # Reset OpenSpiel environment
        time_step = self._ospiel_env.reset()

        # Reset state tracking
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._last_opponent_action = None

        # Autoplay opponent turns until agent's turn
        time_step = self._auto_play_opponents(time_step)

        # Convert to OpenEnv observation
        return self._make_observation(time_step)

    def step(self, action: OpenSpielAction) -> OpenSpielObservation:  # type: ignore[override]
        """
        Execute agent's action and return resulting observation.

        For multi-player games, this will:
        1. Apply the agent's action
        2. Autoplay opponent turns until it's the agent's turn again
        3. Return the observation from the agent's perspective

        Args:
            action: OpenSpielAction containing the action_id to execute.

        Returns:
            Observation after action execution (and opponent turns if multi-player).

        Raises:
            ValueError: If action is not an OpenSpielAction.
        """
        if not isinstance(action, OpenSpielAction):
            raise ValueError(f"Expected OpenSpielAction, got {type(action)}")

        # Apply agent's action
        if self.is_turn_based:
            # Turn-based: single action
            time_step = self._ospiel_env.step([action.action_id])
        else:
            # Simultaneous-move: need actions for all players
            # For now, only support agent as player 0 in simultaneous games
            if self.agent_player != 0:
                raise NotImplementedError(
                    "Simultaneous-move games only support agent_player=0"
                )
            # Get opponent actions
            opponent_actions = []
            for player_id in range(self.num_players):
                if player_id == self.agent_player:
                    opponent_actions.append(action.action_id)
                else:
                    legal_actions = time_step.observations["legal_actions"][player_id]
                    opp_action = self.opponent_policy_fn.select_action(
                        legal_actions, time_step.observations
                    )
                    opponent_actions.append(opp_action)
            time_step = self._ospiel_env.step(opponent_actions)

        self._state.step_count += 1

        # Autoplay opponent turns (for turn-based games)
        if self.is_turn_based:
            time_step = self._auto_play_opponents(time_step)

        # Convert to OpenEnv observation
        return self._make_observation(time_step)

    @property
    def state(self) -> OpenSpielState:
        """Get current environment state."""
        return self._state

    def _auto_play_opponents(self, time_step) -> Any:
        """
        Autoplay opponent turns until it's the agent's turn or game is terminal.

        Args:
            time_step: Current TimeStep from OpenSpiel environment.

        Returns:
            Updated TimeStep after opponent moves.
        """
        # Single-player games: nothing to do
        if self.num_players == 1:
            return time_step

        # Multi-player games: play opponent turns
        while (
            not time_step.last()
            and time_step.observations["current_player"] != self.agent_player
        ):
            current_player = time_step.observations["current_player"]
            legal_actions = time_step.observations["legal_actions"][current_player]

            # Select opponent action
            opp_action = self.opponent_policy_fn.select_action(
                legal_actions, time_step.observations
            )
            self._last_opponent_action = opp_action

            # Apply opponent action
            time_step = self._ospiel_env.step([opp_action])
            self._state.step_count += 1

        return time_step

    def _make_observation(self, time_step) -> OpenSpielObservation:
        """
        Convert OpenSpiel TimeStep to OpenEnv Observation.

        Args:
            time_step: OpenSpiel TimeStep object.

        Returns:
            OpenSpielObservation for the agent.
        """
        # Extract agent's information
        info_state = time_step.observations["info_state"][self.agent_player]
        legal_actions = time_step.observations["legal_actions"][self.agent_player]
        current_player_id = time_step.observations["current_player"]

        # Determine game phase
        if time_step.last():
            game_phase = "terminal"
        elif time_step.first():
            game_phase = "initial"
        else:
            game_phase = "playing"

        # Get reward for agent
        reward = None
        if time_step.rewards is not None:
            reward = float(time_step.rewards[self.agent_player])

        # Create observation
        obs = OpenSpielObservation(
            info_state=info_state.tolist()
            if hasattr(info_state, "tolist")
            else list(info_state),
            legal_actions=legal_actions,
            game_phase=game_phase,
            current_player_id=current_player_id,
            opponent_last_action=self._last_opponent_action,
            done=time_step.last(),
            reward=reward,
        )

        return obs
