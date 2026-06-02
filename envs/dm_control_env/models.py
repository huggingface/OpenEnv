# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the dm_control OpenEnv Environment.

This environment wraps dm_control.suite, providing access to all MuJoCo-based
continuous control tasks (cartpole, walker, humanoid, cheetah, etc.).
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class DMControlAction(Action):
    """
    Action for dm_control environments.

    All dm_control.suite environments use continuous actions represented as
    a list of float values. The size and bounds depend on the specific
    domain/task combination.

    Example (cartpole - 1D action):
        >>> action = DMControlAction(values=[0.5])  # Push cart right

    Example (walker - 6D action):
        >>> action = DMControlAction(values=[0.1, -0.2, 0.3, 0.0, -0.1, 0.2])

    Attributes:
        values: List of continuous action values. Shape and bounds depend on
            the loaded environment's action_spec.
    """

    values: List[float] = Field(
        default_factory=list,
        description="Continuous action values matching the environment's action_spec",
    )


class DMControlObservation(Observation):
    """
    Observation from dm_control environments.

    dm_control environments return observations as a dictionary of named arrays.
    Common observation keys include 'position', 'velocity', 'orientations', etc.
    The exact keys depend on the domain/task combination.

    Example observation keys by domain:
        - cartpole: 'position' (cos/sin of angle), 'velocity'
        - walker: 'orientations', 'height', 'velocity'
        - humanoid: 'joint_angles', 'head_height', 'extremities', 'torso_vertical', 'com_velocity'

    Attributes:
        observations: Dictionary mapping observation names to their values.
            Each value is a flattened list of floats.
        pixels: Optional base64-encoded PNG image of the rendered scene.
            Only included when render=True is passed to reset/step.
    """

    observations: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Named observation arrays from the environment",
    )
    pixels: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG image (when render=True)",
    )


class DMControlState(State):
    """
    Extended state for dm_control environments.

    Provides metadata about the currently loaded environment including
    the domain/task names and action/observation specifications.

    Attributes:
        episode_id: Unique identifier for the current episode.
        step_count: Number of steps taken in the current episode.
        domain_name: The dm_control domain (e.g., 'cartpole', 'walker').
        task_name: The specific task (e.g., 'balance', 'walk').
        action_spec: Specification of the action space including shape and bounds.
        observation_spec: Specification of the observation space.
        physics_timestep: The physics simulation timestep in seconds.
        control_timestep: The control timestep (time between actions) in seconds.
    """

    domain_name: str = Field(
        default="cartpole",
        description="The dm_control domain name",
    )
    task_name: str = Field(
        default="balance",
        description="The task name within the domain",
    )
    action_spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specification of the action space (shape, dtype, bounds)",
    )
    observation_spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specification of the observation space",
    )
    physics_timestep: float = Field(
        default=0.002,
        description="Physics simulation timestep in seconds",
    )
    control_timestep: float = Field(
        default=0.02,
        description="Control timestep (time between actions) in seconds",
    )


# Available dm_control.suite environments
# Format: (domain_name, task_name)
AVAILABLE_ENVIRONMENTS = [
    # Cartpole
    ("cartpole", "balance"),
    ("cartpole", "balance_sparse"),
    ("cartpole", "swingup"),
    ("cartpole", "swingup_sparse"),
    # Pendulum
    ("pendulum", "swingup"),
    # Point mass
    ("point_mass", "easy"),
    ("point_mass", "hard"),
    # Reacher
    ("reacher", "easy"),
    ("reacher", "hard"),
    # Ball in cup
    ("ball_in_cup", "catch"),
    # Finger
    ("finger", "spin"),
    ("finger", "turn_easy"),
    ("finger", "turn_hard"),
    # Fish
    ("fish", "upright"),
    ("fish", "swim"),
    # Cheetah
    ("cheetah", "run"),
    # Walker
    ("walker", "stand"),
    ("walker", "walk"),
    ("walker", "run"),
    # Hopper
    ("hopper", "stand"),
    ("hopper", "hop"),
    # Swimmer
    ("swimmer", "swimmer6"),
    ("swimmer", "swimmer15"),
    # Humanoid
    ("humanoid", "stand"),
    ("humanoid", "walk"),
    ("humanoid", "run"),
    # Manipulator
    ("manipulator", "bring_ball"),
    ("manipulator", "bring_peg"),
    ("manipulator", "insert_ball"),
    ("manipulator", "insert_peg"),
    # Acrobot
    ("acrobot", "swingup"),
    ("acrobot", "swingup_sparse"),
    # Stacker
    ("stacker", "stack_2"),
    ("stacker", "stack_4"),
    # Dog
    ("dog", "stand"),
    ("dog", "walk"),
    ("dog", "trot"),
    ("dog", "run"),
    ("dog", "fetch"),
    # Quadruped
    ("quadruped", "walk"),
    ("quadruped", "run"),
    ("quadruped", "escape"),
    ("quadruped", "fetch"),
]
