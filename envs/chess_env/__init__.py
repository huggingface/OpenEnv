# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Chess Environment for OpenEnv.

This module provides OpenEnv integration for chess, using the moonfish
chess engine for position evaluation and opponent play.

Example:
    >>> from envs.chess_env import ChessEnv, ChessAction
    >>>
    >>> # Connect to a running server or start via Docker
    >>> env = ChessEnv.from_docker_image("chess-env:latest")
    >>>
    >>> # Reset and interact
    >>> result = env.reset()
    >>> print(result.observation.fen)
    >>> print(result.observation.legal_moves)
    >>>
    >>> result = env.step(ChessAction(move="e2e4"))
    >>> print(result.reward, result.done)
    >>>
    >>> # Cleanup
    >>> env.close()
"""

from .client import ChessEnv
from .models import ChessAction, ChessObservation, ChessState

__all__ = ["ChessEnv", "ChessAction", "ChessObservation", "ChessState"]
