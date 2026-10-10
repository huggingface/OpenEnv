# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Wildfire grid drawing in the web playground."""

from envs.wildfire_env.models import WildfireAction
from envs.wildfire_env.server.wildfire_environment import WildfireEnvironment


def test_wildfire_draws_the_grid():
    env = WildfireEnvironment(width=8, height=8)
    obs = env.reset().model_dump()
    fire = obs["grid"].index(2)
    action = WildfireAction(action="water", x=fire % 8, y=fire // 8)
    obs = env.step(action).model_dump()
    board = env.render_web(obs)
    assert 'aria-label="Wildfire grid"' in board
    assert board.count('title="x=') == 64
    assert f"x={fire % 8}, y={fire // 8}: water" in board
    assert f"water {obs['remaining_water']}</span>" in board
    assert "#" not in board  # theme colours, so it reads in dark mode
    assert env.web_actions(obs) == []
    assert env.render_web({}) is None
