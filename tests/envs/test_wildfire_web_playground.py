# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Wildfire grid drawing in the web playground."""

from wildfire_env.models import WildfireAction
from wildfire_env.server.wildfire_environment import WildfireEnvironment


def test_wildfire_web_playground():
    env = WildfireEnvironment(width=8, height=8)
    obs = env.reset().model_dump()
    fire = obs["grid"].index(2)
    action = WildfireAction(action="water", x=fire % 8, y=fire // 8)
    obs = env.step(action).model_dump()
    board = env.render_web(obs)
    assert 'aria-label="Wildfire board"' in board
    assert board.count('title="x=') == 64
    assert f"x={fire % 8}, y={fire // 8}: water" in board
    assert f"water {obs['remaining_water']}</span>" in board
    assert "#" not in board  # theme colours only, so it reads in dark mode
    assert env.render_web({}) is None
