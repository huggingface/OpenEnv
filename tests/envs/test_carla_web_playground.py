# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The CARLA env draws a road sketch and offers the scenario's tools as buttons in the web playground."""

from carla_env.models import CarlaAction
from carla_env.server.carla_environment import CarlaEnvironment


def test_carla_offers_scenario_tools_and_a_sketch():
    env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
    env.reset()
    obs = env.step(CarlaAction(action_type="observe")).model_dump()
    assert [label for label, _ in env.web_actions(obs)] == [
        "Observe",
        "Lane left",
        "Lane right",
        "Emergency stop",
        "Brake 50%",
        "Accelerate",
    ]
    for _, action in env.web_actions(obs):
        CarlaAction(**action)
    sketch = env.render_web(obs)
    assert 'aria-label="CARLA scene sketch: 3 pedestrians in your lane' in sketch
    assert sketch.count("<circle") == 3
    assert "#" not in sketch  # theme colours only, so it reads in dark mode
    assert env.render_web({}) is None
