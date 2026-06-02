# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

# Import your client and models DIRECTLY
from envs.grid_world_env.client import GridWorldEnv
from envs.grid_world_env.models import GridWorldAction, MoveAction


def test_grid_world_flow():
    """
    Test the full flow of the Grid World environment using the WebSocket client.
    """
    # 1. Initialize the client
    try:
        # We use a dummy URL for unit testing logic
        client = GridWorldEnv("ws://localhost:8000/ws")
    except Exception as e:
        pytest.fail(f"Failed to initialize client: {e}")

    # 2. Test Action Creation
    # FIX: Use GridWorldAction directly, not client.action_model
    action_up = GridWorldAction(action=MoveAction.UP)
    assert action_up.action == "UP"

    action_right = GridWorldAction(action=MoveAction.RIGHT)
    assert action_right.action == "RIGHT"

    # 3. Test Payload Serialization (The new abstract method you added)
    # This verifies that the strict method you wrote in client.py works correctly
    payload = client._step_payload(action_up)
    assert isinstance(payload, dict)
    assert payload["action"] == "UP"

    print("Grid World Client tests passed!")
