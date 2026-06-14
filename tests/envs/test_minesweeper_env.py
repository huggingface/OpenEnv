# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test suite for Minesweeper Environment."""

import os
import signal
import subprocess
import sys
import time
import unittest

import requests
from envs.minesweeper_env import (
    GameStatus,
    MinesweeperAction,
    MinesweeperEnv,
    MinesweeperObservation,
)
from envs.minesweeper_env.server.minesweeper_environment import MinesweeperEnvironment


class TestMinesweeperEnv(unittest.IsolatedAsyncioTestCase):
    """Test cases for the Minesweeper environment."""

    server_process = None

    @classmethod
    def setUpClass(cls):
        """Start the server once for all tests."""
        cls.server_process = subprocess.Popen(
            [sys.executable, "-m", "envs.minesweeper_env.server.app"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)  # Give server time to start

        # Verify server is running
        try:
            response = requests.get("http://127.0.0.1:8000/health")
            if response.status_code != 200:
                raise RuntimeError("Server health check failed")
        except requests.ConnectionError:
            raise RuntimeError("Server did not start or is unreachable")

    @classmethod
    def tearDownClass(cls):
        """Clean up server after all tests."""
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.kill(cls.server_process.pid, signal.SIGKILL)

            for stream in [
                cls.server_process.stdin,
                cls.server_process.stdout,
                cls.server_process.stderr,
            ]:
                if stream and not stream.closed:
                    stream.close()

    async def test_minesweeper_env_client(self):
        """Test Minesweeper environment client initialization."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            assert isinstance(client, MinesweeperEnv)

    async def test_minesweeper_initial_state(self):
        """Test the initial state after reset."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            result = await client.reset()
            observation = result.observation

            # Check observation type and attributes
            assert isinstance(observation, MinesweeperObservation)
            assert isinstance(observation.board, list)
            assert isinstance(observation.done, bool)
            assert isinstance(observation.reward, float)
            assert isinstance(observation.num_mines, int)
            assert isinstance(observation.flags_placed, int)
            assert isinstance(observation.cells_revealed, int)
            # game_status may be int or GameStatus enum due to HTTP serialization
            assert isinstance(observation.game_status, (GameStatus, int))

            # Check initial state values
            assert observation.done is False
            assert observation.reward == 0.0
            assert observation.flags_placed == 0
            assert observation.cells_revealed == 0
            # Compare with enum value (handles both int and enum)
            assert (
                observation.game_status == GameStatus.ONGOING.value
                or observation.game_status == GameStatus.ONGOING
            )

            # Check board structure (default 5x5)
            assert len(observation.board) == 5  # 5 rows
            assert all(len(row) == 5 for row in observation.board)  # 5 columns

            # Check all cells are initially unrevealed
            assert all(cell == -1 for row in observation.board for cell in row), (
                "All cells should be unrevealed (-1) at start"
            )

    async def test_reveal_action(self):
        """Test revealing a cell."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            await client.reset()

            # Try revealing a cell
            action = MinesweeperAction(row=0, col=0, action_type="reveal")
            result = await client.step(action)
            observation = result.observation

            assert isinstance(observation, MinesweeperObservation)
            assert observation.cells_revealed > 0, (
                "At least one cell should be revealed"
            )

    async def test_flag_action(self):
        """Test placing a flag."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            await client.reset()

            # Place a flag
            action = MinesweeperAction(row=1, col=1, action_type="flag")
            result = await client.step(action)
            observation = result.observation

            assert isinstance(observation, MinesweeperObservation)
            assert observation.flags_placed == 1, "One flag should be placed"
            assert observation.board[1][1] == "F", "Cell should show flag marker"

    async def test_toggle_flag(self):
        """Test toggling a flag on and off."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            await client.reset()

            # Place a flag
            action = MinesweeperAction(row=2, col=2, action_type="flag")
            result = await client.step(action)
            observation = result.observation
            assert observation.flags_placed == 1

            # Remove the flag
            action = MinesweeperAction(row=2, col=2, action_type="flag")
            result = await client.step(action)
            observation = result.observation
            assert observation.flags_placed == 0, "Flag should be removed"

    async def test_invalid_position(self):
        """Test action with invalid position."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            await client.reset()

            # Try invalid row
            action = MinesweeperAction(row=10, col=0, action_type="reveal")
            result = await client.step(action)
            observation = result.observation

            assert observation.reward < 0, (
                "Should receive negative reward for invalid action"
            )

            # Try invalid column
            action = MinesweeperAction(row=0, col=10, action_type="reveal")
            result = await client.step(action)
            observation = result.observation

            assert observation.reward < 0, (
                "Should receive negative reward for invalid action"
            )

    async def test_reveal_already_revealed(self):
        """Test revealing an already revealed cell."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            await client.reset()

            # Find a safe revealed cell to re-target. Mines end the game, so
            # we keep trying until we land on a non-mine revealed cell.
            candidates = [(2, 2), (1, 1), (0, 0), (4, 4), (3, 3)]
            test_row, test_col = None, None
            for r, c in candidates:
                action = MinesweeperAction(row=r, col=c, action_type="reveal")
                result = await client.step(action)
                ongoing = (
                    result.observation.game_status == GameStatus.ONGOING.value
                    or result.observation.game_status == GameStatus.ONGOING
                )
                if ongoing and result.observation.board[r][c] != -1:
                    test_row, test_col = r, c
                    break
                if not ongoing:
                    await client.reset()

            assert test_row is not None, "Could not reveal a safe cell across resets"

            # Re-revealing the same cell should be penalised.
            action = MinesweeperAction(row=test_row, col=test_col, action_type="reveal")
            result = await client.step(action)
            assert result.observation.reward < 0, (
                f"Should receive penalty for revealing already revealed cell, "
                f"got {result.observation.reward}"
            )

    async def test_game_status_ongoing(self):
        """Test that game status remains ONGOING during normal play."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            await client.reset()

            # Make a few safe moves
            action = MinesweeperAction(row=0, col=0, action_type="reveal")
            result = await client.step(action)

            # Game should still be ongoing if we didn't hit a mine or win
            # Handle both int and enum types for game_status
            if (
                result.observation.game_status == GameStatus.ONGOING.value
                or result.observation.game_status == GameStatus.ONGOING
            ):
                assert result.observation.done is False

    async def test_board_cell_values(self):
        """Test that board cells contain valid values."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            await client.reset()

            # Reveal a cell
            action = MinesweeperAction(row=2, col=2, action_type="reveal")
            result = await client.step(action)
            observation = result.observation

            # Check that revealed cells have valid values (0-8 or '*')
            for row in observation.board:
                for cell in row:
                    assert (
                        cell == -1  # Unrevealed
                        or cell == "F"  # Flagged
                        or cell == "*"  # Mine (if revealed)
                        or (
                            isinstance(cell, int) and 0 <= cell <= 8
                        )  # Number of adjacent mines
                    ), f"Invalid cell value: {cell}"

    async def test_metadata_in_observation(self):
        """Test that observations contain metadata."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            result = await client.reset()
            observation = result.observation

            assert hasattr(observation, "metadata"), "Observation should have metadata"
            assert isinstance(observation.metadata, dict), (
                "Metadata should be a dictionary"
            )

    async def test_multiple_steps(self):
        """Test taking multiple steps in the environment."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            await client.reset()

            # Take several actions
            actions = [
                MinesweeperAction(row=0, col=0, action_type="reveal"),
                MinesweeperAction(row=0, col=1, action_type="flag"),
                MinesweeperAction(row=1, col=0, action_type="reveal"),
            ]

            for action in actions:
                result = await client.step(action)
                assert isinstance(result.observation, MinesweeperObservation)

    async def test_reset_clears_state(self):
        """Test that reset properly clears the game state."""
        async with MinesweeperEnv(base_url="http://127.0.0.1:8000") as client:
            # First game
            await client.reset()
            action = MinesweeperAction(row=0, col=0, action_type="flag")
            await client.step(action)

            # Reset and check state is cleared
            result = await client.reset()
            observation = result.observation

            assert observation.flags_placed == 0, "Flags should be cleared after reset"
            assert observation.cells_revealed == 0, (
                "Revealed cells should be cleared after reset"
            )
            # Compare with enum value (handles both int and enum)
            assert (
                observation.game_status == GameStatus.ONGOING.value
                or observation.game_status == GameStatus.ONGOING
            ), "Game should be ongoing after reset"


class TestMinesweeperEnvironmentLogic(unittest.TestCase):
    """Direct unit tests against MinesweeperEnvironment (no HTTP server)."""

    def test_hitting_last_mine_does_not_overwrite_lost_with_won(self):
        """Revealing a mine must not be reclassified as a win, even if the
        revealed-cell count happens to equal total_cells - num_mines."""
        # 2x2 with 2 mines: safe = 2, mines = 2.
        # Pre-reveal 1 safe cell. Then reveal a mine — _revealed_cells grows
        # to 2, which equals total(4) - mines(2). The buggy check used to
        # count the mine and overwrite LOST with WON.
        env = MinesweeperEnvironment(height=2, width=2, num_mines=2)
        env.reset()

        env._mine_positions = {(0, 1), (1, 1)}
        env._compute_mine_counts()
        env._revealed_cells = {(0, 0)}
        env._flags_placed = set()
        env._game_status = GameStatus.ONGOING

        obs = env.step(MinesweeperAction(row=0, col=1, action_type="reveal"))

        assert env._game_status == GameStatus.LOST, (
            f"Game must be LOST after revealing a mine, got {env._game_status}"
        )
        assert obs.game_status == GameStatus.LOST
        assert obs.done is True
        assert obs.reward == -10.0

    def test_revealing_last_safe_cell_wins(self):
        """The normal win path still works after the LOST guard."""
        env = MinesweeperEnvironment(height=2, width=2, num_mines=1)
        env.reset()

        env._mine_positions = {(1, 1)}
        env._compute_mine_counts()
        env._revealed_cells = {(0, 0), (0, 1)}
        env._flags_placed = set()
        env._game_status = GameStatus.ONGOING

        obs = env.step(MinesweeperAction(row=1, col=0, action_type="reveal"))

        assert env._game_status == GameStatus.WON
        assert obs.game_status == GameStatus.WON
        assert obs.done is True

    def test_revealing_flagged_cell_returns_error_metadata(self):
        """A reject reason for a no-op action is surfaced in metadata.error."""
        env = MinesweeperEnvironment(height=3, width=3, num_mines=1)
        env.reset()

        env._mine_positions = {(2, 2)}
        env._compute_mine_counts()
        env._revealed_cells = set()
        env._flags_placed = {(0, 0)}
        env._game_status = GameStatus.ONGOING

        obs = env.step(MinesweeperAction(row=0, col=0, action_type="reveal"))

        assert obs.reward == -0.05
        assert obs.metadata.get("error"), (
            "Expected metadata.error explaining the rejection"
        )
        assert "flag" in obs.metadata["error"].lower()

    def test_flagging_revealed_cell_returns_error_metadata(self):
        """Flagging an already-revealed cell is rejected with an error message."""
        env = MinesweeperEnvironment(height=3, width=3, num_mines=1)
        env.reset()

        env._mine_positions = {(2, 2)}
        env._compute_mine_counts()
        env._revealed_cells = {(0, 0)}
        env._flags_placed = set()
        env._game_status = GameStatus.ONGOING

        obs = env.step(MinesweeperAction(row=0, col=0, action_type="flag"))

        assert obs.reward == -0.05
        assert obs.metadata.get("error"), (
            "Expected metadata.error explaining the rejection"
        )


if __name__ == "__main__":
    unittest.main()
