# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the openenv __main__ module."""

from unittest.mock import patch

import pytest
from openenv.cli.__main__ import main
from typer.testing import CliRunner


runner = CliRunner()


def test_main_handles_keyboard_interrupt() -> None:
    """Test that main handles KeyboardInterrupt gracefully."""
    with patch("openenv.cli.__main__.app") as mock_app:
        mock_app.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 130


def test_main_handles_generic_exception() -> None:
    """Test that main handles generic exceptions gracefully."""
    with patch("openenv.cli.__main__.app") as mock_app:
        mock_app.side_effect = ValueError("Test error")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1


def test_main_entry_point() -> None:
    """Test that main() can be called as entry point."""
    # This tests the if __name__ == "__main__" block indirectly
    # by ensuring main() function works
    with patch("openenv.cli.__main__.app") as mock_app:
        main()
        mock_app.assert_called_once()
