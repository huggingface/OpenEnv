# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Julia Environment.

The Julia environment executes Julia code and provides feedback through
compilation and unit test results.
"""

from openenv.core.env_server import Action, Observation, State


class JuliaAction(Action):
    """
    Action for the Julia environment - code to execute.

    Attributes:
        core_code: Core Julia code to execute
        test_code: Optional test code to execute. If not provided, only core_code runs.
    """

    core_code: str
    test_code: str | None = None


class JuliaObservation(Observation):
    """
    Observation from the Julia environment - execution results.

    Attributes:
        stdout: Standard output from Julia execution
        stderr: Standard error from Julia execution
        exit_code: Exit code (0 = success, non-zero = error)
        tests_passed: Number of tests passed (if tests were run)
        tests_failed: Number of tests failed (if tests were run)
        code_compiles: Whether the core code compiled/executed successfully
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    code_compiles: bool = True


class JuliaState(State):
    """
    State for Julia environment.

    Attributes:
        episode_id: Unique episode identifier
        step_count: Number of steps taken in episode
        last_exit_code: Exit code from last execution
        last_code_compiles: Whether the last code compiled successfully
        total_tests_passed: Cumulative tests passed in episode
        total_tests_failed: Cumulative tests failed in episode
    """

    last_exit_code: int = 0
    last_code_compiles: bool = True
    total_tests_passed: int = 0
    total_tests_failed: int = 0
