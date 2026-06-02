# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
JuliaEnv
--------
Client-side wrapper for the Julia environment server.

This client maintains a persistent WebSocket connection to the environment
server, enabling efficient multi-step interactions with lower latency.

- Users instantiate JuliaEnv with a base_url provided by the higher-level
  vector/orchestration layer.
- Environment authors ship the Docker image that serves the API.
"""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import JuliaAction, JuliaObservation, JuliaState


class JuliaEnv(EnvClient[JuliaAction, JuliaObservation, JuliaState]):
    """
    WebSocket client for the Julia Environment.

    This client connects to a JuliaEnvironment server and provides
    methods to interact with it: reset(), step(), and state access.

    The default message timeout is set to 180 seconds to accommodate:
    - Server execution timeout: 120s
    - Process pool worker wait: 30s
    - Network overhead: 30s buffer

    Example:
        >>> # Connect to a running server
        >>> client = JuliaEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.stdout)
        >>>
        >>> # Execute Julia code
        >>> action = JuliaAction(
        ...     core_code='''
        ...     function multiply(a, b)
        ...         return a * b
        ...     end
        ...     ''',
        ...     test_code='''
        ...     using Test
        ...     @test multiply(3, 4) == 12
        ...     '''
        ... )
        >>> result = client.step(action)
        >>> print(result.observation.tests_passed)  # 1
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = JuliaEnv.from_docker_image("julia-env:latest")
        >>> result = client.reset()
        >>> result = client.step(JuliaAction(core_code="println(2 + 2)", test_code=""))
        >>> print(result.observation.stdout)  # "4\\n"
        >>> client.close()
    """

    # Override default timeout to accommodate Julia execution + worker wait
    DEFAULT_MESSAGE_TIMEOUT = 180.0  # 120s execution + 30s worker wait + 30s buffer

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float | None = None,
        **kwargs,
    ):
        """
        Initialize JuliaEnv client with appropriate timeout.

        Args:
            base_url: Base URL of the Julia environment server
            connect_timeout_s: Timeout for establishing WebSocket connection
            message_timeout_s: Timeout for receiving responses (default: 180.0)
            **kwargs: Additional arguments passed to EnvClient
        """
        if message_timeout_s is None:
            message_timeout_s = self.DEFAULT_MESSAGE_TIMEOUT
        super().__init__(
            base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            **kwargs,
        )

    # --- EnvClient abstract hooks ---

    def _step_payload(self, action: JuliaAction) -> dict:
        """
        Convert JuliaAction to JSON payload for step request.

        Args:
            action: JuliaAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "core_code": action.core_code,
            "test_code": action.test_code,
        }

    def _parse_result(self, payload: dict) -> StepResult[JuliaObservation]:
        """
        Parse server response into StepResult[JuliaObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with JuliaObservation
        """
        obs_data = payload.get("observation", {})
        observation = JuliaObservation(
            stdout=obs_data.get("stdout", ""),
            stderr=obs_data.get("stderr", ""),
            exit_code=obs_data.get("exit_code", 0),
            tests_passed=obs_data.get("tests_passed", 0),
            tests_failed=obs_data.get("tests_failed", 0),
            code_compiles=obs_data.get("code_compiles", True),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult[JuliaObservation](
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> JuliaState:
        """
        Parse server response into JuliaState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            JuliaState object with episode metadata
        """
        return JuliaState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            last_exit_code=payload.get("last_exit_code", 0),
            last_code_compiles=payload.get("last_code_compiles", True),
            total_tests_passed=payload.get("total_tests_passed", 0),
            total_tests_failed=payload.get("total_tests_failed", 0),
        )
