# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the Terminus environment."""

from typing import Any

from openenv.core.mcp_client import MCPToolClient

from .models import CommandResult, TerminusState


class TerminusEnv(MCPToolClient):
    """MCP client for calling the Terminus single-rollout tool."""

    def _parse_state(self, payload: dict[str, Any]) -> TerminusState:
        """Convert server state payloads to the Terminus state model."""

        def command_results(name: str) -> list[CommandResult]:
            values = payload.get(name, [])
            if not isinstance(values, list):
                return []
            return [
                value if isinstance(value, CommandResult) else CommandResult(**value)
                for value in values
                if isinstance(value, dict) or isinstance(value, CommandResult)
            ]

        return TerminusState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            sandbox_id=payload.get("sandbox_id"),
            setup_results=command_results("setup_results"),
            verify_commands=list(payload.get("verify_commands", []) or []),
            verify_results=command_results("verify_results"),
            commands=command_results("commands"),
            submitted_answer=payload.get("submitted_answer"),
            last_reward=payload.get("last_reward"),
            last_error=payload.get("last_error"),
        )
