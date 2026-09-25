# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Abstract base class for harness adapters (RFC 005).

Each external harness (OpenClaw, Claude Code, etc.) gets a concrete adapter
that handles harness-specific startup, tool injection, and communication.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, List

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harnesses.types import HarnessConfig, HarnessEvent, HarnessResponse


class HarnessAdapter(ABC):
    """Abstract adapter for a specific harness implementation.

    Subclass this to integrate a new harness (OpenClaw, Claude Code, etc.).
    The adapter handles harness-specific startup, tool injection, and communication.

    The harness is a long-lived process that maintains conversation context.
    Each send_message() call is one conversational turn—the harness does its
    ReAct loop and returns when it has a response for the user.
    """

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    @abstractmethod
    async def start(self, working_directory: str) -> None:
        """Start the harness process.

        Args:
            working_directory: Path where the harness should operate.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the harness process and clean up resources."""
        ...

    @abstractmethod
    async def inject_tools(self, tools: List[Tool]) -> None:
        """Inject MCP tool definitions into the harness configuration.

        Called BEFORE start() so the harness discovers the tools at startup.

        Args:
            tools: List of MCP tool definitions to inject.
        """
        ...

    @abstractmethod
    async def send_message(self, message: str) -> HarnessResponse:
        """Send a message to the harness and get the response.

        Triggers one conversational turn. The harness maintains conversation
        context across calls.

        Args:
            message: The user message for this conversational turn.

        Returns:
            HarnessResponse containing the text response and turn events.
        """
        ...

    @abstractmethod
    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        """Send a message and stream intermediate events.

        Same semantics as send_message(), but yields HarnessEvents as
        they happen. The final event has type=TURN_COMPLETE.

        Args:
            message: The user message for this conversational turn.

        Yields:
            HarnessEvent instances as the harness processes the turn.
        """
        ...

    @abstractmethod
    async def is_alive(self) -> bool:
        """Check if the harness process is still running."""
        ...
