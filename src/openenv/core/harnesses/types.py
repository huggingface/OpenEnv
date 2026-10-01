# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Type definitions for agentic harness integration (RFC 005).

This module defines the Pydantic models and enums used for configuring,
communicating with, and observing external agentic harnesses.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Enums
# =============================================================================


class HarnessTransport(str, Enum):
    """How the harness exposes its interface."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "http"
    MCP = "mcp"


class HarnessEventType(str, Enum):
    """Types of events emitted during a harness turn."""

    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_CHUNK = "llm_chunk"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TEXT_OUTPUT = "text_output"
    ERROR = "error"
    TURN_COMPLETE = "turn_complete"


# =============================================================================
# Event and Response Models
# =============================================================================


class HarnessEvent(BaseModel):
    """A single event from a harness turn."""

    model_config = ConfigDict(extra="forbid")

    type: HarnessEventType = Field(description="The type of harness event")
    timestamp: float = Field(description="Unix timestamp of the event")
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Event-type-specific payload"
    )


class HarnessResponse(BaseModel):
    """Complete response from a single conversational turn."""

    model_config = ConfigDict(extra="forbid")

    response: str = Field(description="The harness's text response for this turn")
    events: List[HarnessEvent] = Field(
        default_factory=list, description="All events from this turn"
    )
    done: bool = Field(
        default=False,
        description="True if the harness considers the task complete",
    )


# =============================================================================
# Configuration
# =============================================================================


class HarnessConfig(BaseModel):
    """Configuration for an external agentic harness."""

    model_config = ConfigDict(extra="forbid")

    # Identity
    name: str = Field(description="Harness identifier, e.g. 'openclaw', 'claude-code'")

    # Process management
    command: List[str] = Field(
        description="Command to start the harness, e.g. ['openclaw', 'run']"
    )
    working_directory: str = Field(
        default="/workspace", description="Working directory for the harness process"
    )
    env_vars: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional environment variables for the harness process",
    )

    # Transport
    transport: HarnessTransport = Field(
        default=HarnessTransport.STDIO,
        description="How the harness exposes its interface",
    )

    # MCP tool injection
    mcp_config_path: Optional[str] = Field(
        default=None,
        description="Path to write MCP config for the harness. Auto-detected if None.",
    )

    # Timeouts
    startup_timeout_s: float = Field(
        default=30.0, gt=0, description="Max seconds to wait for harness startup"
    )
    session_timeout_s: float = Field(
        default=600.0, gt=0, description="Max seconds for a single session/episode"
    )

    # LLM configuration
    model: Optional[str] = Field(
        default=None, description="Override the harness's default model"
    )
    api_key_env_var: Optional[str] = Field(
        default=None, description="Environment variable name for LLM API key"
    )


# =============================================================================
# Action Type
# =============================================================================


class HarnessAction(Action):
    """Action for sending a message to an agentic harness.

    Each HarnessAction represents one conversational turn: the orchestrator
    sends a message, and the harness does its ReAct loop to produce a response.
    """

    type: Literal["harness_message"] = Field(
        default="harness_message", description="Action type discriminator"
    )
    message: str = Field(description="The user message for this conversational turn")
