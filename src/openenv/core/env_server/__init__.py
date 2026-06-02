# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core environment interfaces and types."""

from .base_transforms import CompositeTransform, NullTransform
from .exceptions import (
    ConcurrencyConfigurationError,
    EnvironmentFactoryError,
    OpenEnvError,
    SessionCapacityError,
    SessionCreationError,
    SessionNotFoundError,
)
from .http_server import create_app, create_fastapi_app, HTTPEnvServer
from .interfaces import Environment, Message, ModelTokenizer, Transform

try:
    from .mcp_environment import MCPEnvironment
except ModuleNotFoundError:
    MCPEnvironment = None  # type: ignore[assignment]

from .mcp_types import (
    CallToolAction,
    CallToolObservation,
    JsonRpcError,
    # JSON-RPC types
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    ListToolsAction,
    ListToolsObservation,
    McpMethod,
    RESERVED_TOOL_NAMES,
    Tool,
    ToolError,
    ToolErrorType,
    WSMCPMessage,
    WSMCPResponse,
)
from .route_config import GetEndpointConfig
from .serialization import (
    deserialize_action,
    deserialize_action_with_preprocessing,
    serialize_observation,
)
from .types import (
    Action,
    BaseMessage,
    ConcurrencyConfig,
    HealthResponse,
    HealthStatus,
    Observation,
    SchemaResponse,
    ServerCapacityStatus,
    ServerMode,
    SessionInfo,
    State,
    WSCloseMessage,
    WSErrorCode,
    WSErrorResponse,
    WSIncomingMessage,
    WSObservationResponse,
    WSResetMessage,
    WSStateMessage,
    WSStateResponse,
    WSStepMessage,
)

try:
    from .web_interface import create_web_interface_app, WebInterfaceManager
except ModuleNotFoundError:
    create_web_interface_app = None  # type: ignore[assignment]
    WebInterfaceManager = None  # type: ignore[assignment]

__all__ = [
    # Core interfaces
    "Environment",
    "Transform",
    "Message",
    "ModelTokenizer",
    # Types
    "Action",
    "Observation",
    "State",
    "SchemaResponse",
    "HealthResponse",
    # Enums
    "HealthStatus",
    "ServerMode",
    "WSErrorCode",
    # WebSocket message types
    "BaseMessage",
    "WSIncomingMessage",
    "WSResetMessage",
    "WSStepMessage",
    "WSStateMessage",
    "WSCloseMessage",
    "WSObservationResponse",
    "WSStateResponse",
    "WSErrorResponse",
    # Concurrency types
    "ConcurrencyConfig",
    "ServerCapacityStatus",
    "SessionInfo",
    # Exceptions
    "OpenEnvError",
    "ConcurrencyConfigurationError",
    "SessionCapacityError",
    "SessionNotFoundError",
    "SessionCreationError",
    "EnvironmentFactoryError",
    # Base transforms
    "CompositeTransform",
    "NullTransform",
    # HTTP Server
    "HTTPEnvServer",
    "create_app",
    "create_fastapi_app",
    # Web Interface
    "create_web_interface_app",
    "WebInterfaceManager",
    # Serialization utilities
    "deserialize_action",
    "deserialize_action_with_preprocessing",
    "serialize_observation",
    # Route configuration
    "GetEndpointConfig",
    # MCP types
    "Tool",
    "ToolError",
    "ToolErrorType",
    "ListToolsAction",
    "CallToolAction",
    "ListToolsObservation",
    "CallToolObservation",
    "WSMCPMessage",
    "WSMCPResponse",
    "RESERVED_TOOL_NAMES",
    "MCPEnvironment",
    # JSON-RPC types
    "JsonRpcErrorCode",
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "McpMethod",
]
