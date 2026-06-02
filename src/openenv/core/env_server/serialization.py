# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared serialization and deserialization utilities for OpenEnv HTTP servers.

This module provides common utilities for converting between JSON dictionaries
and Pydantic models (Action/Observation) to eliminate code duplication across
HTTP server and web interface implementations.
"""

from typing import Any, Dict, Type

from .mcp_types import CallToolAction, ListToolsAction
from .types import Action, Observation

# MCP action types keyed by their "type" discriminator value.
# These are checked before the environment's own action_cls so that
# ListToolsAction / CallToolAction payloads are never rejected by an
# unrelated Pydantic model.
_MCP_ACTION_TYPES: Dict[str, Type[Action]] = {
    "list_tools": ListToolsAction,
    "call_tool": CallToolAction,
}


def deserialize_action(action_data: Dict[str, Any], action_cls: Type[Action]) -> Action:
    """
    Convert JSON dict to Action instance using Pydantic validation.

    MCP action types (``list_tools``, ``call_tool``) are recognised
    automatically via the ``"type"`` discriminator field, regardless of
    the environment's configured ``action_cls``.  All other payloads
    fall through to ``action_cls.model_validate()``.

    For special cases (e.g., tensor fields, custom type conversions),
    use deserialize_action_with_preprocessing().

    Args:
        action_data: Dictionary containing action data
        action_cls: The Action subclass to instantiate

    Returns:
        Action instance

    Raises:
        ValidationError: If action_data is invalid for the action class

    Note:
        This uses Pydantic's model_validate() for automatic validation.
    """
    # Route MCP action types before falling through to the env action_cls.
    # Only intercept when action_cls is the generic Action base or itself an
    # MCP type (i.e. the server hosts an MCP environment).  This avoids
    # silently bypassing env-specific validation for non-MCP environments
    # that happen to use "call_tool" / "list_tools" as a type discriminator.
    action_type = action_data.get("type")
    if action_type in _MCP_ACTION_TYPES:
        mcp_cls = _MCP_ACTION_TYPES[action_type]
        if action_cls is Action or action_cls in _MCP_ACTION_TYPES.values():
            return mcp_cls.model_validate(action_data)

    return action_cls.model_validate(action_data)


def deserialize_action_with_preprocessing(
    action_data: Dict[str, Any], action_cls: Type[Action]
) -> Action:
    """
    Convert JSON dict to Action instance with preprocessing for special types.

    This version handles common type conversions needed for web interfaces:
    - Converting lists/strings to tensors for 'tokens' field
    - Converting string action_id to int
    - Other custom preprocessing as needed

    Args:
        action_data: Dictionary containing action data
        action_cls: The Action subclass to instantiate

    Returns:
        Action instance

    Raises:
        ValidationError: If action_data is invalid for the action class
    """
    # Route MCP action types before preprocessing (they don't need it).
    # Same guard as deserialize_action: only intercept when action_cls is
    # the generic Action base or itself an MCP type.
    action_type = action_data.get("type")
    if action_type in _MCP_ACTION_TYPES:
        mcp_cls = _MCP_ACTION_TYPES[action_type]
        if action_cls is Action or action_cls in _MCP_ACTION_TYPES.values():
            return mcp_cls.model_validate(action_data)

    processed_data = {}

    for key, value in action_data.items():
        if key == "tokens" and isinstance(value, (list, str)):
            # Convert list or string to tensor
            if isinstance(value, str):
                # If it's a string, try to parse it as a list of numbers
                try:
                    import json

                    value = json.loads(value)
                except Exception:
                    # If parsing fails, treat as empty list
                    value = []
            if isinstance(value, list):
                try:
                    import torch  # type: ignore

                    processed_data[key] = torch.tensor(value, dtype=torch.long)
                except ImportError:
                    # If torch not available, keep as list
                    processed_data[key] = value
            else:
                processed_data[key] = value
        elif key == "action_id" and isinstance(value, str):
            # Convert action_id from string to int
            try:
                processed_data[key] = int(value)
            except ValueError:
                # If conversion fails, keep original value
                processed_data[key] = value
        else:
            processed_data[key] = value

    return action_cls.model_validate(processed_data)


def serialize_observation(observation: Observation) -> Dict[str, Any]:
    """
    Convert Observation instance to JSON-compatible dict using Pydantic.

    Args:
        observation: Observation instance

    Returns:
        Dictionary compatible with EnvClient._parse_result()

    The format matches what EnvClient expects:
    {
        "observation": {...},  # Observation fields
        "reward": float | None,
        "done": bool,
    }
    """
    # Use Pydantic's model_dump() for serialization
    obs_dict = observation.model_dump(
        exclude={
            "reward",
            "done",
            "metadata",
        }  # Exclude these from observation dict
    )

    # Extract reward and done directly from the observation
    reward = observation.reward
    done = observation.done

    # Return in EnvClient expected format
    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
    }
