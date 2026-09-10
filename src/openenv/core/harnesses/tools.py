# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tool conflict resolution utilities for harness integration (RFC 005)."""

from typing import List

from openenv.core.env_server.mcp_types import Tool


def resolve_tool_conflicts(
    env_tools: List[Tool],
    harness_builtin_tools: List[str],
) -> List[Tool]:
    """Detect and resolve tool name conflicts between environment and harness.

    When an environment exposes MCP tools that share names with the harness's
    built-in tools, we prefix the environment tools with ``env_`` to avoid
    collisions.

    Args:
        env_tools: MCP tool definitions from the environment.
        harness_builtin_tools: Names of the harness's built-in tools.

    Returns:
        List of tools with conflicts resolved via prefixing.
    """
    builtin_set = set(harness_builtin_tools)
    resolved: List[Tool] = []
    for tool in env_tools:
        if tool.name in builtin_set:
            resolved.append(tool.model_copy(update={"name": f"env_{tool.name}"}))
        else:
            resolved.append(tool)
    return resolved
