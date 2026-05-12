# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Agent registry and public API for CLI-based agentic harnesses.

The registry maps agent names (``"opencode"``, ``"claude-code"``, etc.) to
their :class:`CLIAgentSpec` declarations. Each agent module registers itself
via :func:`register_agent` at import time.

Usage::

    from openenv.core.harness.agents import get_agent_spec, list_agents

    spec = get_agent_spec("opencode")
    print(list_agents())  # ["opencode"]
"""

from __future__ import annotations

from .base import (
    AgentConfig,
    AgentEvent,
    AgentTask,
    ArtifactSpec,
    CLIAgentSpec,
    MCPConfigSpec,
)

# Registry

_REGISTRY: dict[str, CLIAgentSpec] = {}


def register_agent(spec: CLIAgentSpec) -> None:
    """Register a :class:`CLIAgentSpec` under ``spec.name``.

    Raises :class:`ValueError` if the name is already registered with a
    *different* spec object (re-registering the same object is a no-op,
    which makes ``importlib.reload`` safe).
    """
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing is not spec:
        raise ValueError(
            f"Agent {spec.name!r} is already registered. "
            "Use a unique name or call unregister_agent() first."
        )
    _REGISTRY[spec.name] = spec


def unregister_agent(name: str) -> CLIAgentSpec | None:
    """Remove a registered agent spec, returning it (or ``None``)."""
    return _REGISTRY.pop(name, None)


def get_agent_spec(name: str) -> CLIAgentSpec:
    """Look up a registered agent spec by name.

    Raises :class:`KeyError` if not found. To trigger auto-registration of
    built-in agents, import the specific module first (e.g.
    ``import openenv.core.harness.agents.opencode``).
    """
    if name not in _REGISTRY:
        # Auto-import built-in agent modules to trigger registration.
        _auto_import(name)
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown agent {name!r}. Registered agents: {available}"
        ) from None


def list_agents() -> list[str]:
    """Return sorted names of all registered agents."""
    return sorted(_REGISTRY)


def _auto_import(name: str) -> None:
    """Try to import the built-in module for ``name`` to trigger registration."""
    # Map agent names to module names (handles hyphens).
    module_name = name.replace("-", "_")
    try:
        __import__(f"openenv.core.harness.agents.{module_name}", fromlist=["_"])
    except ImportError:
        pass


# Convenience re-exports

__all__ = [
    # Registry
    "get_agent_spec",
    "list_agents",
    "register_agent",
    "unregister_agent",
    # Base types
    "AgentConfig",
    "AgentEvent",
    "AgentTask",
    "ArtifactSpec",
    "CLIAgentSpec",
    "MCPConfigSpec",
]
