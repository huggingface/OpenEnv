# SPDX-License-Identifier: BSD-3-Clause
"""Pluggable enforcement backends for openenvd."""

from __future__ import annotations

from typing import Callable

from ..policy import OpenEnvDConfig
from .base import EnforcementBackend, EnforcementUnavailable, Sandbox

BackendFactory = Callable[[OpenEnvDConfig], EnforcementBackend]
_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    """
    Register an enforcement backend under `name`.

    Args:
        name (`str`):
            The name manifests use in `openenvd.enforcement.backend`.
        factory (`Callable[[OpenEnvDConfig], EnforcementBackend]`):
            Builds the backend from the environment's `openenvd:` block.
    """
    _REGISTRY[name] = factory


def get_backend(config: OpenEnvDConfig) -> EnforcementBackend:
    """
    Build the backend the config selects.

    Args:
        config ([`~openenv.core.openenvd.policy.OpenEnvDConfig`]):
            The validated `openenvd:` block.

    Returns:
        [`~openenv.core.openenvd.backends.EnforcementBackend`]: the backend.

    Raises:
        [`~openenv.core.openenvd.backends.EnforcementUnavailable`]:
            If no backend is registered under the selected name.
    """
    name = config.enforcement.backend
    factory = _REGISTRY.get(name)
    if factory is None:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise EnforcementUnavailable(name, f"no such backend (registered: {known})")
    return factory(config)


def _local(config: OpenEnvDConfig) -> EnforcementBackend:
    from .local import LocalBackend

    return LocalBackend(config)


def _openshell(config: OpenEnvDConfig) -> EnforcementBackend:
    from .openshell import OpenShellBackend

    return OpenShellBackend(config)


register_backend("local", _local)
register_backend("openshell", _openshell)

__all__ = [
    "EnforcementBackend",
    "EnforcementUnavailable",
    "Sandbox",
    "get_backend",
    "register_backend",
]
