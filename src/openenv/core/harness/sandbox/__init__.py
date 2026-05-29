# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sandbox backends for harness-driven rollouts.

Provides the :class:`SandboxBackend` / :class:`SandboxHandle` protocols and
concrete implementations. Any harness adapter can use any backend -- the
sandbox layer is orthogonal to the agent CLI choice.

Optional backend imports are wrapped in ``try/except`` so this package
loads cleanly when dependencies aren't installed (CI smoke tests, lint).
"""

from typing import Any, Literal

from .base import BgJob, ExecResult, SandboxBackend, SandboxHandle
from .docker_backend import DockerBgJob, DockerSandboxBackend, DockerSandboxHandle
from .local_backend import LocalBgJob, LocalSandboxBackend, LocalSandboxHandle

__all__ = [
    "BgJob",
    "DockerBgJob",
    "DockerSandboxBackend",
    "DockerSandboxHandle",
    "ExecResult",
    "LocalBgJob",
    "LocalSandboxBackend",
    "LocalSandboxHandle",
    "SandboxBackend",
    "SandboxHandle",
    "create_sandbox_backend",
]

try:
    from .e2b_backend import E2BBgJob, E2BSandboxBackend, E2BSandboxHandle  # noqa: F401

    __all__.extend(["E2BBgJob", "E2BSandboxBackend", "E2BSandboxHandle"])
except ImportError:
    pass  # e2b not installed

try:
    from .hf_backend import HFBgJob, HFSandboxBackend, HFSandboxHandle  # noqa: F401

    __all__.extend(["HFBgJob", "HFSandboxBackend", "HFSandboxHandle"])
except ImportError:
    pass  # hf-sandbox not installed


def create_sandbox_backend(
    backend: Literal["e2b", "docker", "hf", "local"] = "e2b",
    **kwargs: Any,
) -> SandboxBackend:
    """Create a sandbox backend by name.

    For ``"e2b"``: works with both E2B cloud and CubeSandbox
    (set ``E2B_API_URL``).

    For ``"docker"``: local Docker, no external dependencies.

    For ``"hf"``: Hugging Face Jobs via ``hf-sandbox``.

    For ``"local"``: isolated temp directories and subprocesses on the host.
    """
    if backend == "e2b":
        from .e2b_backend import E2BSandboxBackend

        return E2BSandboxBackend(**kwargs)
    elif backend == "docker":
        return DockerSandboxBackend(**kwargs)
    elif backend == "hf":
        from .hf_backend import HFSandboxBackend

        return HFSandboxBackend(**kwargs)
    elif backend == "local":
        return LocalSandboxBackend(**kwargs)
    raise ValueError(
        f"Unknown sandbox backend: {backend!r}. Use 'e2b', 'docker', 'hf', or 'local'."
    )
