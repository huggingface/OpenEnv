# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sandbox backends for harness-driven rollouts.

Provides the :class:`SandboxBackend` / :class:`SandboxHandle` protocols and
concrete implementations. Any harness adapter can use any backend — the
sandbox layer is orthogonal to the agent CLI choice.

The ``e2b`` import is wrapped in ``try/except`` so this package loads cleanly
in environments where ``e2b`` isn't installed (CI smoke tests, lint runs).
"""

from .base import BgJob, ExecResult, SandboxBackend, SandboxHandle

__all__ = [
    "BgJob",
    "ExecResult",
    "SandboxBackend",
    "SandboxHandle",
]

try:
    from .e2b_backend import E2BBgJob, E2BSandboxBackend, E2BSandboxHandle

    __all__.extend(["E2BBgJob", "E2BSandboxBackend", "E2BSandboxHandle"])
except ImportError:
    pass  # e2b not installed — stubs live in envs/opencode_env/sandbox/__init__.py
