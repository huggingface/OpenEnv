# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sandbox backends for the OpenCode harness.

The primitive ships with :class:`E2BSandboxBackend` as the default; any backend
that satisfies the :class:`SandboxBackend` / :class:`SandboxHandle` protocols
can be swapped in.

The ``e2b`` import is wrapped in ``try/except`` so this package can be loaded
in environments where ``e2b`` isn't installed (CI smoke tests, lint runs).
Instantiating ``E2BSandboxBackend`` without ``e2b`` raises a clear error.
"""

from .base import BgJob, ExecResult, SandboxBackend, SandboxHandle

try:
    from .e2b import E2BBgJob, E2BSandboxBackend, E2BSandboxHandle  # noqa: F401
except ImportError as _e2b_err:  # pragma: no cover

    class _RequiresE2B:
        """Stub raised when ``e2b`` is not installed.

        Lets the package import cleanly so unit tests, ``openenv validate``,
        and the docs build can run without the heavy ``e2b`` dependency.
        Actually constructing one of these classes raises a clear ImportError.
        """

        _e2b_import_error = _e2b_err

        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "e2b is not installed; install it via "
                "`pip install 'openenv-opencode-env[dev]'` or "
                "`pip install e2b` to use E2BSandboxBackend. "
                f"Original import error: {self._e2b_import_error}"
            )

    E2BBgJob = E2BSandboxBackend = E2BSandboxHandle = _RequiresE2B  # type: ignore[assignment]


__all__ = [
    "BgJob",
    "ExecResult",
    "SandboxBackend",
    "SandboxHandle",
    "E2BBgJob",
    "E2BSandboxBackend",
    "E2BSandboxHandle",
]
