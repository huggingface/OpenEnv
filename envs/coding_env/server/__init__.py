# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coding environment server components.

Keep imports lazy so utility modules (for example transforms) remain importable
without pulling optional runtime dependencies like smolagents.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .python_codeact_env import PythonCodeActEnv

__all__ = ["PythonCodeActEnv"]


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


def __getattr__(name: str) -> Any:
    if name == "PythonCodeActEnv":
        from .python_codeact_env import PythonCodeActEnv

        return PythonCodeActEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
