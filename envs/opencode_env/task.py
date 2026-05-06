# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task payload accepted by :class:`OpenCodeSessionFactory`."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OpenCodeTask(BaseModel):
    """One task for an OpenCode rollout.

    The primitive only needs ``instruction`` (the prompt handed to ``opencode
    run``). Callers may attach ``setup_shell`` (run once inside the sandbox
    before the agent starts) and ``upload_files`` (written into the sandbox at
    absolute paths). Any additional metadata belongs in ``metadata`` and is
    passed through to the verifier untouched.
    """

    instruction: str
    setup_shell: str | None = None
    upload_files: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> "OpenCodeTask":
        """Accept a bare string, a dict, or an existing ``OpenCodeTask``."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(instruction=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            f"Cannot coerce {type(value).__name__} to OpenCodeTask; "
            "pass a str, dict, or OpenCodeTask."
        )
