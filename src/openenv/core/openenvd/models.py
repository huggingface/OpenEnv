# SPDX-License-Identifier: BSD-3-Clause

"""Process settings for the isolated environment worker."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskSpec(BaseModel):
    """Process settings passed to the isolation helper.

    Attributes:
        name (`str`): Unique task name (lowercase alphanumerics, ``-``, ``_``).
        argv (`list[str]`): Command to execute.
        cwd (`str`, *optional*): Child working directory; defaults to ``/``.
        uid (`int`, *optional*): Non-root UID; must be supplied with ``gid``.
        gid (`int`, *optional*): Non-root GID; must be supplied with ``uid``.
        network_isolated (`bool`): Require a separate Linux network namespace
            and an unprivileged UID/GID. Never falls back to shared networking.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=128)
    argv: list[str] = Field(min_length=1)
    cwd: Optional[str] = None
    uid: Optional[int] = Field(default=None, gt=0, lt=2**32 - 1, strict=True)
    gid: Optional[int] = Field(default=None, gt=0, lt=2**32 - 1, strict=True)
    network_isolated: bool = False

    @model_validator(mode="after")
    def validate_process_settings(self) -> TaskSpec:
        if (self.uid is None) != (self.gid is None):
            raise ValueError("uid and gid must be supplied together")
        if not self.argv[0] or any("\0" in arg for arg in self.argv):
            raise ValueError("argv requires a nonempty executable and no NUL bytes")
        if self.cwd is not None and (not self.cwd.startswith("/") or "\0" in self.cwd):
            raise ValueError("cwd must be an absolute path without NUL bytes")
        return self
