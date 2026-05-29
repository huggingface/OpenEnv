# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small hf-sandbox wrapper for terminal-style environments."""

from __future__ import annotations

import os
from dataclasses import dataclass

_HF_SANDBOX_IMPORT_ERROR: ImportError | None = None

try:
    from hf_sandbox import Sandbox
except ImportError as _hf_sandbox_import_error:  # pragma: no cover
    _HF_SANDBOX_IMPORT_ERROR = _hf_sandbox_import_error
    Sandbox = None  # type: ignore[assignment]


DEFAULT_IMAGE = "python:3.12"
DEFAULT_FLAVOR = "cpu-basic"
DEFAULT_TIMEOUT = "1h"


@dataclass
class ShellResult:
    """Normalized result from a command executed in an HF sandbox."""

    stdout: str
    stderr: str
    error: str | None
    success: bool


class HFSandbox:
    """Manages one hf-sandbox job for one OpenEnv episode."""

    def __init__(
        self,
        *,
        image: str | None = None,
        flavor: str | None = None,
        timeout: str | None = None,
        forward_hf_token: bool | None = None,
    ):
        if Sandbox is None:
            raise ImportError(
                "hf-sandbox is not installed. Install the terminus_env package "
                "dependencies to use HFSandbox. Original import error: "
                f"{_HF_SANDBOX_IMPORT_ERROR}"
            )

        resolved_forward = _coerce_bool(
            os.getenv("HF_SANDBOX_FORWARD_HF_TOKEN", "false")
        )
        if forward_hf_token is not None:
            resolved_forward = bool(forward_hf_token)

        self._sandbox = Sandbox.create(
            image=image or os.getenv("HF_SANDBOX_IMAGE", DEFAULT_IMAGE),
            flavor=flavor or os.getenv("HF_SANDBOX_FLAVOR", DEFAULT_FLAVOR),
            timeout=timeout or os.getenv("HF_SANDBOX_TIMEOUT", DEFAULT_TIMEOUT),
            forward_hf_token=resolved_forward,
        )
        self.sandbox_id: str = self._sandbox.job_id

    def run_shell(self, command: str, timeout_s: int = 120) -> ShellResult:
        process = self._sandbox.exec(
            "bash",
            "-lc",
            command,
            timeout=timeout_s,
        )
        success = process.returncode == 0
        return ShellResult(
            stdout=process.stdout or "",
            stderr=process.stderr or "",
            error=None if success else f"exit code {process.returncode}",
            success=success,
        )

    def kill(self) -> None:
        try:
            self._sandbox.terminate()
        except Exception:
            pass


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
