# SPDX-License-Identifier: BSD-3-Clause
"""Write-only event publication for workload-side harness adapters."""

import json
import os
from dataclasses import asdict, is_dataclass


class HarnessEventSink:
    """Publish adapter events without access to the retained trajectory.

    The supervisor supplies a write-only pipe descriptor through
    ``OPENENVD_EVENT_FD``. Payloads are observations reported by the workload,
    never evidence that an OS operation actually happened.
    """

    def __init__(self, fd: int | None = None):
        self.fd = int(os.environ["OPENENVD_EVENT_FD"]) if fd is None else fd

    def __call__(self, event):
        if is_dataclass(event):
            event = asdict(event)
        elif hasattr(event, "model_dump"):
            event = event.model_dump(mode="json")
        payload = json.dumps(event).encode() + b"\n"
        if len(payload) > os.fpathconf(self.fd, "PC_PIPE_BUF"):
            raise ValueError("harness event exceeds atomic pipe capacity")
        os.write(self.fd, payload)
