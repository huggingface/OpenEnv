# SPDX-License-Identifier: BSD-3-Clause
"""Linux cgroup v2 ownership for complete workload process-tree cleanup."""

import asyncio
import os
import uuid
from pathlib import Path

from .isolation import IsolationError


class WorkloadCgroup:
    """A dedicated cgroup prevents descendants escaping cleanup with setsid()."""

    def __init__(self, root: Path = Path("/sys/fs/cgroup")):
        self.path = root / ("openenvd-" + uuid.uuid4().hex)

    def create(self):
        try:
            self.path.mkdir(mode=0o700)
            if not (self.path / "cgroup.kill").exists():
                raise IsolationError("cgroup v2 with cgroup.kill is required")
            os.chmod(self.path, 0o700)
        except OSError as error:
            self.close()
            raise IsolationError(
                "a writable delegated cgroup v2 is required"
            ) from error

    async def kill(self):
        (self.path / "cgroup.kill").write_text("1")
        for _ in range(100):
            events = (self.path / "cgroup.events").read_text().splitlines()
            if "populated 0" in events:
                return
            await asyncio.sleep(0.01)
        raise IsolationError(
            "workload processes did not exit; refusing workspace restore"
        )

    def close(self):
        if self.path.exists():
            self.path.rmdir()
