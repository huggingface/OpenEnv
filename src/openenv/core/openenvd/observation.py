# SPDX-License-Identifier: BSD-3-Clause
"""Daemon-owned episode events and filesystem snapshots."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import stat
import time
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .policy import ObservationEventType


class ObservationEvent(BaseModel):
    seq: int
    ts: float = Field(default_factory=time.time)
    type: ObservationEventType
    data: dict[str, Any]


class Collector:
    """Retain episode history outside the workload; subscribers use a sequence cursor."""

    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self.events: list[ObservationEvent] = []
        self.changed = asyncio.Event()

    def record(self, kind: ObservationEventType, data: dict) -> ObservationEvent:
        if len(self.events) >= self.max_events:
            raise RuntimeError("episode observation capacity exceeded")
        event = ObservationEvent(seq=len(self.events), type=kind, data=data)
        self.events.append(event.model_copy(deep=True))
        self.changed.set()
        return event

    def trajectory(self) -> list[dict]:
        return [event.model_dump(mode="json") for event in self.events]


def snapshot_changes(before: dict[str, str], after: dict[str, str]) -> list[dict]:
    """Compare scans using their relative paths in stable order."""
    return [
        {
            "path": path,
            "kind": "delete"
            if path not in after
            else "create"
            if path not in before
            else "modify",
        }
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path)
    ]


class Workspace:
    """Snapshot a dedicated workspace; never follow workload-controlled symlinks."""

    def __init__(
        self, path: Path, snapshot: Path, max_file_bytes: int = 64 * 1024 * 1024
    ):
        self.max_file_bytes = max_file_bytes
        self.path = path.resolve(strict=True)
        self.snapshot = snapshot.resolve()
        if self.path == Path("/") or self.snapshot.is_relative_to(self.path):
            raise ValueError("snapshot must be outside a dedicated workspace")
        self.ownership: dict[str, tuple[int, int]] = {}

    @cached_property
    def baseline(self) -> dict[str, str]:
        """Hash the immutable snapshot only when observation or grading needs it."""
        return self.scan(self.snapshot)

    def scan(self, directory: Path | None = None) -> dict[str, str]:
        directory = self.path if directory is None else directory
        result = {}
        for root, dirs, files, root_fd in os.fwalk(directory, follow_symlinks=False):
            for name in dirs + files:
                path = Path(root) / name
                relative = str(path.relative_to(directory))
                try:
                    info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        result[relative] = "link:" + os.readlink(name, dir_fd=root_fd)
                    elif stat.S_ISDIR(info.st_mode):
                        result[relative] = f"dir:{stat.S_IMODE(info.st_mode)}"
                    elif stat.S_ISREG(info.st_mode):
                        fd = os.open(
                            name,
                            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                            dir_fd=root_fd,
                        )
                        with os.fdopen(fd, "rb") as stream:
                            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                                continue
                            file_info = os.fstat(stream.fileno())
                            if file_info.st_size > self.max_file_bytes:
                                raise RuntimeError(
                                    "workspace file exceeds snapshot observation limit"
                                )
                            remaining = file_info.st_size
                            digest = hashlib.sha256()
                            while remaining:
                                chunk = stream.read(min(65536, remaining))
                                if not chunk:
                                    break
                                digest.update(chunk)
                                remaining -= len(chunk)
                            # Read exactly the initial size: concurrent growth
                            # must not turn an observation into an endless read.
                            digest.update(
                                f"{file_info.st_size}:{file_info.st_mtime_ns}".encode()
                            )
                            result[relative] = (
                                f"file:{stat.S_IMODE(info.st_mode)}:{digest.hexdigest()}"
                            )
                except (FileNotFoundError, OSError):
                    continue
        return result

    def disk_usage(self) -> int:
        total = 0
        for _, _, files, root_fd in os.fwalk(self.path, follow_symlinks=False):
            for name in files:
                try:
                    info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                    if stat.S_ISREG(info.st_mode):
                        total += info.st_size
                except OSError:
                    continue
        return total

    def capture(self) -> None:
        shutil.copytree(self.path, self.snapshot, symlinks=True)
        for root, dirs, files, root_fd in os.fwalk(self.path, follow_symlinks=False):
            info = os.fstat(root_fd)
            self.ownership[str(Path(root).relative_to(self.path))] = (
                info.st_uid,
                info.st_gid,
            )
            for name in dirs + files:
                info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                relative = str((Path(root) / name).relative_to(self.path))
                self.ownership[relative] = (info.st_uid, info.st_gid)

    def diff(self) -> list[dict]:
        return [
            {**change, "path": str(self.path / change["path"])}
            for change in snapshot_changes(self.baseline, self.scan())
        ]

    def restore(self) -> None:
        # Caller must stop all workload processes before restoring.
        for path in self.path.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        shutil.copytree(self.snapshot, self.path, symlinks=True, dirs_exist_ok=True)
        for relative, (uid, gid) in self.ownership.items():
            os.chown(self.path / relative, uid, gid, follow_symlinks=False)
