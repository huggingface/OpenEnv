# SPDX-License-Identifier: BSD-3-Clause
"""Privileged Linux resource observation."""

from pathlib import Path


def resource_sample(cgroup: Path) -> dict:
    result = {"memory_bytes": int((cgroup / "memory.current").read_text())}
    result["cpu"] = {
        key: int(value)
        for key, value in (
            line.split() for line in (cgroup / "cpu.stat").read_text().splitlines()
        )
    }
    return result
