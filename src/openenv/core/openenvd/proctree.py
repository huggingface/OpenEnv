# SPDX-License-Identifier: BSD-3-Clause

"""Zones as process trees, for units where openenvd can't create cgroups.

In the `landlock` tier there is no writable cgroup root. Each container is a
shim process openenvd starts directly; its subtree is found by walking
`/proc` parent links (the shim is a child subreaper, so orphans stay below it).
Limits are enforced by a watchdog that kills the whole container when it goes
over: `detected_and_reaped`, never `prevented`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Any, Callable

from .contract import Resources, Strength

logger = logging.getLogger(__name__)

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _children(proc: Path) -> dict[int, list[int]]:
    tree: dict[int, list[int]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        ppid = int(stat[stat.rindex(")") + 2 :].split()[1])
        tree.setdefault(ppid, []).append(int(entry.name))
    return tree


def descendants(root: int, proc: Path = Path("/proc")) -> list[int]:
    """`root` and every process below it."""
    tree = _children(proc)
    out, stack = [], [root]
    while stack:
        pid = stack.pop()
        if (proc / str(pid)).exists():
            out.append(pid)
        stack.extend(tree.get(pid, []))
    return out


def _rss(pid: int, proc: Path) -> int:
    try:
        return int((proc / str(pid) / "statm").read_text().split()[1]) * _PAGE
    except (OSError, IndexError, ValueError):
        return 0


class ProcessTree:
    """The [`CgroupTree`] interface, implemented with process trees and a watchdog.

    Args:
        proc (`Path`, *optional*, defaults to `/proc`):
            The procfs to read.
        on_violation (`Callable[[str, str], None]`, *optional*):
            Called with the container's path and what it exceeded.
    """

    def __init__(
        self,
        proc: Path = Path("/proc"),
        on_violation: Callable[[str, str], None] | None = None,
    ) -> None:
        self.proc = proc
        self.on_violation = on_violation
        self._roots: dict[str, int] = {}
        self._limits: dict[str, Resources] = {}
        self._frozen: set[str] = set()
        self._watchdog: asyncio.Task | None = None

    def register(self, rel: str, pid: int) -> None:
        """Track `pid` (a shim) as the root of container `rel`."""
        self._roots[rel.strip("/")] = pid
        if self._watchdog is None or self._watchdog.done():
            try:
                self._watchdog = asyncio.get_running_loop().create_task(self._watch())
            except RuntimeError:
                pass

    def _under(self, rel: str) -> list[str]:
        rel = rel.strip("/")
        return [
            r for r in self._roots if r == rel or r.startswith(rel + "/") or not rel
        ]

    def create(self, rel: str) -> Path:
        return Path(rel)

    def set_limits(self, rel: str, resources: Resources) -> None:
        self._limits[rel.strip("/")] = resources

    def pids(self, rel: str) -> list[int]:
        out: list[int] = []
        for r in self._under(rel):
            out += descendants(self._roots[r], self.proc)
        return out

    def _signal(self, rel: str, sig: int) -> None:
        for pid in self.pids(rel):
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    def freeze(self, rel: str) -> None:
        self._signal(rel, signal.SIGSTOP)
        self._frozen.add(rel.strip("/"))

    def thaw(self, rel: str) -> None:
        self._signal(rel, signal.SIGCONT)
        self._frozen.discard(rel.strip("/"))

    async def wait_frozen(self, rel: str, timeout: float) -> bool:
        return True

    def kill(self, rel: str) -> Strength:
        for _ in range(50):
            pids = self.pids(rel)
            if not pids:
                break
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        return Strength.DETECTED_AND_REAPED

    async def wait_empty(self, rel: str, timeout: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            for r in self._under(rel):
                try:
                    os.waitpid(self._roots[r], os.WNOHANG)
                except ChildProcessError:
                    pass
            if not self.pids(rel):
                return True
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(0.02)

    def stats(self, rel: str) -> dict[str, Any]:
        pids = self.pids(rel)
        return {
            "memory_current": sum(_rss(p, self.proc) for p in pids),
            "memory_events": {},
            "pids_current": len(pids),
            "cpu_usage_usec": None,
        }

    def remove(self, rel: str) -> None:
        for r in self._under(rel):
            self._roots.pop(r, None)
            self._limits.pop(r, None)

    async def _watch(self) -> None:
        while self._roots:
            for rel, limits in list(self._limits.items()):
                if rel not in self._roots or rel in self._frozen:
                    continue
                pids = self.pids(rel)
                over = None
                if limits.pids and len(pids) > limits.pids:
                    over = "pids"
                elif limits.memory_mb:
                    used = sum(_rss(p, self.proc) for p in pids)
                    if used > limits.memory_mb * 1024 * 1024:
                        over = "memory"
                if over:
                    logger.warning("%s exceeded its %s limit; reaping", rel, over)
                    self.kill(rel)
                    if self.on_violation is not None:
                        self.on_violation(rel, over)
            await asyncio.sleep(0.1)


class ProcessLauncher:
    """Starts each container's shim directly (the `landlock` and `none` tiers).

    Args:
        tree (`ProcessTree` or [`CgroupTree`]):
            Where container processes are tracked. With a writable cgroup root,
            pass a `CgroupTree` and limits are enforced by the kernel.
        python (`str`):
            Interpreter used to run the shim.
        log_dir (`Path`):
            Where each container's output goes.
    """

    namespaced = False

    def __init__(self, tree: Any, python: str, log_dir: Path) -> None:
        self.tree = tree
        self.python = python
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self.hidden: list[str] = []

    async def start(self, plan: Any, bundle_dir: Path) -> None:
        log = os.open(
            self.log_dir / f"{plan.name}.log",
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                self.python,
                "-m",
                "openenv.core.openenvd.shim",
                "--spec",
                plan.shim_spec_path,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": "/tmp",
                    "LANG": "C.UTF-8",
                },
            )
        finally:
            os.close(log)
        self._procs[plan.name] = proc
        rel = plan.cgroups_path.strip("/")
        if isinstance(self.tree, ProcessTree):
            self.tree.register(rel, proc.pid)
        else:
            try:
                self.tree.move(proc.pid, rel)
            except OSError:
                logger.warning("could not move %s into its cgroup", plan.name)

    async def stop(self, name: str) -> None:
        proc = self._procs.pop(name, None)
        if proc is None:
            return
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            await proc.wait()

    async def wait(self, name: str, timeout: float) -> bool:
        proc = self._procs.get(name)
        if proc is None:
            return True
        try:
            await asyncio.wait_for(proc.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
