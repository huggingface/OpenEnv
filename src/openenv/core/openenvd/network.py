# SPDX-License-Identifier: BSD-3-Clause
"""Own one episode's kernel network and its Python lifecycle supervisor."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import stat
import sys
import tempfile
from pathlib import Path

from ._network import write_state
from .isolation import IsolationError
from .policy import ObservationEventType


STATE_ROOT = Path("/run/openenvd-networks")


def _helper_command(state, *, cleanup=False):
    return [
        sys.executable,
        "-I",
        "-S",
        str(Path(__file__).with_name("_network.py")),
        str(state),
        *(["--cleanup"] if cleanup else []),
    ]


class Network:
    """Create a private namespace, supervise policy observation, and clean up."""

    def __init__(self, policy, collector, *, observe=False):
        self.policy = policy
        self.collector = collector
        self.observe = observe
        self.name = "oe" + secrets.token_hex(5)
        self.namespace = "/run/netns/" + self.name + "_ns"
        self.directory = None
        self.process = None
        self.reader = None
        self.ready = asyncio.Event()
        self.failed = asyncio.Event()
        self.closing = False

    async def start(self):
        STATE_ROOT.mkdir(mode=0o700, exist_ok=True)
        info = STATE_ROOT.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise IsolationError(
                "network state directory must be daemon-owned and private"
            )
        self.directory = Path(tempfile.mkdtemp(prefix="episode-", dir=STATE_ROOT))
        self.state = self.directory / "state.json"
        write_state(
            self.state,
            {
                "name": self.name,
                "allow": [rule.model_dump() for rule in self.policy.allow],
                "observe": self.observe,
                "owned": False,
            },
        )
        try:
            self.process = await asyncio.create_subprocess_exec(
                *_helper_command(self.state),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,
                env={},
                cwd="/",
                start_new_session=True,
            )
            self.reader = asyncio.create_task(self._read())
        except BaseException:
            await self.close()
            raise

    async def _read(self):
        try:
            while line := await self.process.stdout.readline():
                event = json.loads(line)
                if event.get("kind") == "ready":
                    if self.ready.is_set() or event.get("namespace") != self.namespace:
                        raise ValueError("invalid network readiness")
                    self.ready.set()
                elif event.get("kind") == "packet" and self.ready.is_set():
                    if self.observe:
                        self.collector.record(ObservationEventType.NETWORK, event)
                else:
                    raise ValueError("invalid network event")
        except Exception:
            pass
        finally:
            if not self.closing:
                self.failed.set()

    async def wait_ready(self):
        async def wait():
            ready = asyncio.create_task(self.ready.wait())
            failed = asyncio.create_task(self.failed.wait())
            try:
                await asyncio.wait((ready, failed), return_when=asyncio.FIRST_COMPLETED)
                self.check()
            finally:
                ready.cancel()
                failed.cancel()
                await asyncio.gather(ready, failed, return_exceptions=True)

        try:
            await asyncio.wait_for(wait(), 15)
        except asyncio.TimeoutError as error:
            raise IsolationError("kernel network setup timed out") from error

    def check(self):
        if self.failed.is_set() or (
            self.process and self.process.returncode is not None
        ):
            raise IsolationError(
                "network supervisor stopped; episode must be terminated"
            )

    async def close(self):
        self.closing = True
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
            self.reader = None
        if self.directory:
            if json.loads(self.state.read_text()).get("owned"):
                # SIGKILL cannot run finally blocks. The private journal lets a
                # fresh interpreter finish cleanup before reset can proceed.
                cleanup = await asyncio.create_subprocess_exec(
                    *_helper_command(self.state, cleanup=True),
                    env={},
                    cwd="/",
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                )
                try:
                    await asyncio.wait_for(cleanup.wait(), 30)
                except BaseException:
                    if cleanup.returncode is None:
                        cleanup.kill()
                    await cleanup.wait()
                    raise
                if cleanup.returncode:
                    raise IsolationError("network cleanup incomplete; reset refused")
            shutil.rmtree(self.directory)
            self.directory = None
