# SPDX-License-Identifier: BSD-3-Clause

"""Who is on the other end of a Unix socket.

Inside the unit a caller is identified twice: by which socket it can reach (the
bind mounts openenvd gives its container) and by the cgroup its process is in.
[`serve_guarded_unix`] checks the second before splicing a connection through
to an internal listener.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import struct
import sys
from typing import Callable

_SOL_LOCAL = 0  # macOS
_LOCAL_PEERPID = 0x002  # macOS
_UCRED = struct.Struct("3i")  # Linux struct ucred: pid, uid, gid


def peer_pid(sock) -> int | None:
    """The PID of the process that connected `sock`, or `None` if unknown.

    Args:
        sock (`socket.socket`):
            A connected `AF_UNIX` socket (asyncio's transport sockets work too).

    Returns:
        `int` or `None`: The peer's PID, from `SO_PEERCRED` on Linux or
        `LOCAL_PEERPID` on macOS.
    """
    try:
        if hasattr(socket, "SO_PEERCRED"):
            raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED.size)
            pid = _UCRED.unpack(raw)[0]
        elif sys.platform == "darwin":
            pid = sock.getsockopt(_SOL_LOCAL, _LOCAL_PEERPID)
        else:
            return None
    except OSError:
        return None
    return pid if pid > 0 else None


def cgroup_of_pid(pid: int, proc: str = "/proc") -> str | None:
    """The cgroup v2 path of `pid`, from `/proc/<pid>/cgroup`.

    Args:
        pid (`int`):
            The process.
        proc (`str`, *optional*, defaults to `"/proc"`):
            Where procfs is mounted.

    Returns:
        `str` or `None`: For example `/unit/agent/env`; `None` if unreadable.
    """
    try:
        with open(os.path.join(proc, str(pid), "cgroup")) as f:
            for line in f:
                if line.startswith("0::"):
                    return line[3:].strip() or "/"
    except OSError:
        return None
    return None


def within(cgroup: str, prefix: str) -> bool:
    """Whether `cgroup` is `prefix` or below it, by whole path components.

    Args:
        cgroup (`str`):
            A cgroup path.
        prefix (`str`):
            The expected ancestor.

    Returns:
        `bool`: `/unit/agent/env` is within `/unit/agent`; `/unit/agentx` isn't.
    """
    c = [p for p in cgroup.split("/") if p]
    want = [p for p in prefix.split("/") if p]
    if ".." in c or ".." in want:
        return False
    return c[: len(want)] == want


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError):
        writer.close()


async def serve_guarded_unix(
    public_path: str,
    internal_path: str,
    expected_cgroup_prefix: str,
    *,
    cgroup_of_pid: Callable[[int], str | None] = cgroup_of_pid,
    on_reject: Callable[[int | None, str | None], None] | None = None,
) -> asyncio.Server:
    """Listen on `public_path` and splice callers from the expected cgroup through.

    A connection whose peer PID is unknown, whose cgroup can't be read, or whose
    cgroup is outside `expected_cgroup_prefix` is closed at once and reported to
    `on_reject`. An empty prefix turns the check off (tiers without cgroups,
    and tests).

    Args:
        public_path (`str`):
            Socket to listen on. A stale socket there is replaced; mode `0o666`.
        internal_path (`str`):
            The listener accepted connections are spliced to.
        expected_cgroup_prefix (`str`):
            The cgroup callers must be in, matched by whole path components.
        cgroup_of_pid (`Callable[[int], str | None]`, *optional*):
            Looks up a PID's cgroup; defaults to reading `/proc/<pid>/cgroup`.
        on_reject (`Callable[[int | None, str | None], None]`, *optional*):
            Called with the peer's PID and cgroup for every rejected connection.

    Returns:
        `asyncio.Server`: The listening server.
    """
    lookup = cgroup_of_pid

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        if expected_cgroup_prefix:
            pid = peer_pid(writer.get_extra_info("socket"))
            cgroup = lookup(pid) if pid is not None else None
            if cgroup is None or not within(cgroup, expected_cgroup_prefix):
                if on_reject is not None:
                    on_reject(pid, cgroup)
                writer.close()
                with contextlib.suppress(ConnectionError, OSError):
                    await writer.wait_closed()
                return
        try:
            up_reader, up_writer = await asyncio.open_unix_connection(internal_path)
        except OSError:
            writer.close()
            return
        try:
            await asyncio.gather(_pump(reader, up_writer), _pump(up_reader, writer))
        finally:
            for w in (writer, up_writer):
                w.close()
                with contextlib.suppress(ConnectionError, OSError):
                    await w.wait_closed()

    with contextlib.suppress(FileNotFoundError):
        os.unlink(public_path)
    server = await asyncio.start_unix_server(handle, path=public_path)
    os.chmod(public_path, 0o666)
    return server
