# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import os
import shutil
import socket
import stat
import sys
import tempfile

import pytest
from openenv.core.openenvd.peercred import (
    cgroup_of_pid,
    peer_pid,
    serve_guarded_unix,
    within,
)

pytestmark = pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "darwin"),
    reason="peer credentials need Linux or macOS",
)


@pytest.fixture
def sockdir():
    d = tempfile.mkdtemp(prefix="oe", dir="/tmp")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_peer_pid_of_a_socketpair_is_this_process():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with a, b:
        assert peer_pid(a) == os.getpid()


def test_peer_pid_is_none_for_a_socket_without_a_peer():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        assert peer_pid(s) is None


@pytest.mark.parametrize(
    "cgroup,prefix,expected",
    [
        ("/unit/agent", "/unit/agent", True),
        ("/unit/agent/env", "/unit/agent", True),
        ("/unit/agent/env", "/unit/agent/", True),
        ("/unit/agentx", "/unit/agent", False),
        ("/unit", "/unit/agent", False),
        ("/unit/agent/../services", "/unit/agent", False),
        ("/", "/unit/agent", False),
    ],
)
def test_within_matches_whole_path_components(cgroup, prefix, expected):
    assert within(cgroup, prefix) is expected


def test_cgroup_of_pid_reads_the_unified_hierarchy(tmp_path):
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "cgroup").write_text("1:name=systemd:/old\n0::/unit/agent/env\n")
    assert cgroup_of_pid(42, proc=str(tmp_path)) == "/unit/agent/env"
    assert cgroup_of_pid(43, proc=str(tmp_path)) is None


async def _internal_echo(path):
    async def handle(reader, writer):
        data = await reader.read()
        writer.write(b"echo:" + data)
        await writer.drain()
        writer.close()

    return await asyncio.start_unix_server(handle, path=path)


async def _roundtrip(path, payload=b"ping"):
    reader, writer = await asyncio.open_unix_connection(path)
    writer.write(payload)
    writer.write_eof()
    data = await asyncio.wait_for(reader.read(), 5)
    writer.close()
    return data


@pytest.mark.parametrize(
    "prefix,cgroup",
    [("", None), ("/unit/agent", "/unit/agent/env")],
)
async def test_guard_splices_callers_from_the_expected_cgroup(sockdir, prefix, cgroup):
    internal = os.path.join(sockdir, "internal.sock")
    public = os.path.join(sockdir, "public.sock")
    looked_up = []

    def lookup(pid):
        looked_up.append(pid)
        return cgroup

    inner = await _internal_echo(internal)
    guard = await serve_guarded_unix(public, internal, prefix, cgroup_of_pid=lookup)
    try:
        assert await _roundtrip(public) == b"echo:ping"
        assert stat.S_IMODE(os.stat(public).st_mode) == 0o666
    finally:
        guard.close()
        inner.close()
    assert looked_up == ([os.getpid()] if prefix else [])


@pytest.mark.parametrize("cgroup", ["/unit/agentx", "/unit/services/db", None])
async def test_guard_rejects_callers_outside_the_cgroup(sockdir, cgroup):
    internal = os.path.join(sockdir, "internal.sock")
    public = os.path.join(sockdir, "public.sock")
    rejected = []
    inner = await _internal_echo(internal)
    guard = await serve_guarded_unix(
        public,
        internal,
        "/unit/agent",
        cgroup_of_pid=lambda pid: cgroup,
        on_reject=lambda pid, cg: rejected.append((pid, cg)),
    )
    try:
        assert await _roundtrip(public) == b""
    finally:
        guard.close()
        inner.close()
    assert rejected == [(os.getpid(), cgroup)]
