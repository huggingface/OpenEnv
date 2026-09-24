# SPDX-License-Identifier: BSD-3-Clause

"""Probe actual OS boundaries; capability-dependent tests never run unisolated."""

import asyncio
import os
import signal
import socket
import sys

import pytest
from openenv.core.openenvd.isolation import detect_capabilities, spawn_task
from openenv.core.openenvd.models import TaskSpec

_CAPABILITIES = detect_capabilities()
requires_uids = pytest.mark.skipif(
    not _CAPABILITIES.can_allocate_uids,
    reason="requires Linux root with unmapped task UIDs available",
)
requires_netns = pytest.mark.skipif(
    not (_CAPABILITIES.can_allocate_uids and _CAPABILITIES.can_unshare_net),
    reason="requires Linux UID separation and network namespace capability",
)


async def run_probe(argv, *, uid=None, network_isolated=False):
    spec = TaskSpec(
        name="probe",
        argv=argv,
        uid=uid,
        gid=uid,
        network_isolated=network_isolated,
    )
    proc = await spawn_task(
        spec,
        _CAPABILITIES,
        env={"PATH": os.defpath},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return proc.returncode, out.decode()
    finally:
        if proc.returncode is None:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()


def connect_probe(port):
    return [
        sys.executable,
        "-c",
        "import socket; s=socket.socket(); s.settimeout(1);"
        f"s.connect(('127.0.0.1', {port})); print('CONNECTED')",
    ]


async def test_shared_network_control_can_reach_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        code, output = await run_probe(connect_probe(listener.getsockname()[1]))
    assert code == 0 and "CONNECTED" in output


@requires_netns
async def test_isolated_child_cannot_reach_daemon_network():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        code, output = await run_probe(
            connect_probe(listener.getsockname()[1]),
            uid=65534,
            network_isolated=True,
        )
    assert code != 0 and "CONNECTED" not in output


@requires_netns
async def test_isolated_child_has_only_loopback_and_cannot_regain_root():
    script = """
import os, socket
assert [name for _, name in socket.if_nameindex()] == ["lo"]
try:
    os.setuid(0)
except PermissionError:
    print("DENIED")
else:
    raise AssertionError("child regained root")
"""
    code, output = await run_probe(
        [sys.executable, "-c", script], uid=65534, network_isolated=True
    )
    assert code == 0 and "DENIED" in output


@requires_uids
async def test_workload_cannot_read_daemon_assets(tmp_path):
    path = tmp_path / "solution"
    path.write_text("SECRET_SOLUTION")
    path.chmod(0o600)
    control_code, control_output = await run_probe(["cat", str(path)])
    assert control_code == 0 and "SECRET_SOLUTION" in control_output
    code, output = await run_probe(["cat", str(path)], uid=65534)
    assert code != 0 and "SECRET_SOLUTION" not in output


@requires_uids
async def test_workload_cannot_signal_daemon():
    script = f"""
import os
try:
    os.kill({os.getpid()}, 0)
except PermissionError:
    print("DENIED")
else:
    raise AssertionError("daemon can be signaled")
"""
    code, output = await run_probe([sys.executable, "-c", script], uid=65534)
    assert code == 0 and "DENIED" in output


@requires_uids
async def test_separate_workload_identities_cannot_signal_each_other():
    victim = await spawn_task(
        TaskSpec(
            name="victim",
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
            uid=65534,
            gid=65534,
        ),
        _CAPABILITIES,
        env={"PATH": os.defpath},
    )
    try:
        os.kill(victim.pid, 0)
        script = f"""
import os
try:
    os.kill({victim.pid}, 0)
except PermissionError:
    print("DENIED")
else:
    raise AssertionError("sibling can be signaled")
"""
        code, output = await run_probe([sys.executable, "-c", script], uid=65533)
        assert code == 0 and "DENIED" in output
    finally:
        if victim.returncode is None:
            os.killpg(victim.pid, signal.SIGKILL)
        await victim.wait()
