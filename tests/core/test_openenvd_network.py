# SPDX-License-Identifier: BSD-3-Clause
"""Packet mediation configuration, supervision, and real namespace setup."""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from openenv.core.openenvd.isolation import (
    detect_capabilities,
    IsolationError,
    spawn_task,
)
from openenv.core.openenvd.models import TaskSpec
from openenv.core.openenvd.network import Network
from openenv.core.openenvd.observation import Collector
from openenv.core.openenvd.policy import EgressPolicy, OpenEnvDConfig
from openenv.core.openenvd.runtime import Runtime
from pydantic import ValidationError


@pytest.mark.parametrize(
    "rule",
    [
        {"cidr": "::/0", "protocol": "tcp", "ports": [443]},
        {"cidr": "1.2.3.4/24", "protocol": "tcp", "ports": [443]},
        {"cidr": "0.0.0.0/0", "protocol": "icmp", "ports": [443]},
        {"cidr": "0.0.0.0/0", "protocol": "tcp", "ports": []},
        {"cidr": "0.0.0.0/0", "protocol": "tcp", "ports": [0]},
        {"cidr": "0.0.0.0/0", "protocol": "tcp", "ports": [65536]},
    ],
)
def test_reject_invalid_egress_policy(rule):
    with pytest.raises(ValidationError):
        OpenEnvDConfig(network={"allow": [rule]})


def test_policy_roundtrip_and_default_deny():
    assert not OpenEnvDConfig().network.allow
    config = OpenEnvDConfig(
        network={
            "allow": [
                {"cidr": "1.1.1.1/32", "protocol": "udp", "ports": [53]},
            ]
        }
    )
    assert OpenEnvDConfig.model_validate_json(config.model_dump_json()) == config


def helper_script(tmp_path, monkeypatch, body):
    monkeypatch.setattr(
        "openenv.core.openenvd.network.STATE_ROOT", tmp_path / "networks"
    )
    path = tmp_path / "helper"
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o700)
    monkeypatch.setattr(
        "openenv.core.openenvd.network._helper_command",
        lambda state, **kw: [sys.executable, str(path), str(state)],
    )
    return path


@pytest.mark.asyncio
async def test_network_startup_failure_and_cleanup(tmp_path, monkeypatch):
    helper_script(tmp_path, monkeypatch, "import sys\nsys.exit(1)\n")
    network = Network(EgressPolicy(), Collector())
    try:
        await network.start()
        with pytest.raises(IsolationError, match="stopped"):
            await network.wait_ready()
    finally:
        await network.close()
    assert network.process.returncode is not None
    assert network.directory is None


@pytest.mark.asyncio
async def test_network_observes_before_workload_and_stops_on_owner_eof(
    tmp_path, monkeypatch
):
    helper_script(
        tmp_path,
        monkeypatch,
        "import sys, json\n"
        "state=json.load(open(sys.argv[1]))\n"
        'print(json.dumps({"kind": "ready", "namespace": "/run/netns/"+state["name"]+"_ns"}), flush=True)\n'
        'print(json.dumps({"kind": "packet", "dest": "1.1.1.1:443", "denied": True}), flush=True)\n'
        "sys.stdin.read()\n",
    )
    collector = Collector()
    network = Network(EgressPolicy(), collector, observe=True)
    try:
        await network.start()
        await network.wait_ready()
        assert network.ready.is_set()
        await asyncio.wait_for(collector.changed.wait(), 1)
        assert collector.events[0].data == {
            "kind": "packet",
            "dest": "1.1.1.1:443",
            "denied": True,
        }
        network.check()
    finally:
        await network.close()
    assert network.process.returncode == 0
    assert not network.failed.is_set()


@pytest.mark.asyncio
async def test_malformed_helper_events_fail_closed(tmp_path, monkeypatch):
    helper_script(
        tmp_path,
        monkeypatch,
        'import sys\nprint("not json", flush=True)\nsys.stdin.read()\n',
    )
    network = Network(EgressPolicy(), Collector())
    try:
        await network.start()
        with pytest.raises(IsolationError):
            await network.wait_ready()
    finally:
        await network.close()


@pytest.mark.asyncio
async def test_stop_cleans_network_even_before_worker_spawn(tmp_path):
    runtime = Runtime(
        OpenEnvDConfig(enabled=True),
        "unused",
        "unused",
        tmp_path,
        uid=65530,
        gid=65530,
        asset_root=tmp_path,
    )
    network = AsyncMock()
    runtime.network = network
    await runtime._stop()
    network.close.assert_awaited_once()
    assert runtime.network is None


@pytest.mark.asyncio
async def test_linux_kernel_denies_egress_and_hides_control_descriptors():
    if sys.platform != "linux" or os.geteuid() != 0:
        pytest.skip("Linux root required")
    capabilities = detect_capabilities()
    if not capabilities.can_unshare_net:
        pytest.skip("network namespace capability required")
    collector = Collector()
    network = Network(EgressPolicy(), collector, observe=True)
    proc = None
    try:
        await network.start()
        await network.wait_ready()
        code = (
            "import json, socket, os\n"
            "s=socket.socket();s.settimeout(2)\n"
            "try:\n s.connect(('1.1.1.1',443));outcome='connected'\n"
            "except OSError as e:\n outcome=e.errno\n"
            "s.close()\n"
            "u=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);u.settimeout(2)\n"
            "try:\n u.connect(('1.1.1.1',53));u.send(b'dns');u.recv(32);udp='connected'\n"
            "except OSError as e:\n udp=e.errno\n"
            "u.close()\n"
            "print(json.dumps({'outcome':outcome,'udp':udp,'interfaces':socket.if_nameindex(),"
            "'fds':len(os.listdir('/proc/self/fd'))}),flush=True)\n"
        )
        proc = await spawn_task(
            TaskSpec(
                name="network-test",
                argv=[sys.executable, "-I", "-S", "-c", code],
                uid=65530,
                gid=65530,
                network_isolated=True,
            ),
            capabilities,
            env={},
            stdout=asyncio.subprocess.PIPE,
            network_namespace=network.namespace,
        )
        await network.wait_ready()
        stdout, _ = await asyncio.wait_for(proc.communicate(), 5)
        assert proc.returncode == 0
        result = json.loads(stdout)
        assert result["outcome"] == 111  # ECONNREFUSED from policy, not ENETUNREACH.
        assert result["udp"] == 111
        await asyncio.wait_for(collector.changed.wait(), 1)
        assert collector.events[0].data["denied"]
        assert {name for _, name in result["interfaces"]} == {"lo", "eth0"}
        assert result["fds"] == 4  # stdin/out/err plus listdir's own descriptor.
    finally:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        await network.close()


@pytest.mark.asyncio
async def test_observation_overflow_still_terminates_episode(tmp_path):
    runtime = Runtime(
        OpenEnvDConfig(enabled=True),
        "unused",
        "unused",
        tmp_path,
        uid=65530,
        gid=65530,
        asset_root=tmp_path,
    )
    runtime.collector = Collector(max_events=0)
    network = type("Network", (), {"failed": asyncio.Event()})()
    network.failed.set()
    runtime.network = network
    runtime._stop = AsyncMock()
    with pytest.raises(RuntimeError, match="capacity"):
        await runtime._watch_network()
    runtime._stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_linux_setup_failure_never_executes_workload(tmp_path):
    if sys.platform != "linux" or os.geteuid() != 0:
        pytest.skip("Linux root required")
    caps = detect_capabilities()
    if not caps.can_unshare_net:
        pytest.skip("network namespace capability required")
    marker = tmp_path / "executed"
    with pytest.raises(IsolationError, match="join configured network namespace"):
        await spawn_task(
            TaskSpec(
                name="never-run",
                argv=[sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
                network_isolated=True,
                uid=65530,
                gid=65530,
            ),
            caps,
            env={},
            network_namespace="/run/netns/does-not-exist",
        )
    assert not marker.exists()


@pytest.mark.asyncio
async def test_helper_failure_interrupts_inflight_worker_request(
    tmp_path, socket_directory
):
    runtime = Runtime(
        OpenEnvDConfig(enabled=True),
        "unused",
        "unused",
        tmp_path,
        uid=65530,
        gid=65530,
        asset_root=tmp_path,
    )
    runtime.directory = socket_directory
    runtime.network = Network(EgressPolicy(), runtime.collector)
    accepted = asyncio.Event()
    closed = asyncio.Event()

    async def stalled_worker(reader, writer):
        try:
            await reader.readline()
            accepted.set()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            closed.set()

    server = await asyncio.start_unix_server(
        stalled_worker, path=socket_directory / "worker.sock"
    )
    try:
        request = asyncio.create_task(runtime._request("step"))
        await asyncio.wait_for(accepted.wait(), 1)
        runtime.network.failed.set()
        with pytest.raises(IsolationError):
            await asyncio.wait_for(request, 1)
        await asyncio.wait_for(closed.wait(), 1)
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
def socket_directory():
    # macOS Unix sockets cannot use pytest's long per-test temporary paths.
    with tempfile.TemporaryDirectory(prefix="oe-net-", dir="/tmp") as directory:
        yield Path(directory)


def test_kernel_policy_protects_control_plane_before_allowances():
    from openenv.core.openenvd._network import ruleset

    rules = ruleset(
        {
            "name": "oe0123456789",
            "guest_ip": "198.18.0.2",
            "allow": [{"cidr": "0.0.0.0/0", "protocol": "tcp", "ports": [443]}],
        },
        17,
    )
    assert (
        rules.index("169.254.0.0/16")
        < rules.index("ct state established accept")
        < rules.index("ip daddr 0.0.0.0/0")
    )
    assert 'iifname "oe0123456789" jump denied' in rules
    assert "ip saddr != 198.18.0.2 drop" in rules
    assert "meta nfproto != ipv4 drop" in rules
    assert 'log prefix "allow" group 17' in rules


def test_nflog_headers_sequence_and_payload_privacy():
    if sys.platform != "linux" or os.geteuid() != 0:
        pytest.skip("Linux root for libnetfilter_log")
    import struct

    from openenv.core.openenvd._network import NFLog

    def message(sequence):
        packet = bytearray(24)
        packet[0], packet[9] = 0x45, 6
        packet[16:20] = bytes([1, 1, 1, 1])
        packet[22:24] = struct.pack("!H", 443)
        packet += b"private payload"
        attrs = b""
        for kind, value in [
            (9, bytes(packet)),
            (10, b"deny\0"),
            (12, struct.pack("!I", sequence)),
        ]:
            item = struct.pack("=HH", len(value) + 4, kind) + value
            attrs += item + b"\0" * (-len(item) % 4)
        body = struct.pack("!BBH", 2, 0, logger.group) + attrs
        return struct.pack("=IHHII", len(body) + 16, 0x400, 0, 0, 0) + body

    logger = NFLog()
    try:
        assert list(logger.decode(message(0)))[0] == {
            "kind": "packet",
            "protocol": "tcp",
            "dest": "1.1.1.1:443",
            "denied": True,
            "outcome": "deny",
            "source": "nftables",
        }
        with pytest.raises(OSError, match="sequence gap"):
            list(logger.decode(message(2)))
    finally:
        logger.close()


@pytest.mark.asyncio
async def test_linux_allowed_egress_control_plane_and_crash_cleanup():
    if (
        sys.platform != "linux"
        or os.geteuid() != 0
        or not detect_capabilities().can_unshare_net
    ):
        pytest.skip("Linux root with network namespace capability required")
    import secrets

    from openenv.core.openenvd._network import command

    # A synthetic public destination in another namespace: no Internet needed.
    name = "up" + secrets.token_hex(4)
    server = None
    network = None
    other = None
    try:
        await command("ip", "netns", "add", name)
        await command(
            "ip", "link", "add", name, "type", "veth", "peer", "name", name + "p"
        )
        await command("ip", "link", "set", name + "p", "netns", name)
        await command("ip", "addr", "add", "10.250.0.1/30", "dev", name)
        await command("ip", "link", "set", name, "up")
        await command(
            "ip", "-n", name, "addr", "add", "10.250.0.2/30", "dev", name + "p"
        )
        await command("ip", "-n", name, "link", "set", name + "p", "up")
        await command("ip", "-n", name, "link", "set", "lo", "up")
        await command("ip", "-n", name, "addr", "add", "93.184.216.34/32", "dev", "lo")
        await command("ip", "route", "add", "93.184.216.34/32", "via", "10.250.0.2")
        server_code = """import socket, threading, json, time
s=socket.socket(); s.bind(('93.184.216.34',0)); s.listen()
u=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); u.bind(('93.184.216.34',0))
def tcp():
 while True:
  c,_=s.accept(); c.sendall(c.recv(32)); c.close()
def udp():
 while True:
  data,addr=u.recvfrom(32); u.sendto(data,addr)
threading.Thread(target=tcp,daemon=True).start()
threading.Thread(target=udp,daemon=True).start()
print(json.dumps([s.getsockname()[1],u.getsockname()[1]]),flush=True)
time.sleep(60)
"""
        server = await asyncio.create_subprocess_exec(
            "ip",
            "netns",
            "exec",
            name,
            sys.executable,
            "-I",
            "-S",
            "-c",
            server_code,
            stdout=asyncio.subprocess.PIPE,
        )
        tcp, udp = json.loads(await asyncio.wait_for(server.stdout.readline(), 5))
        collector = Collector()
        network = Network(
            EgressPolicy(
                allow=[
                    {"cidr": "0.0.0.0/0", "protocol": protocol, "ports": [port]}
                    for protocol, port in [("tcp", tcp), ("udp", udp)]
                ]
            ),
            collector,
            observe=True,
        )
        other = Network(EgressPolicy(), Collector())
        for item in (network, other):
            await item.start()
            await item.wait_ready()
        code = f"""import socket,json
results=[]
for typ,port in [(socket.SOCK_STREAM,{tcp}),(socket.SOCK_DGRAM,{udp})]:
 s=socket.socket(socket.AF_INET,typ);s.settimeout(2);s.connect(('93.184.216.34',port));s.sendall(b'echo');results.append(s.recv(32).decode());s.close()
for host,port in [('10.250.0.1',{tcp}),('169.254.169.254',{tcp}),('93.184.216.34',1)]:
 s=socket.socket();s.settimeout(2)
 try: s.connect((host,port));results.append('unexpected')
 except OSError as e: results.append(e.errno)
 s.close()
print(json.dumps(results))
"""
        proc = await spawn_task(
            TaskSpec(
                name="egress",
                argv=[sys.executable, "-I", "-S", "-c", code],
                uid=65530,
                gid=65530,
                network_isolated=True,
            ),
            detect_capabilities(),
            env={},
            stdout=asyncio.subprocess.PIPE,
            network_namespace=network.namespace,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), 10)
        assert proc.returncode == 0
        assert json.loads(stdout) == ["echo", "echo", 111, 111, 111]
        # Wait for both allowed packet observations without assuming scheduling.
        for _ in range(100):
            if len(collector.events) >= 5:
                break
            await asyncio.sleep(0.01)
        events = [e.data for e in collector.events]
        assert {e["protocol"] for e in events if e["outcome"] == "allow"} == {
            "tcp",
            "udp",
        }
        assert len([e for e in events if e["denied"]]) >= 3
        network.process.kill()
        await network.process.wait()
        await network.close()  # Recover resources from the private journal.
        assert not Path(network.namespace).exists()
        assert Path(other.namespace).exists()
        tables = json.loads(await command("nft", "-j", "list", "tables"))["nftables"]
        assert not any(t.get("table", {}).get("name") == network.name for t in tables)
        links = json.loads(await command("ip", "-j", "link", "show"))
        assert not any(link["ifname"] == network.name for link in links)
        other.check()
    finally:
        for item in (network, other):
            if item:
                await item.close()
        if server and server.returncode is None:
            server.kill()
            await server.wait()
        await command("ip", "netns", "delete", name)


@pytest.mark.asyncio
async def test_cleanup_preserves_policy_and_reservation_until_disconnected(
    tmp_path, monkeypatch
):
    from openenv.core.openenvd import _network

    name = "oe0123456789"
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"name": name, "guest_ip": "198.18.0.2", "owned": True})
    )
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lock():
        yield

    monkeypatch.setattr(_network, "allocation_lock", lock)
    monkeypatch.setattr(
        _network.KernelNetwork,
        "tools",
        lambda self: {t: t for t in ("ip", "nft", "conntrack")},
    )
    calls = []
    fail = True

    async def command(*args, **kwargs):
        calls.append(args)
        if args == ("ip", "-j", "link", "show"):
            return json.dumps([{"ifname": name}]).encode()
        if args == ("ip", "link", "delete", name) and fail:
            raise OSError("cannot disconnect")
        if args == ("nft", "-j", "list", "tables"):
            return json.dumps(
                {
                    "nftables": [
                        {"table": {"name": name, "family": "inet"}},
                        {"table": {"name": "unrelated", "family": "inet"}},
                    ]
                }
            ).encode()
        if args == ("ip", "-j", "netns", "list"):
            return json.dumps([{"name": name + "_ns"}, {"name": "unrelated"}]).encode()
        return b""

    monkeypatch.setattr(_network, "command", command)
    network = _network.KernelNetwork(state)
    with pytest.raises(OSError, match="cannot disconnect"):
        await network.close()
    assert json.loads(state.read_text())["owned"]
    assert not any(args[0] in ("nft", "conntrack") for args in calls)
    fail = False
    calls.clear()
    await network.close()
    assert not json.loads(state.read_text())["owned"]
    assert ("conntrack", "-D", "-f", "ipv4", "--orig-src", "198.18.0.2") in calls
    assert ("conntrack", "-D", "-f", "ipv4", "--orig-dst", "198.18.0.2") in calls
    assert not any("unrelated" in args for args in calls)


@pytest.mark.asyncio
async def test_allocation_reserves_journal_address_until_cleanup_finishes(
    tmp_path, monkeypatch
):
    from contextlib import asynccontextmanager

    from openenv.core.openenvd import _network

    old = tmp_path / "old" / "state.json"
    new = tmp_path / "new" / "state.json"
    for path in (old, new):
        path.parent.mkdir()
    # Simulate a crash after deleting the old veth, before conntrack cleanup.
    _network.write_state(
        old, {"name": "oe0000000001", "guest_ip": "198.18.0.2", "owned": True}
    )
    _network.write_state(new, {"name": "oe0000000002", "owned": False})

    @asynccontextmanager
    async def lock():
        yield

    real_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            "1"
            if str(path) == "/proc/sys/net/ipv4/ip_forward"
            else real_read(path, *a, **kw)
        ),
    )
    monkeypatch.setattr(_network, "allocation_lock", lock)
    monkeypatch.setattr(
        _network.KernelNetwork,
        "tools",
        lambda self: {t: t for t in ("ip", "nft", "conntrack")},
    )
    candidates = iter([0, 1])
    monkeypatch.setattr(_network.secrets, "randbelow", lambda _: next(candidates))
    monkeypatch.setattr(_network, "emit", lambda _: None)

    async def command(*args, **kwargs):
        if args == ("nft", "-j", "list", "tables"):
            return b'{"nftables": []}'
        return b"[]"

    monkeypatch.setattr(_network, "command", command)
    network = _network.KernelNetwork(new)
    await network.start()
    assert json.loads(new.read_text())["guest_ip"] == "198.18.0.6"
    assert json.loads(old.read_text())["owned"]
    await network.close()
    assert not json.loads(new.read_text())["owned"]
    assert json.loads(old.read_text())["owned"]


def test_failed_atomic_journal_update_preserves_the_reservation(tmp_path, monkeypatch):
    from openenv.core.openenvd import _network

    path = tmp_path / "state.json"
    original = {"name": "oe0000000001", "owned": True, "guest_ip": "198.18.0.2"}
    _network.write_state(path, original)

    def fail(*args):
        raise OSError("publish failed")

    monkeypatch.setattr(_network.os, "replace", fail)
    with pytest.raises(OSError, match="publish failed"):
        _network.write_state(path, {**original, "owned": False})
    assert json.loads(path.read_text()) == original
    assert list(tmp_path.glob("*.json")) == [path]
