# SPDX-License-Identifier: BSD-3-Clause
"""Standalone Linux network owner. Only standard-library imports are allowed.

Python configures and supervises kernel networking; it never forwards packets.
The private state file permits idempotent cleanup even after this process dies.
"""

from __future__ import annotations

import asyncio
import ctypes as C
import fcntl
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import struct
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# These exclusions take precedence over every workload allowance.
PROTECTED = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
)
POOL = ipaddress.IPv4Network("198.18.0.0/15")
TOOL_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"


def validate(state):
    if not re.fullmatch(r"oe[0-9a-f]{10}", state["name"]):
        raise ValueError("invalid network identity")
    for rule in state.get("allow", []):
        if str(ipaddress.IPv4Network(rule["cidr"])) != rule["cidr"]:
            raise ValueError("expected a canonical IPv4 CIDR")
        if rule["protocol"] not in ("tcp", "udp") or not rule["ports"]:
            raise ValueError("explicit protocol and ports are required")
        if any(type(p) is not int or not 1 <= p <= 65535 for p in rule["ports"]):
            raise ValueError("invalid egress port")
    if state.get("guest_ip"):
        guest = ipaddress.IPv4Address(state["guest_ip"])
        if guest not in POOL or int(guest) % 4 != 2:
            raise ValueError("invalid allocated guest address")


def ruleset(state, group=None):
    """Render an atomic, interface-scoped nft transaction from validated input."""
    validate(state)
    name, guest = state["name"], state["guest_ip"]

    def log(verdict):
        return (
            f'log prefix "{verdict}" group {group} snaplen 80 queue-threshold 1 '
            if group
            else ""
        )

    allow = []
    for rule in state.get("allow", []):
        ports = ", ".join(str(p) for p in rule["ports"])
        allow.append(
            f"ip daddr {rule['cidr']} {rule['protocol']} dport {{ {ports} }} "
            f"ct state new counter {log('allow')}accept"
        )
    return f'''table inet {name} {{
 chain raw {{
  type filter hook prerouting priority raw; policy accept;
  iifname "{name}" meta nfproto != ipv4 drop
  iifname "{name}" ip saddr != {guest} drop
 }}
 chain input {{
  type filter hook input priority -150; policy accept;
  iifname "{name}" jump denied
 }}
 chain forward {{
  type filter hook forward priority -150; policy accept;
  iifname "{name}" jump egress
  oifname "{name}" jump ingress
 }}
 chain egress {{
  ip daddr {{ {", ".join(PROTECTED)} }} jump denied
  ct state invalid drop
  ct state established accept
  {chr(10).join(allow)}
  jump denied
 }}
 chain ingress {{
  meta nfproto != ipv4 drop
  ip daddr != {guest} drop
  ct state established,related accept
  drop
 }}
 chain denied {{
  counter {log("deny")}
  meta l4proto tcp reject with tcp reset
  reject with icmpx type port-unreachable
 }}
 chain nat {{
  type nat hook postrouting priority srcnat; policy accept;
  iifname "{name}" ip saddr {guest} masquerade
 }}
}}
'''


def packet_event(packet, verdict):
    """Retain network headers and policy verdict only, never packet payloads."""
    if verdict not in ("allow", "deny"):
        raise ValueError("unexpected NFLOG verdict")
    if len(packet) < 20 or packet[0] >> 4 != 4:
        raise ValueError("expected IPv4 NFLOG packet")
    offset = (packet[0] & 15) * 4
    if offset < 20 or len(packet) < offset:
        raise ValueError("truncated IPv4 header")
    protocol = {6: "tcp", 17: "udp", 1: "icmp"}.get(packet[9], "other")
    dest = str(ipaddress.IPv4Address(packet[16:20]))
    # Non-initial fragments do not contain transport ports.
    if protocol in ("tcp", "udp") and not (
        struct.unpack_from("!H", packet, 6)[0] & 0x1FFF
    ):
        if len(packet) < offset + 4:
            raise ValueError("truncated transport header")
        dest += ":" + str(struct.unpack_from("!H", packet, offset + 2)[0])
    return {
        "kind": "packet",
        "protocol": protocol,
        "dest": dest,
        "denied": verdict == "deny",
        "outcome": verdict,
        "source": "nftables",
    }


class NFLog:
    """libnetfilter_log owns the wire protocol; Python retains bounded events."""

    def __init__(self):
        self.group = secrets.randbelow(65535) + 1
        self.sequence = None
        self.handle = None
        self.socket = None
        self.error = None
        self.events = []
        self.lib = C.CDLL("libnetfilter_log.so.1", use_errno=True)
        callback_type = C.CFUNCTYPE(
            C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p
        )
        signatures = {
            "open": (C.c_void_p, []),
            "close": (C.c_int, [C.c_void_p]),
            "fd": (C.c_int, [C.c_void_p]),
            "bind_group": (C.c_void_p, [C.c_void_p, C.c_uint16]),
            "set_mode": (C.c_int, [C.c_void_p, C.c_uint8, C.c_uint32]),
            "set_qthresh": (C.c_int, [C.c_void_p, C.c_uint32]),
            "set_flags": (C.c_int, [C.c_void_p, C.c_uint16]),
            "callback_register": (C.c_int, [C.c_void_p, callback_type, C.c_void_p]),
            "handle_packet": (C.c_int, [C.c_void_p, C.c_char_p, C.c_int]),
            "get_seq": (C.c_int, [C.c_void_p, C.POINTER(C.c_uint32)]),
            "get_prefix": (C.c_char_p, [C.c_void_p]),
            "get_payload": (C.c_int, [C.c_void_p, C.POINTER(C.c_void_p)]),
        }
        for name, (result, args) in signatures.items():
            function = getattr(self.lib, "nflog_" + name)
            function.restype, function.argtypes = result, args
        try:
            self.handle = self.lib.nflog_open()
            if not self.handle:
                raise OSError("could not open NFLOG")
            self.socket = socket.fromfd(
                self.lib.nflog_fd(self.handle), socket.AF_NETLINK, socket.SOCK_RAW
            )
            # The library does synchronous configuration. Bound its receive wait
            # without making the shared descriptor nonblocking until setup ends.
            self.socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVTIMEO, struct.pack("ll", 3, 0)
            )
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            group = self.lib.nflog_bind_group(self.handle, self.group)
            if not group:
                raise OSError("could not bind NFLOG group")
            self.callback = callback_type(self._receive)
            for status in (
                self.lib.nflog_set_mode(group, 2, 80),
                self.lib.nflog_set_qthresh(group, 1),
                self.lib.nflog_set_flags(group, 1),
                self.lib.nflog_callback_register(group, self.callback, None),
            ):
                if status < 0:
                    raise OSError("could not configure NFLOG")
            self.socket.setblocking(False)
        except BaseException:
            self.close()
            raise

    def _receive(self, group, message, data, context):
        # Exceptions must not escape a ctypes callback (ctypes would swallow
        # them). Decode raises them after the library returns, failing closed.
        try:
            sequence = C.c_uint32()
            if self.lib.nflog_get_seq(data, C.byref(sequence)) < 0:
                raise OSError("NFLOG sequence missing")
            if (
                self.sequence is not None
                and sequence.value != (self.sequence + 1) & 0xFFFFFFFF
            ):
                raise OSError("NFLOG sequence gap; observations incomplete")
            self.sequence = sequence.value
            payload = C.c_void_p()
            size = self.lib.nflog_get_payload(data, C.byref(payload))
            prefix = self.lib.nflog_get_prefix(data)
            if size < 0 or size > 80 or not payload.value or not prefix:
                raise ValueError("invalid NFLOG packet")
            self.events.append(
                packet_event(C.string_at(payload, size), prefix.decode("ascii"))
            )
            return 0
        except Exception as error:
            self.error = error
            return -1

    def decode(self, data):
        self.events = []
        self.error = None
        status = self.lib.nflog_handle_packet(self.handle, data, len(data))
        if self.error:
            raise self.error
        if status < 0:
            raise OSError("invalid NFLOG message")
        return self.events

    async def observe(self):
        loop = asyncio.get_running_loop()
        while True:
            # ENOBUFS is fatal; never enable NETLINK_NO_ENOBUFS.
            data = await loop.sock_recv(self.socket, 256 * 1024)
            if not data:
                raise OSError("NFLOG socket closed")
            for event in self.decode(data):
                emit(event)

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.handle:
            self.lib.nflog_close(self.handle)
            self.handle = None


async def command(*args, data=None, empty_conntrack=False):
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE
        if data is not None
        else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(data), 5)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise
    if proc.returncode and not (
        empty_conntrack
        and proc.returncode == 1
        and b"0 flow entries have been deleted" in stderr
    ):
        raise OSError(
            f"{Path(args[0]).name} failed: {stderr.decode(errors='replace').strip()}"
        )
    return stdout


def emit(event):
    data = json.dumps(event).encode() + b"\n"
    # Nonblocking, bounded observation transport. A slow/dead reader must
    # terminate the network owner rather than silently lose observations.
    if os.write(1, data) != len(data):
        raise OSError("network observation pipe overflow")


@asynccontextmanager
async def allocation_lock():
    with open("/run/openenvd-network.lock", "a") as lock:
        for _ in range(100):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.05)
        else:
            raise OSError("network allocation lock timed out")
        yield


def write_state(path, state):
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(state))
    os.replace(temporary, path)


class KernelNetwork:
    def __init__(self, state_path):
        self.path = Path(state_path)
        self.state = json.loads(self.path.read_text())
        validate(self.state)
        self.name = self.state["name"]
        self.namespace = self.name + "_ns"
        self.logger = None

    def tools(self):
        tools = {
            name: shutil.which(name, path=TOOL_PATH)
            for name in ("ip", "nft", "conntrack")
        }
        if not all(tools.values()):
            raise OSError("openenvd requires iproute2, nftables, and conntrack-tools")
        return tools

    async def start(self):
        tools = self.tools()
        ip, nft = tools["ip"], tools["nft"]
        if Path("/proc/sys/net/ipv4/ip_forward").read_text().strip() != "1":
            raise OSError(
                "enable net.ipv4.ip_forward in the daemon's network namespace"
            )
        # Serialize address allocation across daemons sharing this namespace.
        async with allocation_lock():
            links = json.loads(await command(ip, "-j", "link", "show"))
            namespaces = json.loads(await command(ip, "-j", "netns", "list") or b"[]")
            tables = json.loads(await command(nft, "-j", "list", "tables"))["nftables"]
            if (
                any(x["ifname"] in (self.name, self.name + "p") for x in links)
                or any(x["name"] == self.namespace for x in namespaces)
                or any(x.get("table", {}).get("name") == self.name for x in tables)
            ):
                raise OSError("network identity already exists")
            routes = json.loads(
                await command(ip, "-j", "-4", "route", "show", "table", "all")
            )
            occupied = [
                ipaddress.IPv4Network(x["dst"], strict=False)
                for x in routes
                if x.get("dst", "default") != "default"
            ]
            # One atomic record is both the cleanup journal and reservation.
            # Keep the address reserved through disconnect AND conntrack cleanup.
            for path in self.path.parent.parent.glob("*/state.json"):
                try:
                    state = json.loads(path.read_text())
                except FileNotFoundError:
                    continue  # Its owner finished cleanup and removed the record.
                validate(state)
                if path != self.path and state["name"] == self.name:
                    raise OSError("network identity already exists")
                if state.get("owned"):
                    occupied.append(
                        ipaddress.IPv4Network(state["guest_ip"] + "/30", strict=False)
                    )
            for _ in range(128):
                subnet = ipaddress.IPv4Network(
                    (
                        int(POOL.network_address)
                        + secrets.randbelow(POOL.num_addresses // 4) * 4,
                        30,
                    )
                )
                if not any(subnet.overlaps(other) for other in occupied):
                    break
            else:
                raise OSError("no free episode network in 198.18.0.0/15")
            gateway, guest = str(subnet[1]), str(subnet[2])
            self.state.update(guest_ip=guest, owned=True)
            write_state(self.path, self.state)  # Journal before the first mutation.
            await command(ip, "netns", "add", self.namespace)
            await command(
                ip,
                "link",
                "add",
                self.name,
                "type",
                "veth",
                "peer",
                "name",
                self.name + "p",
            )
            await command(ip, "link", "set", self.name + "p", "netns", self.namespace)
            await command(
                ip, "-n", self.namespace, "link", "set", self.name + "p", "name", "eth0"
            )
            await command(ip, "addr", "add", gateway + "/30", "dev", self.name)
            await command(
                ip, "-n", self.namespace, "addr", "add", guest + "/30", "dev", "eth0"
            )
            if self.state.get("observe"):
                self.logger = NFLog()
            await command(
                nft,
                "-f",
                "-",
                data=ruleset(
                    self.state, self.logger.group if self.logger else None
                ).encode(),
            )
            # No packets can leave the guest until all policy is installed.
            await command(ip, "link", "set", self.name, "up")
            await command(ip, "-n", self.namespace, "link", "set", "lo", "up")
            await command(ip, "-n", self.namespace, "link", "set", "eth0", "up")
            await command(
                ip, "-n", self.namespace, "route", "add", "default", "via", gateway
            )
        emit({"kind": "ready", "namespace": "/run/netns/" + self.namespace})

    async def close(self):
        # A replacement cleanup process reloads the journal after a crash.
        self.state = json.loads(self.path.read_text())
        validate(self.state)
        if not self.state.get("owned"):
            return
        tools = self.tools()
        ip, nft, conntrack = tools["ip"], tools["nft"], tools["conntrack"]
        async with allocation_lock():
            links = json.loads(await command(ip, "-j", "link", "show"))
            for name in (self.name, self.name + "p"):
                if any(x["ifname"] == name for x in links):
                    await command(ip, "link", "delete", name)
                    break  # Deleting either end removes the pair.
            # Disconnect before removing enforcement; leave the rules installed if
            # link deletion fails. Never flush another episode's conntrack state.
            for direction in ("--orig-src", "--orig-dst"):
                await command(
                    conntrack,
                    "-D",
                    "-f",
                    "ipv4",
                    direction,
                    self.state["guest_ip"],
                    empty_conntrack=True,
                )
            tables = json.loads(await command(nft, "-j", "list", "tables"))["nftables"]
            if any(
                x.get("table", {}).get("name") == self.name
                and x["table"]["family"] == "inet"
                for x in tables
            ):
                await command(nft, "delete", "table", "inet", self.name)
            namespaces = json.loads(await command(ip, "-j", "netns", "list") or b"[]")
            if any(x["name"] == self.namespace for x in namespaces):
                await command(ip, "netns", "delete", self.namespace)
            if self.logger:
                self.logger.close()
                self.logger = None
            self.state["owned"] = False
            write_state(self.path, self.state)


async def run(state_path, cleanup=False):
    network = KernelNetwork(state_path)
    if cleanup:
        await network.close()
        return
    os.set_blocking(1, False)
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer
    )
    owner = asyncio.create_task(reader.read())
    setup = asyncio.create_task(network.start())
    observer = None
    try:
        done, _ = await asyncio.wait(
            (owner, setup), return_when=asyncio.FIRST_COMPLETED
        )
        if owner in done:
            return
        await setup
        if network.logger:
            observer = asyncio.create_task(network.logger.observe())
            done, _ = await asyncio.wait(
                (owner, observer), return_when=asyncio.FIRST_COMPLETED
            )
            if observer in done:
                await observer
        else:
            await owner
    finally:
        for task in (setup, owner, observer):
            if task:
                task.cancel()
        await asyncio.gather(
            *(t for t in (setup, owner, observer) if t), return_exceptions=True
        )
        transport.close()
        await network.close()


if __name__ == "__main__":
    try:
        asyncio.run(run(sys.argv[1], cleanup=sys.argv[2:] == ["--cleanup"]))
    except Exception as error:
        print(f"openenvd network: {error}", file=sys.stderr)
        sys.exit(1)
