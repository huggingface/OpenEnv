# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest
from openenv.core.openenvd.landlock import LandlockSpec
from openenv.core.openenvd.shim import shim_spec, socket_filter_program

SRC = str(Path(__file__).resolve().parents[3] / "src")


@pytest.fixture
def sock_dir():
    path = tempfile.mkdtemp(prefix="oe-", dir="/tmp" if os.path.isdir("/tmp") else None)
    yield path
    shutil.rmtree(path, ignore_errors=True)


class _Echo(socketserver.BaseRequestHandler):
    def handle(self):
        data = self.request.recv(1024)
        self.request.sendall(b"echo:" + data)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_shim(spec, tmp_path, timeout=20):
    path = tmp_path / "shim.json"
    path.write_text(json.dumps(spec))
    env = {**os.environ, "PYTHONPATH": SRC}
    return subprocess.run(
        [sys.executable, "-m", "openenv.core.openenvd.shim", "--spec", str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_principal_runs_with_forward_and_exit_code(tmp_path, sock_dir):
    unix_path = os.path.join(sock_dir, "agent.sock")
    server = socketserver.UnixStreamServer(unix_path, _Echo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = _free_port()
    principal = (
        "import socket\n"
        f"s = socket.create_connection(('127.0.0.1', {port}))\n"
        "s.sendall(b'ping')\n"
        "print(s.recv(1024).decode())\n"
        "print('hi')\n"
        "raise SystemExit(3)\n"
    )
    spec = shim_spec(
        argv=[sys.executable, "-c", principal],
        env={"PATH": os.environ.get("PATH", "")},
        landlock=LandlockSpec(read_write=["/tmp"]),
        forwards=[
            {
                "kind": "tcp_to_unix",
                "host": "127.0.0.1",
                "port": port,
                "path": unix_path,
            }
        ],
        principal_filter={},
        workdir=str(tmp_path),
    )
    try:
        result = _run_shim(spec, tmp_path)
    finally:
        server.shutdown()
        server.server_close()
    assert result.returncode == 3, result.stderr
    assert result.stdout.splitlines() == ["echo:ping", "hi"]


def test_signaled_principal_exits_128_plus_signal(tmp_path):
    spec = shim_spec(
        argv=[sys.executable, "-c", "import os, signal; os.kill(os.getpid(), 9)"],
        env={},
        landlock=None,
        forwards=[],
        principal_filter={},
        workdir=None,
    )
    assert _run_shim(spec, tmp_path).returncode == 128 + 9


def test_missing_command_exits_127(tmp_path):
    spec = shim_spec(["/nonexistent/bin"], {}, None, [], {}, None)
    result = _run_shim(spec, tmp_path)
    assert result.returncode == 127
    assert "failed to start the principal" in result.stderr


def test_shim_spec_validates_forwards():
    with pytest.raises(ValueError):
        shim_spec(["x"], {}, None, [{"kind": "udp"}], {}, None)
    with pytest.raises(ValueError):
        shim_spec(["x"], {}, None, [{"kind": "unix_to_tcp", "path": "/p"}], {}, None)
    with pytest.raises(ValueError):
        shim_spec([], {}, None, [], {}, None)
    spec = shim_spec(["x"], {"A": "1"}, LandlockSpec(read_only=["/usr"]), [], {}, "/w")
    assert json.loads(json.dumps(spec))["landlock"]["read_only"] == ["/usr"]


def _run_bpf(program, arch, nr, arg0):
    """A tiny classic-BPF interpreter for the instructions the shim emits."""
    import struct

    data = struct.pack("=iIQ6Q", nr, arch, 0, arg0, 0, 0, 0, 0, 0)
    ins = [
        struct.unpack("=HBBI", program[i : i + 8]) for i in range(0, len(program), 8)
    ]
    pc, acc = 0, 0
    while True:
        code, jt, jf, k = ins[pc]
        if code == 0x20:
            acc = struct.unpack_from("=I", data, k)[0]
            pc += 1
        elif code == 0x15:
            pc += 1 + (jt if acc == k else jf)
        elif code == 0x06:
            return k
        else:
            raise AssertionError(f"unexpected opcode {code:#x}")


@pytest.mark.parametrize(
    "machine,arch,nrs",
    [("x86_64", 0xC000003E, (41, 41 | 0x40000000)), ("aarch64", 0xC00000B7, (198,))],
)
def test_socket_filter_program(machine, arch, nrs):
    prog = socket_filter_program([2, 10], machine)
    allow, eperm = 0x7FFF0000, 0x00050001
    for nr in nrs:
        assert _run_bpf(prog, arch, nr, 2) == eperm
        assert _run_bpf(prog, arch, nr, 10) == eperm
        assert _run_bpf(prog, arch, nr, 1) == allow  # AF_UNIX
    assert _run_bpf(prog, arch, 0, 2) == allow  # another syscall
    assert _run_bpf(prog, 0x40000003, nrs[0], 1) == eperm  # foreign arch


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux seccomp")
def test_strict_principal_cannot_open_inet_socket(tmp_path):
    principal = (
        "import socket\n"
        "socket.socket(socket.AF_UNIX).close()\n"
        "try:\n"
        "    socket.socket(socket.AF_INET)\n"
        "except PermissionError:\n"
        "    raise SystemExit(7)\n"
    )
    spec = shim_spec(
        [sys.executable, "-c", principal],
        {},
        None,
        [],
        {"deny_socket_families": [2, 10]},
        None,
    )
    assert _run_shim(spec, tmp_path).returncode == 7
