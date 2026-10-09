# SPDX-License-Identifier: BSD-3-Clause

"""The container shim: PID 1 of every openenvd container.

Run as `python -m openenv.core.openenvd.shim --spec <path>`. It starts the
socket forwarders, then `exec`s the principal in a forked child after applying
Landlock and the principal's socket-family filter there, so the shim keeps its
own sockets. It reaps every child, forwards `SIGTERM`/`SIGINT` to the
principal's process group and exits with the principal's status.

Stdlib only (plus `landlock` and `forwarder`), since it runs inside minimal
containers.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import fcntl
import json
import os
import platform
import signal
import socket
import struct
import sys
import threading
from typing import Any

from . import forwarder
from .landlock import LandlockSpec, restrict_self

PR_SET_SECCOMP = 22
PR_SET_CHILD_SUBREAPER = 36
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2

BPF_LD_W_ABS = 0x20
BPF_JMP_JEQ_K = 0x15
BPF_RET_K = 0x06
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
EPERM = 1

_X32_SYSCALL_BIT = 0x40000000
# machine -> (AUDIT_ARCH_*, socket syscall numbers)
_SOCKET_SYSCALLS = {
    "x86_64": (0xC000003E, (41, 41 | _X32_SYSCALL_BIT)),
    "amd64": (0xC000003E, (41, 41 | _X32_SYSCALL_BIT)),
    "aarch64": (0xC00000B7, (198,)),
    "arm64": (0xC00000B7, (198,)),
}

SIOCGIFFLAGS = 0x8913
SIOCSIFFLAGS = 0x8914
IFF_UP = 0x1


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _warn(message: str) -> None:
    print(f"openenvd-shim: {message}", file=sys.stderr, flush=True)


def shim_spec(
    argv: list[str],
    env: dict[str, str],
    landlock: LandlockSpec | None,
    forwards: list[dict],
    principal_filter: dict,
    workdir: str | None,
    uid: int | None = None,
    gid: int | None = None,
) -> dict:
    """Build the JSON spec the shim reads with `--spec`.

    Args:
        argv (`list[str]`):
            The principal's command, resolved against `PATH` in `env`.
        env (`dict[str, str]`):
            The principal's full environment.
        landlock ([`LandlockSpec`], *optional*):
            Landlock domain for the principal. `None` skips Landlock.
        forwards (`list[dict]`):
            Forwarders the shim runs for the container's lifetime. Each is one of
            `{"kind": "tcp_to_unix", "host", "port", "path"}`,
            `{"kind": "unix_to_tcp", "path", "host", "port"}` or
            `{"kind": "unix_to_unix", "path", "target"}`.
        principal_filter (`dict`):
            Output of [`~openenv.core.openenvd.seccomp.principal_filter_spec`].
        workdir (`str`, *optional*):
            Working directory of the principal.
        uid (`int`, *optional*):
            Switch the principal to this UID before `exec`.
        gid (`int`, *optional*):
            Switch the principal to this GID before `exec`.

    Returns:
        `dict`: The JSON-serializable shim spec.

    Raises:
        `ValueError`: If `argv` is empty or a forward is malformed.
    """
    if not argv:
        raise ValueError("argv must not be empty")
    required = {
        "tcp_to_unix": ("host", "port", "path"),
        "unix_to_tcp": ("path", "host", "port"),
        "unix_to_unix": ("path", "target"),
    }
    for fwd in forwards:
        kind = fwd.get("kind")
        if kind not in required:
            raise ValueError(f"unknown forward kind {kind!r}")
        missing = [k for k in required[kind] if k not in fwd]
        if missing:
            raise ValueError(f"{kind} forward is missing {missing}")
    return {
        "argv": list(argv),
        "env": dict(env),
        "landlock": landlock.to_json() if landlock is not None else None,
        "forwards": [dict(f) for f in forwards],
        "principal_filter": dict(principal_filter),
        "workdir": workdir,
        "uid": uid,
        "gid": gid,
    }


def socket_filter_program(families: list[int], machine: str | None = None) -> bytes:
    """Assemble a classic-BPF seccomp program that denies `socket(family, ...)`.

    Syscalls from a foreign architecture are denied too, so a 32-bit ABI can't
    bypass the filter.

    Args:
        families (`list[int]`):
            Address families to deny with `EPERM`.
        machine (`str`, *optional*):
            A `platform.machine()` value. Defaults to the running machine.

    Returns:
        `bytes`: Packed `struct sock_filter` instructions.

    Raises:
        `ValueError`: If the architecture is not x86_64 or aarch64.
    """
    key = (machine or platform.machine()).lower()
    if key not in _SOCKET_SYSCALLS:
        raise ValueError(f"unsupported architecture {key!r}")
    audit_arch, socket_nrs = _SOCKET_SYSCALLS[key]
    deny = SECCOMP_RET_ERRNO | EPERM
    n = len(families)

    def ins(code: int, jt: int, jf: int, k: int) -> bytes:
        return struct.pack("=HBBI", code, jt, jf, k)

    prog = [
        ins(BPF_LD_W_ABS, 0, 0, 4),  # seccomp_data.arch
        ins(BPF_JMP_JEQ_K, 1, 0, audit_arch),
        ins(BPF_RET_K, 0, 0, deny),
        ins(BPF_LD_W_ABS, 0, 0, 0),  # seccomp_data.nr
    ]
    for i, nr in enumerate(socket_nrs):
        # Match: jump to the arg0 load. Miss on the last: jump to ALLOW.
        remaining = len(socket_nrs) - i - 1
        prog.append(ins(BPF_JMP_JEQ_K, remaining, 0 if remaining else n + 1, nr))
    prog.append(ins(BPF_LD_W_ABS, 0, 0, 16))  # low 32 bits of args[0]
    for i, family in enumerate(families):
        prog.append(ins(BPF_JMP_JEQ_K, n - i, 0, family))
    prog.append(ins(BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW))
    prog.append(ins(BPF_RET_K, 0, 0, deny))
    return b"".join(prog)


def install_socket_filter(families: list[int]) -> None:
    """Install a seccomp filter on the calling thread denying the given families.

    Args:
        families (`list[int]`):
            Address families `socket()` must refuse with `EPERM`.
    """
    program = socket_filter_program(families)
    count = len(program) // 8
    buf = ctypes.create_string_buffer(program, len(program))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = SockFprog(count, ctypes.addressof(buf))
    libc = ctypes.CDLL(None, use_errno=True)
    for args in (
        (PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0),
        (PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.addressof(fprog), 0, 0),
    ):
        if libc.prctl(*(ctypes.c_ulong(a) for a in args)) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))


def _set_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        _warn("could not become a child subreaper")


def _bring_up_lo() -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            ifreq = struct.pack("16sh22x", b"lo", 0)
            flags = struct.unpack("16sh22x", fcntl.ioctl(s, SIOCGIFFLAGS, ifreq))[1]
            if not flags & IFF_UP:
                fcntl.ioctl(
                    s, SIOCSIFFLAGS, struct.pack("16sh22x", b"lo", flags | IFF_UP)
                )
    except OSError:
        pass


async def _start_forward(fwd: dict) -> asyncio.base_events.Server:
    kind = fwd["kind"]
    if kind == "tcp_to_unix":
        return await forwarder.serve_tcp_to_unix(fwd["host"], fwd["port"], fwd["path"])
    if kind == "unix_to_tcp":
        return await forwarder.serve_unix_to_tcp(fwd["path"], fwd["host"], fwd["port"])
    if kind == "unix_to_unix":
        return await forwarder.serve_unix_to_unix(fwd["path"], fwd["target"])
    raise ValueError(f"unknown forward kind {kind!r}")


def _start_forwarders(forwards: list[dict]) -> asyncio.AbstractEventLoop | None:
    if not forwards:
        return None
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    errors: list[BaseException] = []

    async def start() -> None:
        try:
            for fwd in forwards:
                await _start_forward(fwd)
        except BaseException as exc:  # reported to the main thread
            errors.append(exc)
        finally:
            ready.set()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.create_task(start())
        loop.run_forever()

    threading.Thread(target=run, name="openenvd-forwarders", daemon=True).start()
    ready.wait()
    if errors:
        raise errors[0]
    return loop


def _exec_principal(spec: dict, go_fd: int) -> None:
    """Child side of the fork. Never returns."""
    try:
        os.read(go_fd, 1)
        os.close(go_fd)
        os.setsid()
        if spec.get("gid") is not None:
            os.setgroups([])
            os.setgid(spec["gid"])
        if spec.get("uid") is not None:
            os.setuid(spec["uid"])
        families = spec.get("principal_filter", {}).get("deny_socket_families") or []
        if _is_linux():
            if spec.get("landlock") is not None:
                restrict_self(LandlockSpec.from_json(spec["landlock"]))
            if families:
                install_socket_filter(families)
        if spec.get("workdir"):
            os.chdir(spec["workdir"])
        argv = spec["argv"]
        os.execvpe(argv[0], argv, spec.get("env") or {})
    except BaseException as exc:
        _warn(f"failed to start the principal: {exc}")
        os._exit(127)


def _exit_code(status: int) -> int:
    code = os.waitstatus_to_exitcode(status)
    return 128 - code if code < 0 else code


def run(spec: dict[str, Any]) -> int:
    """Run the shim with a parsed spec until the principal exits.

    Args:
        spec (`dict`):
            A spec built by [`shim_spec`].

    Returns:
        `int`: The principal's exit code, or `128 + signal` if it was killed.
    """
    if _is_linux():
        _set_subreaper()
        _bring_up_lo()
    else:
        skipped = ["child subreaper"]
        if spec.get("landlock") is not None:
            skipped.append("Landlock")
        if spec.get("principal_filter", {}).get("deny_socket_families"):
            skipped.append("seccomp socket filter")
        _warn(f"not Linux, skipping: {', '.join(skipped)}")

    # Fork before starting the forwarder thread; the child waits for the go byte.
    go_read, go_write = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        os.close(go_write)
        _exec_principal(spec, go_read)
    os.close(go_read)

    def forward_signal(signum: int, _frame: Any) -> None:
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            pass
        except PermissionError:
            os.kill(pid, signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    loop = None
    try:
        loop = _start_forwarders(spec.get("forwards") or [])
    except BaseException as exc:
        _warn(f"failed to start forwarders: {exc}")
        os.kill(pid, signal.SIGKILL)
        os.close(go_write)
        os.waitpid(pid, 0)
        return 126
    os.write(go_write, b"g")
    os.close(go_write)

    code = 1
    while True:
        try:
            reaped, status = os.wait()
        except ChildProcessError:
            break
        if reaped == pid:
            code = _exit_code(status)
            break
    # Collect zombies that already exited; PID 1 leaving tears down the rest.
    while True:
        try:
            reaped, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if reaped == 0:
            break
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
    return code


def main(argv: list[str] | None = None) -> int:
    """Entry point: `python -m openenv.core.openenvd.shim --spec <path>`.

    Args:
        argv (`list[str]`, *optional*):
            Command-line arguments. Defaults to `sys.argv[1:]`.

    Returns:
        `int`: The principal's exit code.
    """
    parser = argparse.ArgumentParser(prog="openenvd-shim")
    parser.add_argument("--spec", required=True, help="path to the shim JSON spec")
    args = parser.parse_args(argv)
    with open(args.spec) as f:
        spec = json.load(f)
    return run(spec)


if __name__ == "__main__":
    sys.exit(main())
