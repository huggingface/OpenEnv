# SPDX-License-Identifier: BSD-3-Clause

"""Landlock via ctypes: restrict the calling process's filesystem, ports and scope.

Stdlib only, since the container shim imports it. `restrict_self` negotiates
with the running kernel's ABI and only handles the access rights it supports.
"""

from __future__ import annotations

import ctypes
import json
import os
import stat
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13  # ABI 2
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14  # ABI 3
LANDLOCK_ACCESS_FS_IOCTL_DEV = 1 << 15  # ABI 5

LANDLOCK_ACCESS_NET_BIND_TCP = 1 << 0  # ABI 4
LANDLOCK_ACCESS_NET_CONNECT_TCP = 1 << 1  # ABI 4

LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0  # ABI 6
LANDLOCK_SCOPE_SIGNAL = 1 << 1  # ABI 6

LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_RULE_NET_PORT = 2

SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446

PR_SET_NO_NEW_PRIVS = 38

_FS_ABI1 = (1 << 13) - 1
_FS_BY_ABI = {
    1: _FS_ABI1,
    2: _FS_ABI1 | LANDLOCK_ACCESS_FS_REFER,
    3: _FS_ABI1 | LANDLOCK_ACCESS_FS_REFER | LANDLOCK_ACCESS_FS_TRUNCATE,
}
_FS_BY_ABI[4] = _FS_BY_ABI[3]
_FS_BY_ABI_MAX = _FS_BY_ABI[3] | LANDLOCK_ACCESS_FS_IOCTL_DEV

_FS_READ = (
    LANDLOCK_ACCESS_FS_EXECUTE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
)
# Rights that make sense on a non-directory; add_rule rejects the others.
_FS_FILE_RIGHTS = (
    LANDLOCK_ACCESS_FS_EXECUTE
    | LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_TRUNCATE
    | LANDLOCK_ACCESS_FS_IOCTL_DEV
)


class LandlockUnavailable(RuntimeError):
    """Raised when Landlock is required but the kernel doesn't provide it."""


class LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class LandlockPathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


class LandlockNetPortAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("port", ctypes.c_uint64),
    ]


def handled_fs_access(abi: int) -> int:
    """Return every filesystem access right a Landlock ABI version can handle.

    Args:
        abi (`int`):
            The Landlock ABI version.

    Returns:
        `int`: A bitmask of `LANDLOCK_ACCESS_FS_*`, `0` for ABI 0.
    """
    if abi <= 0:
        return 0
    return _FS_BY_ABI.get(abi, _FS_BY_ABI_MAX)


def ruleset_attr_size(abi: int) -> int:
    """Return the `landlock_ruleset_attr` size a Landlock ABI version accepts.

    Args:
        abi (`int`):
            The Landlock ABI version.

    Returns:
        `int`: 8 below ABI 4, 16 for ABI 4 and 5, 24 from ABI 6.
    """
    if abi >= 6:
        return 24
    if abi >= 4:
        return 16
    return 8


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _syscall(libc: ctypes.CDLL, nr: int, *args: Any) -> int:
    ret = libc.syscall(ctypes.c_long(nr), *args)
    if ret < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return ret


def abi_version() -> int:
    """Return the kernel's Landlock ABI version.

    Returns:
        `int`: The ABI version, or `0` when Landlock is unsupported or not Linux.
    """
    if not sys.platform.startswith("linux"):
        return 0
    try:
        return _syscall(
            _libc(),
            SYS_LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(None),
            ctypes.c_size_t(0),
            ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION),
        )
    except OSError:
        return 0


@dataclass
class LandlockSpec:
    """What a process may still reach after `restrict_self`.

    Attributes:
        read_only (`list[str]`):
            Paths the process may read and execute beneath.
        read_write (`list[str]`):
            Paths the process may do anything beneath.
        connect_tcp (`list[int]`, *optional*):
            TCP ports the process may connect to. `None` leaves networking alone.
        bind_tcp (`list[int]`, *optional*):
            TCP ports the process may bind. `None` leaves binding alone.
        scope (`bool`, *optional*, defaults to `True`):
            Scope abstract Unix sockets and signals to the domain (ABI 6).
        hard_requirement (`bool`, *optional*, defaults to `False`):
            Raise [`LandlockUnavailable`] instead of running unrestricted.
    """

    read_only: list[str] = field(default_factory=list)
    read_write: list[str] = field(default_factory=list)
    connect_tcp: list[int] | None = None
    bind_tcp: list[int] | None = None
    scope: bool = True
    hard_requirement: bool = False

    def to_json(self) -> dict:
        """Return the spec as a JSON-serializable `dict`."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict | str) -> "LandlockSpec":
        """Build a spec from a `dict` or JSON string produced by `to_json`."""
        if isinstance(data, str):
            data = json.loads(data)
        return cls(**data)


def _add_path_rule(libc: ctypes.CDLL, ruleset_fd: int, path: str, access: int) -> None:
    try:
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except FileNotFoundError:
        return
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            access &= _FS_FILE_RIGHTS
        if not access:
            return
        attr = LandlockPathBeneathAttr(allowed_access=access, parent_fd=fd)
        _syscall(
            libc,
            SYS_LANDLOCK_ADD_RULE,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_uint32(0),
        )
    finally:
        os.close(fd)


def _add_port_rule(libc: ctypes.CDLL, ruleset_fd: int, port: int, access: int) -> None:
    attr = LandlockNetPortAttr(allowed_access=access, port=port)
    _syscall(
        libc,
        SYS_LANDLOCK_ADD_RULE,
        ctypes.c_int(ruleset_fd),
        ctypes.c_int(LANDLOCK_RULE_NET_PORT),
        ctypes.byref(attr),
        ctypes.c_uint32(0),
    )


def restrict_self(spec: LandlockSpec) -> int:
    """Apply a Landlock domain to the calling thread and its future children.

    Also sets `PR_SET_NO_NEW_PRIVS`. Paths that don't exist are skipped.

    Args:
        spec ([`LandlockSpec`]):
            What the process may still reach.

    Returns:
        `int`: The ABI version used, or `0` when Landlock is unavailable.

    Raises:
        [`LandlockUnavailable`]: If Landlock is unavailable and `spec.hard_requirement`.
        `OSError`: If the kernel rejects the ruleset.
    """
    abi = abi_version()
    if abi == 0:
        if spec.hard_requirement:
            raise LandlockUnavailable("this kernel does not support Landlock")
        return 0

    fs = handled_fs_access(abi)
    net = 0
    if abi >= 4:
        if spec.connect_tcp is not None:
            net |= LANDLOCK_ACCESS_NET_CONNECT_TCP
        if spec.bind_tcp is not None:
            net |= LANDLOCK_ACCESS_NET_BIND_TCP
    scoped = 0
    if spec.scope and abi >= 6:
        scoped = LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET | LANDLOCK_SCOPE_SIGNAL

    libc = _libc()
    attr = LandlockRulesetAttr(
        handled_access_fs=fs, handled_access_net=net, scoped=scoped
    )
    ruleset_fd = _syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.c_size_t(ruleset_attr_size(abi)),
        ctypes.c_uint32(0),
    )
    try:
        for path in spec.read_only:
            _add_path_rule(libc, ruleset_fd, path, fs & _FS_READ)
        for path in spec.read_write:
            _add_path_rule(libc, ruleset_fd, path, fs)
        if net & LANDLOCK_ACCESS_NET_CONNECT_TCP:
            for port in spec.connect_tcp or ():
                _add_port_rule(libc, ruleset_fd, port, LANDLOCK_ACCESS_NET_CONNECT_TCP)
        if net & LANDLOCK_ACCESS_NET_BIND_TCP:
            for port in spec.bind_tcp or ():
                _add_port_rule(libc, ruleset_fd, port, LANDLOCK_ACCESS_NET_BIND_TCP)
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        _syscall(
            libc,
            SYS_LANDLOCK_RESTRICT_SELF,
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint32(0),
        )
    finally:
        os.close(ruleset_fd)
    return abi
