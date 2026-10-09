# SPDX-License-Identifier: BSD-3-Clause

"""What the unit's runtime lets openenvd build.

Each probe tries one kernel feature and reports whether it worked. [`assess`]
turns the results into a [`Tier`] and a strength per [`Guarantee`];
[`ensure`] refuses to start when the manifest needs more than that.
Probes never raise: a failure is a result with a short reason.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import signal
import struct
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .contract import EnforcementSpec, Guarantee, Strength, Tier

CLONE_NEWNS = 0x00020000
CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000
CLONE_NEWNET = 0x40000000
MS_RDONLY = 1
MS_NOSYMFOLLOW = 256
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_PRIVATE = 1 << 18
_MS_LOCKED = 2 | 4 | 8 | 1024 | 2048 | 4096  # nosuid nodev noexec *atime
SYS_LANDLOCK_CREATE_RULESET = 444
LANDLOCK_CREATE_RULESET_VERSION = 1
PR_GET_SECCOMP = 21
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
FS_IOC_GETFLAGS = 0x80086601
FS_IOC_SETFLAGS = 0x40086602
FS_APPEND_FL = 0x20
OCI_RUNTIMES: tuple[str, ...] = ("crun", "runc", "runsc")
_CHILD_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ProbeResult:
    """The outcome of one probe.

    Attributes:
        name (`str`):
            Probe name, e.g. `userns` or `oci_crun`.
        ok (`bool`):
            Whether the feature works here.
        detail (`str`, *optional*):
            Why not, or a value such as `abi=4` or a runtime path.
    """

    name: str
    ok: bool
    detail: str = ""


@dataclass
class EnforcementReport:
    """The tier openenvd can build and how strongly each guarantee holds.

    Attributes:
        tier ([`Tier`]):
            What the runtime permits.
        strengths (`dict[Guarantee, Strength]`):
            Obtained strength per guarantee.
        probes (`list[ProbeResult]`):
            The raw probe results. Internal; never sent to the orchestrator.
    """

    tier: Tier
    strengths: dict[Guarantee, Strength]
    probes: list[ProbeResult]

    @property
    def landlock_abi(self) -> int:
        return landlock_abi(self.probes)

    def to_info(self) -> dict:
        """The orchestrator-facing summary: tier and strengths, nothing else."""
        return {
            "tier": self.tier.value,
            "guarantees": {g.value: s.value for g, s in self.strengths.items()},
        }


class EnforcementUnavailable(RuntimeError):
    """The unit can't provide what the manifest requires."""


def _linux() -> bool:
    return sys.platform.startswith("linux")


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _check(ret: int, what: str) -> None:
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{what}: {os.strerror(err)}")


def _not_linux(name: str) -> ProbeResult | None:
    return None if _linux() else ProbeResult(name, False, "not linux")


def _in_child(name: str, body: Callable[[], str]) -> ProbeResult:
    """Run `body` in a forked child. It returns a detail or raises on failure."""
    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        status = 1
        try:
            os.close(read_fd)
            try:
                msg = "1" + body()
                status = 0
            except BaseException as exc:  # noqa: BLE001 - report, never propagate
                msg = "0" + (str(exc) or type(exc).__name__)
            os.write(write_fd, msg.encode()[:512])
        finally:
            os._exit(status)
    os.close(write_fd)
    try:
        deadline = time.monotonic() + _CHILD_TIMEOUT_S
        while True:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() >= deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return ProbeResult(name, False, "probe timed out")
            time.sleep(0.005)
        msg = os.read(read_fd, 512).decode(errors="replace")
    finally:
        os.close(read_fd)
    if not msg:
        return ProbeResult(name, False, "probe child died")
    return ProbeResult(name, msg[0] == "1", msg[1:])


def _safe(name: str, fn: Callable[[], ProbeResult]) -> ProbeResult:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - probes never raise
        return ProbeResult(name, False, str(exc) or type(exc).__name__)


def _enter_mount_ns(libc: ctypes.CDLL) -> None:
    """Enter a private mount namespace, via a user namespace when not root."""
    if os.geteuid() == 0:
        _check(libc.unshare(CLONE_NEWNS), "unshare(CLONE_NEWNS)")
    else:
        uid, gid = os.geteuid(), os.getegid()
        _check(libc.unshare(CLONE_NEWUSER | CLONE_NEWNS), "unshare(CLONE_NEWUSER)")
        Path("/proc/self/uid_map").write_text(f"0 {uid} 1")
        Path("/proc/self/setgroups").write_text("deny")
        Path("/proc/self/gid_map").write_text(f"0 {gid} 1")
    _check(
        libc.mount(None, b"/", None, MS_REC | MS_PRIVATE, None),
        "make / private",
    )


def probe_userns() -> ProbeResult:
    """Whether an unprivileged child can create user, mount, pid and net namespaces.

    Returns:
        [`ProbeResult`]: Named `userns`.
    """
    name = "userns"
    if (skip := _not_linux(name)) is not None:
        return skip

    def body() -> str:
        libc = _libc()
        _check(libc.unshare(CLONE_NEWUSER), "unshare(CLONE_NEWUSER)")
        _check(
            libc.unshare(CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET),
            "unshare(CLONE_NEWNS|CLONE_NEWPID|CLONE_NEWNET)",
        )
        return ""

    return _safe(name, lambda: _in_child(name, body))


def probe_cgroup_writable(root: Path) -> ProbeResult:
    """Whether openenvd can create groups and delegate controllers under `root`.

    Args:
        root (`Path`):
            The unit's cgroup root.

    Returns:
        [`ProbeResult`]: Named `cgroup_writable`.
    """
    name = "cgroup_writable"

    def run() -> ProbeResult:
        root_ = Path(root)
        control = root_ / "cgroup.subtree_control"
        if not control.exists():
            return ProbeResult(name, False, "no cgroup v2 at the root")
        probe = root_ / ".openenvd-probe"
        try:
            probe.mkdir(exist_ok=True)
        except OSError as exc:
            return ProbeResult(name, False, f"mkdir: {exc.strerror}")
        try:
            enabled = {c.lstrip("+") for c in control.read_text().split()}
            if enabled:
                control.write_text(" ".join(f"+{c}" for c in sorted(enabled)))
                back = {c.lstrip("+") for c in control.read_text().split()}
                if not enabled <= back:
                    return ProbeResult(name, False, "subtree_control did not stick")
            elif not os.access(control, os.W_OK):
                return ProbeResult(name, False, "subtree_control is read-only")
        except OSError as exc:
            return ProbeResult(name, False, f"subtree_control: {exc.strerror}")
        finally:
            _rmdir_probe(probe)
        return ProbeResult(name, True)

    return _safe(name, run)


def _rmdir_probe(probe: Path) -> None:
    try:
        probe.rmdir()
    except OSError:
        shutil.rmtree(probe, ignore_errors=True)


def probe_cgroup_kill(root: Path) -> ProbeResult:
    """Whether the kernel has `cgroup.kill` (5.14+).

    The true root cgroup has no `cgroup.kill`, so a child group is checked too.

    Args:
        root (`Path`):
            The unit's cgroup root.

    Returns:
        [`ProbeResult`]: Named `cgroup_kill`.
    """
    name = "cgroup_kill"

    def run() -> ProbeResult:
        root_ = Path(root)
        if (root_ / "cgroup.kill").exists():
            return ProbeResult(name, True)
        probe = root_ / ".openenvd-probe-kill"
        try:
            probe.mkdir(exist_ok=True)
        except OSError:
            return ProbeResult(name, False, "no cgroup.kill")
        try:
            found = (probe / "cgroup.kill").exists()
        finally:
            _rmdir_probe(probe)
        return ProbeResult(name, found, "" if found else "no cgroup.kill")

    return _safe(name, run)


def probe_landlock() -> ProbeResult:
    """The Landlock ABI version, via `landlock_create_ruleset(NULL, 0, VERSION)`.

    Returns:
        [`ProbeResult`]: Named `landlock`, detail `abi=<n>`.
    """
    name = "landlock"
    if (skip := _not_linux(name)) is not None:
        return skip

    def run() -> ProbeResult:
        libc = _libc()
        libc.syscall.restype = ctypes.c_long
        abi = libc.syscall(
            ctypes.c_long(SYS_LANDLOCK_CREATE_RULESET),
            None,
            ctypes.c_size_t(0),
            ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION),
        )
        if abi < 0:
            return ProbeResult(name, False, os.strerror(ctypes.get_errno()))
        return ProbeResult(name, abi >= 1, f"abi={abi}")

    return _safe(name, run)


def landlock_abi(results: list[ProbeResult]) -> int:
    """The Landlock ABI from a `landlock` probe result, or `0`.

    Args:
        results (`list[ProbeResult]`):
            Probe results.

    Returns:
        `int`: The ABI version, `0` when Landlock is unavailable.
    """
    for r in results:
        if r.name == "landlock" and r.ok:
            m = re.search(r"abi=(\d+)", r.detail)
            return int(m.group(1)) if m else 0
    return 0


def probe_seccomp() -> ProbeResult:
    """Whether the kernel supports seccomp filters. Installs none.

    Returns:
        [`ProbeResult`]: Named `seccomp`.
    """
    name = "seccomp"
    if (skip := _not_linux(name)) is not None:
        return skip

    def run() -> ProbeResult:
        status = Path("/proc/self/status").read_text()
        if not any(line.startswith("Seccomp:") for line in status.splitlines()):
            return ProbeResult(name, False, "kernel built without seccomp")
        libc = _libc()
        if libc.prctl(PR_GET_SECCOMP, 0, 0, 0, 0) < 0:
            return ProbeResult(name, False, os.strerror(ctypes.get_errno()))
        # A NULL filter faults (EFAULT) only when filter mode is compiled in.
        libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, None, 0, 0)
        err = ctypes.get_errno()
        if err != 14:  # EFAULT
            return ProbeResult(name, False, f"filter mode: {os.strerror(err)}")
        return ProbeResult(name, True)

    return _safe(name, run)


def _scratch_dir(tmp: Path, prefix: str) -> Path:
    Path(tmp).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=tmp))


def _cleanup(base: Path) -> None:
    # overlayfs leaves a mode-000 work/work dir that rmtree can't enter.
    for dirpath, dirnames, _ in os.walk(base):
        for name in dirnames:  # before os.walk descends into them
            try:
                os.chmod(os.path.join(dirpath, name), 0o700)
            except OSError:
                pass
    shutil.rmtree(base, ignore_errors=True)


def probe_overlay(tmp: Path) -> ProbeResult:
    """Whether overlayfs can be mounted (in a user namespace when not root).

    Args:
        tmp (`Path`):
            Scratch directory for the lower, upper and work dirs.

    Returns:
        [`ProbeResult`]: Named `overlay`.
    """
    name = "overlay"
    if (skip := _not_linux(name)) is not None:
        return skip

    def run() -> ProbeResult:
        base = _scratch_dir(tmp, "overlay-")
        try:
            dirs = {d: base / d for d in ("lower", "upper", "work", "merged")}
            for d in dirs.values():
                d.mkdir()

            def body() -> str:
                libc = _libc()
                _enter_mount_ns(libc)
                opts = (
                    f"lowerdir={dirs['lower']},upperdir={dirs['upper']},"
                    f"workdir={dirs['work']}"
                )
                _check(
                    libc.mount(
                        b"overlay",
                        str(dirs["merged"]).encode(),
                        b"overlay",
                        0,
                        opts.encode(),
                    ),
                    "mount overlay",
                )
                return ""

            return _in_child(name, body)
        finally:
            _cleanup(base)

    return _safe(name, run)


def probe_nosymfollow(tmp: Path) -> ProbeResult:
    """Whether a bind mount can be remounted `nosymfollow` (5.10+).

    Args:
        tmp (`Path`):
            Scratch directory.

    Returns:
        [`ProbeResult`]: Named `nosymfollow`.
    """
    name = "nosymfollow"
    if (skip := _not_linux(name)) is not None:
        return skip

    def run() -> ProbeResult:
        base = _scratch_dir(tmp, "nosymfollow-")
        try:

            def body() -> str:
                libc = _libc()
                _enter_mount_ns(libc)
                target = str(base).encode()
                _check(libc.mount(target, target, None, MS_BIND, None), "bind")
                locked = os.statvfs(base).f_flag & _MS_LOCKED
                flags = MS_REMOUNT | MS_BIND | MS_RDONLY | MS_NOSYMFOLLOW | locked
                _check(libc.mount(None, target, None, flags, None), "remount")
                return ""

            return _in_child(name, body)
        finally:
            _cleanup(base)

    return _safe(name, run)


def probe_append_only(tmp: Path) -> ProbeResult:
    """Whether a file can be made append-only (`FS_APPEND_FL`).

    Args:
        tmp (`Path`):
            Scratch directory on the filesystem that will hold the trace.

    Returns:
        [`ProbeResult`]: Named `append_only`. The flag is cleared again.
    """
    name = "append_only"
    if (skip := _not_linux(name)) is not None:
        return skip

    def run() -> ProbeResult:
        import fcntl

        base = _scratch_dir(tmp, "append-")
        path = base / "probe"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            buf = bytearray(4)
            fcntl.ioctl(fd, FS_IOC_GETFLAGS, buf)
            flags = struct.unpack("i", buf)[0]
            try:
                fcntl.ioctl(fd, FS_IOC_SETFLAGS, struct.pack("i", flags | FS_APPEND_FL))
            except OSError as exc:
                return ProbeResult(name, False, f"set append flag: {exc.strerror}")
            fcntl.ioctl(fd, FS_IOC_SETFLAGS, struct.pack("i", flags & ~FS_APPEND_FL))
            return ProbeResult(name, True)
        finally:
            os.close(fd)
            _cleanup(base)

    return _safe(name, run)


def probe_oci_runtimes() -> dict[str, ProbeResult]:
    """Which OCI runtimes are on `PATH`.

    Returns:
        `dict[str, ProbeResult]`: Runtime name to a result named `oci_<runtime>`.
    """
    out = {}
    for runtime in OCI_RUNTIMES:
        path = shutil.which(runtime)
        out[runtime] = ProbeResult(f"oci_{runtime}", path is not None, path or "")
    return out


def probe_all(cgroup_root: Path, scratch: Path) -> list[ProbeResult]:
    """Run every probe.

    Args:
        cgroup_root (`Path`):
            The unit's cgroup root.
        scratch (`Path`):
            Scratch directory, ideally on the filesystem that will hold the trace.

    Returns:
        `list[ProbeResult]`: One result per probe.
    """
    return [
        probe_userns(),
        probe_cgroup_writable(cgroup_root),
        probe_cgroup_kill(cgroup_root),
        probe_landlock(),
        probe_seccomp(),
        probe_overlay(scratch),
        probe_nosymfollow(scratch),
        probe_append_only(scratch),
        *probe_oci_runtimes().values(),
    ]


_P, _D, _N = Strength.PREVENTED, Strength.DETECTED_AND_REAPED, Strength.NOT_SUPPORTED
_G = Guarantee

# Guarantee -> strength in each tier. In the landlock tier some rows depend on
# the ABI (4+ adds TCP rules) or on seccomp: (strength if met, condition).
_STRENGTHS: dict[Guarantee, dict[Tier, Strength | tuple[Strength, str]]] = {
    _G.ASSET_ISOLATION: {Tier.CONTAINERS: _P, Tier.LANDLOCK: _P},
    _G.CONTROL_PLANE_ISOLATION: {Tier.CONTAINERS: _P, Tier.LANDLOCK: (_P, "abi>=4")},
    _G.EGRESS_CONTROL: {Tier.CONTAINERS: _P, Tier.LANDLOCK: (_P, "seccomp")},
    _G.PRIVILEGE_DROP: {Tier.CONTAINERS: _P, Tier.LANDLOCK: _D},
    _G.PRINCIPAL_ISOLATION: {Tier.CONTAINERS: _P, Tier.LANDLOCK: _N},
    _G.SERVICE_ISOLATION: {Tier.CONTAINERS: _P, Tier.LANDLOCK: (_P, "abi>=4")},
    _G.OBSERVER_ISOLATION: {Tier.CONTAINERS: _P, Tier.LANDLOCK: _P},
    _G.RESOURCE_ISOLATION: {Tier.CONTAINERS: _P, Tier.LANDLOCK: _D},
    _G.TRACE_INTEGRITY: {Tier.CONTAINERS: _P, Tier.LANDLOCK: _D},
}


def _by_name(probes: list[ProbeResult]) -> dict[str, ProbeResult]:
    return {p.name: p for p in probes}


def _ok(probes: dict[str, ProbeResult], name: str) -> bool:
    return name in probes and probes[name].ok


def _has_oci(probes: dict[str, ProbeResult]) -> bool:
    return any(_ok(probes, f"oci_{r}") for r in OCI_RUNTIMES)


def assess(probes: list[ProbeResult]) -> EnforcementReport:
    """Turn probe results into a tier and a strength per guarantee.

    Args:
        probes (`list[ProbeResult]`):
            Results from [`probe_all`].

    Returns:
        [`EnforcementReport`]: The tier and strengths.
    """
    named = _by_name(probes)
    abi = landlock_abi(probes)
    if _ok(named, "userns") and _ok(named, "cgroup_writable") and _has_oci(named):
        tier = Tier.CONTAINERS
    elif abi >= 1:
        tier = Tier.LANDLOCK
    else:
        tier = Tier.NONE
    met = {"abi>=4": abi >= 4, "seccomp": _ok(named, "seccomp")}
    strengths: dict[Guarantee, Strength] = {}
    for guarantee, row in _STRENGTHS.items():
        cell = row.get(tier, _N)
        if isinstance(cell, tuple):
            cell = cell[0] if met[cell[1]] else _N
        strengths[guarantee] = cell
    return EnforcementReport(tier=tier, strengths=strengths, probes=list(probes))


def _detail(probes: dict[str, ProbeResult], name: str, label: str) -> str:
    p = probes.get(name)
    if p is None:
        return f"{label}: not probed"
    return f"{label}: {p.detail or 'unavailable'}"


def _tier_blockers(report: EnforcementReport) -> list[str]:
    """Why the unit didn't reach the containers tier."""
    named = _by_name(report.probes)
    out = []
    if not _ok(named, "userns"):
        out.append(_detail(named, "userns", "user namespaces"))
    if not _ok(named, "cgroup_writable"):
        out.append(_detail(named, "cgroup_writable", "cgroups"))
    if not _has_oci(named):
        out.append(f"no OCI runtime ({', '.join(OCI_RUNTIMES)}) on PATH")
    if report.tier is Tier.NONE:
        out.append(_detail(named, "landlock", "landlock"))
    return out


def _why(guarantee: Guarantee, report: EnforcementReport) -> list[str]:
    named = _by_name(report.probes)
    reasons = _tier_blockers(report)
    cell = _STRENGTHS[guarantee].get(Tier.LANDLOCK)
    if report.tier is Tier.LANDLOCK and isinstance(cell, tuple):
        if cell[1] == "abi>=4":
            reasons.append(f"landlock abi={report.landlock_abi} < 4")
        elif cell[1] == "seccomp":
            reasons.append(_detail(named, "seccomp", "seccomp"))
    return reasons


def ensure(enforcement: EnforcementSpec, report: EnforcementReport) -> None:
    """Refuse to start if the unit can't meet the manifest's enforcement block.

    Args:
        enforcement ([`EnforcementSpec`]):
            Required guarantees and acceptable tiers.
        report ([`EnforcementReport`]):
            What [`assess`] found.

    Raises:
        [`EnforcementUnavailable`]: Listing every gap and the probe that explains it.
    """
    gaps = []
    if enforcement.tiers and report.tier not in enforcement.tiers:
        allowed = ", ".join(t.value for t in enforcement.tiers)
        reasons = "; ".join(_tier_blockers(report))
        gaps.append(
            f"tier: one of [{allowed}] required, {report.tier.value} obtained"
            + (f" ({reasons})" if reasons else "")
        )
    for guarantee, minimum in enforcement.require.items():
        obtained = report.strengths.get(guarantee, Strength.NOT_SUPPORTED)
        if obtained.satisfies(minimum):
            continue
        reasons = "; ".join(_why(guarantee, report))
        gaps.append(
            f"{guarantee.value}: {minimum.value} required, {obtained.value} obtained"
            + (f" ({reasons})" if reasons else "")
        )
    if gaps:
        raise EnforcementUnavailable("; ".join(gaps))
