# SPDX-License-Identifier: BSD-3-Clause
"""Protect the SSH control stream before importing environment code.

The kernel resets the dumpable attribute at every execve (fs/exec.c
``setup_new_exec``): when real and effective credentials match, it becomes
SUID_DUMP_USER regardless of any earlier ``prctl(PR_SET_DUMPABLE, 0)``. The
seal therefore has to happen in the worker's own address space, and it must
run before third-party imports, which otherwise leave a long dumpable window
in which a same-UID principal could steal the daemon control descriptors
with pidfd_getfd (the ptrace family requires a dumpable target).

OpenShell executes this source as ``python -I -S -c <source>`` over SSH without
a PTY. The first input line supplies factory, action_class, and agent_policy;
subsequent lines contain requests. ``-I -S`` keeps environment paths and site
hooks from running before the control descriptors and process are protected.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import site
import sys

PR_SET_DUMPABLE = 4
WORKER_MODULE = "openenv.core.openenvd.worker"


def seal_process() -> None:
    """Refuse same-UID inspection: ptrace, pidfd_getfd, and /proc/pid/{fd,mem,environ}."""
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def protect_stdio() -> tuple[int, int]:
    """Keep control pipes private; ordinary input is EOF and output is diagnostics."""
    control_read = os.dup(0)
    control_write = os.dup(1)
    os.set_inheritable(control_read, False)
    os.set_inheritable(control_write, False)
    null = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(null, 0)
        os.dup2(2, 1)
    finally:
        os.close(null)
    return control_read, control_write


def main() -> None:
    seal_process()
    control_read, control_write = protect_stdio()
    # Site hooks and all later imports see safe standard descriptors. The worker
    # owns the private duplicates; subprocesses cannot inherit them even when
    # launched with close_fds=False.
    site.main()
    sys.argv[:] = [WORKER_MODULE]
    worker = importlib.import_module(WORKER_MODULE)
    worker.main(control_read, control_write)


if __name__ == "__main__":
    main()
