# SPDX-License-Identifier: BSD-3-Clause
"""Environment worker entrypoint: seal the process before untrusted imports.

The kernel resets the dumpable attribute at every execve (fs/exec.c
``setup_new_exec``): when real and effective credentials match, it becomes
SUID_DUMP_USER regardless of any earlier ``prctl(PR_SET_DUMPABLE, 0)``. The
seal therefore has to happen in the worker's own address space, and it must
run before third-party imports, which otherwise leave a long dumpable window
in which a same-UID principal could steal the inherited daemon control
listener with pidfd_getfd (the ptrace family requires a dumpable target).

This module is executed directly as a script (``python -S
_worker_bootstrap.py``), so no ``openenv`` package import happens before the
seal; ``-S`` defers site initialization until after it.
"""

from __future__ import annotations

import ctypes
import os
import runpy
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


def main() -> None:
    seal_process()
    # Restore the sys.path setup that -S skipped, now that the process is sealed.
    site.main()
    # This script's own directory is not a package root; remove it so its
    # modules cannot shadow real dependencies during the worker import.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [
        entry
        for entry in sys.path
        if os.path.abspath(entry or os.getcwd()) != script_dir
    ]
    sys.argv[0] = WORKER_MODULE
    runpy.run_module(WORKER_MODULE, run_name="__main__")


if __name__ == "__main__":
    main()
