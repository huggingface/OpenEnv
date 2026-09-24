# SPDX-License-Identifier: BSD-3-Clause

"""Isolation requests must succeed completely or never run the target."""

import asyncio
import errno
import json
import os
import socket
import sys
from unittest.mock import Mock

import pytest
from openenv.core.openenvd import isolation
from openenv.core.openenvd.models import TaskSpec


ROOT_CAPS = isolation.IsolationCapabilities(True, True, True)
USER_CAPS = isolation.IsolationCapabilities(False, False)


def task(**kwargs):
    return TaskSpec(name="test", argv=["true"], **kwargs)


class TestIsolationRequirements:
    @pytest.mark.parametrize(
        "spec, message",
        [
            (task(uid=1234, gid=1234), "privileged"),
            (task(network_isolated=True), "network namespace"),
        ],
    )
    async def test_unavailable_isolation_never_spawns(self, spec, message, monkeypatch):
        spawn = Mock()
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        with pytest.raises(isolation.IsolationError, match=message):
            await isolation.spawn_task(spec, USER_CAPS, env={})
        spawn.assert_not_called()

    def test_network_isolation_cannot_retain_root(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        with pytest.raises(isolation.IsolationError, match="dedicated UID"):
            isolation.validate_isolation(task(network_isolated=True), ROOT_CAPS)

    def test_kernel_namespace_failure_is_not_ignored(self, monkeypatch):
        libc = Mock()
        libc.unshare.return_value = -1
        monkeypatch.setattr(isolation.ctypes, "CDLL", lambda *a, **kw: libc)
        monkeypatch.setattr(isolation.ctypes, "get_errno", lambda: errno.EPERM)
        with pytest.raises(OSError) as error:
            isolation._unshare_net()
        assert error.value.errno == errno.EPERM

    def test_ipc_namespace_failure_is_not_ignored(self, monkeypatch):
        libc = Mock()
        libc.unshare.return_value = -1
        monkeypatch.setattr(isolation.ctypes, "CDLL", lambda *a, **kw: libc)
        monkeypatch.setattr(isolation.ctypes, "get_errno", lambda: errno.EPERM)
        with pytest.raises(OSError) as error:
            isolation._unshare_ipc()
        assert error.value.errno == errno.EPERM

    def test_mount_namespace_failure_is_not_ignored(self, monkeypatch):
        libc = Mock()
        libc.unshare.return_value = -1
        monkeypatch.setattr(isolation.ctypes, "CDLL", lambda *a, **kw: libc)
        monkeypatch.setattr(isolation.ctypes, "get_errno", lambda: errno.EPERM)
        with pytest.raises(OSError) as error:
            isolation._unshare_mount()
        assert error.value.errno == errno.EPERM

    def test_seccomp_load_failure_releases_filter_and_fails_closed(self, monkeypatch):
        lib = Mock()
        lib.seccomp_init.return_value = 1
        lib.seccomp_attr_set.return_value = 0
        lib.seccomp_syscall_resolve_name.return_value = 1
        lib.seccomp_rule_add.return_value = 0
        lib.seccomp_load.return_value = -errno.EPERM
        monkeypatch.setattr(isolation.ctypes, "CDLL", lambda *a, **kw: lib)
        with pytest.raises(OSError) as error:
            isolation._seccomp_restrict_kernel_state()
        assert error.value.errno == errno.EPERM
        lib.seccomp_release.assert_called_once_with(1)

    def test_no_new_privileges_failure_is_not_ignored(self, monkeypatch):
        libc = Mock()
        libc.prctl.return_value = -1
        monkeypatch.setattr(isolation.ctypes, "CDLL", lambda *a, **kw: libc)
        monkeypatch.setattr(isolation.ctypes, "get_errno", lambda: errno.EPERM)
        with pytest.raises(OSError) as error:
            isolation._no_new_privileges()
        assert error.value.errno == errno.EPERM

    def test_capability_probe_never_uses_python_after_fork(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        fork = Mock(side_effect=AssertionError("unsafe fork"))
        monkeypatch.setattr(os, "fork", fork)
        probe = Mock(return_value=Mock(returncode=0))
        monkeypatch.setattr(isolation.subprocess, "run", probe)
        assert isolation._probe_unshare_net()
        fork.assert_not_called()
        args, kwargs = probe.call_args
        assert args[0][1:3] == ["-I", "-S"]
        assert kwargs["env"] == {}
        assert kwargs["cwd"] == "/"
        assert "preexec_fn" not in kwargs


class TestHelperSetup:
    def _run(self, monkeypatch, *, failure=None):
        calls = []

        def unshare_net():
            calls.append("network")
            if failure == "network":
                raise PermissionError(errno.EPERM, "denied")

        def unshare_ipc():
            calls.append("ipc")
            if failure == "ipc":
                raise PermissionError(errno.EPERM, "denied")

        def unshare_mount():
            calls.append("mounts")
            if failure == "mounts":
                raise PermissionError(errno.EPERM, "denied")

        def seccomp_restrict_kernel_state():
            calls.append("seccomp")
            if failure == "seccomp":
                raise PermissionError(errno.EPERM, "denied")

        def no_new_privileges():
            calls.append("no_new_privileges")
            if failure == "no_new_privileges":
                raise PermissionError(errno.EPERM, "denied")

        def execute(argv0, argv, env):
            calls.append(("exec", argv, env))
            raise FileNotFoundError(errno.ENOENT, "missing executable")

        monkeypatch.setattr(isolation, "_unshare_net", unshare_net)
        monkeypatch.setattr(isolation, "_unshare_ipc", unshare_ipc)
        monkeypatch.setattr(isolation, "_unshare_mount", unshare_mount)
        monkeypatch.setattr(
            isolation, "_seccomp_restrict_kernel_state", seccomp_restrict_kernel_state
        )
        monkeypatch.setattr(isolation, "_no_new_privileges", no_new_privileges)
        monkeypatch.setattr(
            os, "setgroups", lambda value: calls.append(("groups", value))
        )
        monkeypatch.setattr(os, "setgid", lambda value: calls.append(("gid", value)))
        monkeypatch.setattr(os, "setuid", lambda value: calls.append(("uid", value)))
        monkeypatch.setattr(os, "chdir", lambda value: calls.append(("cwd", value)))
        monkeypatch.setattr(os, "execvpe", execute)
        config = {
            "argv": ["missing-command", "argument"],
            "env": {"LD_PRELOAD": "target-only", "SECRET": "private-value"},
            "cwd": "/task",
            "network_isolated": True,
            "uid": 1234,
            "gid": 5678,
        }
        parent, child = socket.socketpair()
        with parent:
            parent.sendall(json.dumps(config).encode())
            parent.shutdown(socket.SHUT_WR)
            assert isolation._run_helper(child.detach()) == 1
            response = parent.recv(4096)
        return calls, response

    @pytest.mark.parametrize(
        "failure", ["ipc", "mounts", "network", "seccomp", "no_new_privileges"]
    )
    def test_failed_setup_never_executes_target(self, monkeypatch, failure):
        calls, response = self._run(monkeypatch, failure=failure)
        assert not any(isinstance(call, tuple) for call in calls)
        assert not response.startswith(isolation._READY)
        assert b"could not" in response
        assert b"private-value" not in response

    def test_privileges_drop_before_target_cwd_and_environment(self, monkeypatch):
        calls, response = self._run(monkeypatch)
        assert calls[:9] == [
            "ipc",
            "mounts",
            "network",
            "no_new_privileges",
            "seccomp",
            ("groups", []),
            ("gid", 5678),
            ("uid", 1234),
            ("cwd", "/task"),
        ]
        assert calls[9] == (
            "exec",
            ["missing-command", "argument"],
            {"LD_PRELOAD": "target-only", "SECRET": "private-value"},
        )
        assert response.startswith(isolation._READY + b"could not execute task")
        assert b"private-value" not in response

    async def test_native_credentials_drop_clears_supplementary_groups(
        self, monkeypatch
    ):
        monkeypatch.setattr(sys, "platform", "darwin")
        captured = {}

        async def spawn(*args, **kwargs):
            captured.update(kwargs)
            return "process"

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        result = await isolation.spawn_task(task(uid=1234, gid=5678), ROOT_CAPS, env={})
        assert result == "process"
        assert captured["user"] == 1234
        assert captured["group"] == 5678
        assert captured["extra_groups"] == ()
        assert captured["umask"] == 0o077
        assert captured["stdin"] == asyncio.subprocess.DEVNULL
        assert "preexec_fn" not in captured

    @pytest.mark.parametrize("cancel", [False, True])
    async def test_incomplete_setup_kills_and_reaps_helper(self, monkeypatch, cancel):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(isolation, "_HELPER_TIMEOUT_S", 0.05)
        monkeypatch.setattr(
            isolation,
            "_helper_command",
            lambda: [sys.executable, "-I", "-S", "-c", "import time; time.sleep(30)"],
        )
        started = asyncio.Event()
        processes = []
        real_spawn = asyncio.create_subprocess_exec

        async def spawn(*args, **kwargs):
            process = await real_spawn(*args, **kwargs)
            processes.append(process)
            started.set()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        pending = asyncio.create_task(isolation.spawn_task(task(), USER_CAPS, env={}))
        await started.wait()
        if cancel:
            pending.cancel()
            expected = asyncio.CancelledError
        else:
            expected = isolation.IsolationError
        with pytest.raises(expected):
            await pending
        assert len(processes) == 1
        assert processes[0].returncode is not None


class TestRealSpawn:
    async def test_target_receives_exact_environment_cwd_and_private_umask(
        self, tmp_path
    ):
        script = (
            "import json,os,sys; "
            "print(json.dumps([dict(os.environ),os.getcwd(),os.umask(0),sys.stdin.read()]))"
        )
        spec = TaskSpec(
            name="probe",
            argv=[sys.executable, "-I", "-S", "-c", script],
            cwd=str(tmp_path),
        )
        proc = await isolation.spawn_task(
            spec, USER_CAPS, env={"TASK_ONLY": "value"}, stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        assert proc.returncode == 0
        env, cwd, umask, stdin = json.loads(stdout)
        # Python may normalize locale variables itself; daemon secrets must not leak.
        assert env.get("TASK_ONLY") == "value"
        assert set(env) <= {"TASK_ONLY", "LC_CTYPE", "__CF_USER_TEXT_ENCODING"}
        assert os.path.samefile(cwd, tmp_path)
        assert umask == 0o077
        assert stdin == ""

    async def test_exec_failure_is_reported_to_caller(self):
        spec = TaskSpec(name="missing", argv=["/no/such/openenvd-command"])
        with pytest.raises(OSError):
            await isolation.spawn_task(spec, USER_CAPS, env={})

    async def test_default_working_directory_is_root(self):
        spec = TaskSpec(name="cwd", argv=["/bin/pwd"])
        proc = await isolation.spawn_task(
            spec, USER_CAPS, env={}, stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        assert proc.returncode == 0
        assert stdout.strip() == b"/"

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux isolation helper")
    async def test_target_python_configuration_cannot_execute_in_helper(self, tmp_path):
        marker = tmp_path / "injected"
        injection = f"open({str(marker)!r}, 'w').write('executed')\n"
        (tmp_path / "sitecustomize.py").write_text(injection)
        (tmp_path / "socket.py").write_text(injection)
        spec = TaskSpec(
            name="injection",
            argv=["/bin/sh", "-c", 'printf "%s" "$PYTHONPATH"'],
            cwd=str(tmp_path),
        )
        proc = await isolation.spawn_task(
            spec,
            USER_CAPS,
            env={"PYTHONPATH": str(tmp_path)},
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        assert proc.returncode == 0
        assert stdout.decode() == str(tmp_path)
        assert not marker.exists()

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux no_new_privs")
    async def test_linux_target_cannot_gain_privileges_on_exec(self):
        spec = TaskSpec(
            name="probe",
            argv=["/bin/cat", "/proc/self/status"],
        )
        proc = await isolation.spawn_task(
            spec, USER_CAPS, env={}, stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        assert proc.returncode == 0
        assert b"NoNewPrivs:\t1" in stdout

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux isolation helper")
    async def test_target_inherits_neither_helper_socket_nor_daemon_descriptors(self):
        script = """
import json, os
opened = []
for name in os.listdir('/proc/self/fd'):
    if int(name) <= 2:
        continue
    try:
        os.fstat(int(name))
    except OSError:
        continue
    opened.append(name)
print(json.dumps(opened))
"""
        spec = TaskSpec(
            name="descriptors",
            argv=[sys.executable, "-I", "-S", "-c", script],
        )
        with socket.socket() as daemon_socket:
            daemon_socket.set_inheritable(True)
            proc = await isolation.spawn_task(
                spec, USER_CAPS, env={}, stdout=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
        assert proc.returncode == 0
        assert json.loads(stdout) == []

    @pytest.mark.skipif(
        os.geteuid() != 0 or not isolation.in_initial_user_ns(),
        reason="requires Linux root with mapped task IDs",
    )
    async def test_real_uid_drop_has_no_inherited_groups(self):
        spec = TaskSpec(
            name="probe",
            argv=["/bin/cat", "/proc/self/status"],
            uid=65534,
            gid=65534,
        )
        proc = await isolation.spawn_task(
            spec, ROOT_CAPS, env={}, stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        assert proc.returncode == 0
        fields = dict(line.split(":", 1) for line in stdout.decode().splitlines())
        assert fields["Uid"].split() == ["65534"] * 4
        assert fields["Gid"].split() == ["65534"] * 4
        assert fields["Groups"].strip() == ""

    @pytest.mark.skipif(
        os.geteuid() != 0 or not isolation.in_initial_user_ns(),
        reason="requires Linux root with mapped task IDs",
    )
    async def test_real_network_namespace_and_loopback(self):
        caps = isolation.detect_capabilities()
        if not caps.can_unshare_net:
            pytest.skip("runtime disallows network namespaces")
        script = (
            "import os,socket; "
            "s=socket.socket(); s.bind(('127.0.0.1',0)); "
            "print(os.readlink('/proc/self/ns/net'))"
        )
        spec = TaskSpec(
            name="probe",
            argv=[sys.executable, "-I", "-S", "-c", script],
            uid=65534,
            gid=65534,
            network_isolated=True,
        )
        proc = await isolation.spawn_task(
            spec, caps, env={}, stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        assert proc.returncode == 0
        assert stdout.decode().strip() != os.readlink("/proc/self/ns/net")

    @pytest.mark.skipif(
        os.geteuid() != 0 or not isolation.in_initial_user_ns(),
        reason="requires Linux root with mapped task IDs",
    )
    async def test_sysvipc_cannot_outlive_the_task_in_the_daemon_namespace(self):
        """Workloads cannot create persistent SysV objects at all."""
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
        libc.shmget.restype = ctypes.c_int
        libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        libc.shmctl.restype = ctypes.c_int
        IPC_CREAT = 0o1000
        IPC_EXCL = 0o2000
        IPC_RMID = 0
        key = 0x000A11CE
        # Probe key must be free in the daemon namespace before the test.
        assert libc.shmget(key, 4096, 0) < 0

        script = (
            "import ctypes;"
            "libc=ctypes.CDLL(None,use_errno=True);"
            "libc.shmget.argtypes=[ctypes.c_int,ctypes.c_size_t,ctypes.c_int];"
            "libc.shmget.restype=ctypes.c_int;"
            f"id=libc.shmget({key},4096,{IPC_CREAT | IPC_EXCL | 0o600});"
            "raise SystemExit(0 if id == -1 and ctypes.get_errno() == 1 else 1)"
        )
        spec = TaskSpec(
            name="sysvipc",
            argv=[sys.executable, "-I", "-S", "-c", script],
            uid=65534,
            gid=65534,
        )
        proc = await isolation.spawn_task(spec, ROOT_CAPS, env={})
        await proc.wait()
        assert proc.returncode == 0
        # No segment can remain in the daemon after the denied creation.
        leaked = libc.shmget(key, 4096, 0)
        assert leaked < 0
        if leaked >= 0:
            libc.shmctl(leaked, IPC_RMID, None)

    @pytest.mark.skipif(
        sys.platform != "linux" or os.geteuid() != 0, reason="Linux root"
    )
    async def test_workload_cannot_create_keyrings_or_shared_ipc_files(self):
        # Test-only syscall probe stays in the child; production never needs an
        # add_key helper or a sweep of shared host resources.
        script = """
import ctypes, os
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
number = 248 if os.uname().machine == 'x86_64' else 217
result = libc.syscall(ctypes.c_long(number), ctypes.c_char_p(b'user'),
    ctypes.c_char_p(b'openenvd-test'), ctypes.c_char_p(b'payload'),
    ctypes.c_long(7), ctypes.c_long(-4))
assert result == -1 and ctypes.get_errno() == 1
for root in ('/dev/shm', '/dev/mqueue'):
    try:
        with open(root + '/openenvd-test', 'w') as f:
            f.write('must not exist')
    except PermissionError:
        continue
    raise AssertionError('shared IPC filesystem writable')
"""
        proc = await isolation.spawn_task(
            TaskSpec(
                name="ipc-denied",
                argv=[sys.executable, "-I", "-S", "-c", script],
                uid=65534,
                gid=65534,
            ),
            ROOT_CAPS,
            env={},
        )
        await proc.wait()
        assert proc.returncode == 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "linux" or os.uname().machine != "x86_64", reason="Linux x86 ABI"
)
async def test_seccomp_rejects_x32_compatibility_bypass():
    import signal

    proc = await isolation.spawn_task(
        TaskSpec(
            name="x32-denied",
            argv=[
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import ctypes; ctypes.CDLL(None).syscall(ctypes.c_long(0x40000000+248),0,0,0,0,0)",
            ],
        ),
        ROOT_CAPS,
        env={},
    )
    await proc.wait()
    assert proc.returncode == -signal.SIGSYS
