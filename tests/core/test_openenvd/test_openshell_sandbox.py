# SPDX-License-Identifier: BSD-3-Clause
"""OpenShell admission, transport isolation, and confirmed cleanup."""

import asyncio
import io
import json
import os
import shlex
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from openenv.core.openenvd.backends.openshell import OpenShellSandbox, WORKSPACE
from openenv.core.openenvd.isolation import IsolationError


@pytest.fixture
def settings():
    return SimpleNamespace(
        image="registry.example/openenv@sha256:1234",
        gateway="training",
        workspace="episodes",
        python="/usr/local/bin/python3",
        policy={
            "version": 1,
            "filesystem_policy": {
                "include_workdir": False,
                "read_only": ["/usr", "/lib", "/etc"],
                "read_write": ["/sandbox", "/tmp", "/dev/null"],
            },
            "landlock": {"compatibility": "hard_requirement"},
            "process": {"run_as_user": "1000", "run_as_group": "1000"},
            "network_policies": {},
        },
    )


class Gateway:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.calls = []
        self.exists = False
        self.version = "openshell 0.1.2"
        self.policy_source = "sandbox"
        self.extra_write = None
        self.create_error = False
        self.get_error = False

    def metadata(self):
        sandbox = self.sandbox
        return {
            "id": "immutable-sandbox-id",
            "name": sandbox.name,
            "workspace": sandbox.settings.workspace,
            "phase": "Ready",
            "labels": {"openenv-session": sandbox._label},
        }

    async def run(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[-1] == "--version":
            return self.version
        operation = argv[argv.index("sandbox") + 1]
        if operation == "create":
            self.exists = True
            if self.create_error:
                raise IsolationError("OpenShell command failed")
            return "Created sandbox successfully\n"
        if operation == "get":
            if self.get_error:
                raise IsolationError("OpenShell metadata unavailable")
            policy = json.loads(json.dumps(self.sandbox.policy))
            if self.extra_write:
                policy["filesystem_policy"]["read_write"].append(self.extra_write)
            return json.dumps(
                {
                    **self.metadata(),
                    "policy_source": self.policy_source,
                    "policy": policy,
                }
            )
        if operation == "list":
            return json.dumps(
                {
                    "sandboxes": [self.metadata()] if self.exists else [],
                    "next_page_token": "",
                }
            )
        if operation == "delete":
            self.exists = False
        if operation == "ssh-config":
            return f"Host openshell-{self.sandbox.name}.episodes\n    User sandbox\n"
        return ""


@pytest.fixture
def sandbox(settings, monkeypatch):
    value = OpenShellSandbox(settings, settings.policy)
    gateway = Gateway(value)
    monkeypatch.setattr(
        "openenv.core.openenvd.backends.openshell.shutil.which",
        lambda binary: "/usr/bin/" + binary,
    )
    monkeypatch.setattr(value, "_run", AsyncMock(side_effect=gateway.run))
    return value, gateway


def directories(tmp_path):
    seed, private = tmp_path / "workspace", tmp_path / "private"
    seed.mkdir()
    private.mkdir(mode=0o700)
    (seed / "initial.txt").write_text("initial")
    return seed, private


async def test_start_scopes_commands_and_verifies_policy_before_upload(
    sandbox, tmp_path
):
    value, gateway = sandbox
    seed, private = directories(tmp_path)
    await value.start(seed, private)
    calls = gateway.calls
    assert all(
        call[:7]
        == [
            "/usr/bin/openshell",
            "--gateway",
            "training",
            "--workspace",
            "episodes",
            "--color",
            "never",
        ]
        for call in calls
    )
    assert [call[8] for call in calls] == ["create", "get", "upload", "ssh-config"]
    create = calls[0]
    assert create[-3:] == ["--", "/bin/sleep", "1260"]
    assert {"--detach", "--no-keep", "--no-tty", "--no-auto-providers"} <= set(create)
    assert "--output" not in create
    assert "--provider" not in create
    assert value.id == "immutable-sandbox-id"
    assert (
        yaml.safe_load((private / "openshell-policy.yaml").read_text()) == value.policy
    )
    assert (private / "openshell-ssh.config").stat().st_mode & 0o777 == 0o600
    assert calls[2][-3:] == [str(seed), "/sandbox", "--no-git-ignore"]


async def test_main_process_retention_backstop_rounds_up_timeout(sandbox, tmp_path):
    value, gateway = sandbox
    value.timeout_s = 1.01
    await value.start(*directories(tmp_path))
    create = gateway.calls[0]
    assert create[-3:] == ["--", "/bin/sleep", "65"]
    assert "--no-keep" in create
    assert "--output" not in create


async def test_metadata_failure_recovers_and_deletes_owned_sandbox(sandbox, tmp_path):
    value, gateway = sandbox
    gateway.get_error = True
    with pytest.raises(IsolationError, match="metadata unavailable"):
        await value.start(*directories(tmp_path))
    assert not gateway.exists
    assert value.id is None
    assert not any("upload" in call for call in gateway.calls)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_finite_and_positive(settings, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        OpenShellSandbox(settings, settings.policy, timeout_s=timeout)


@pytest.mark.parametrize("entry", ["symlink", "fifo"])
async def test_unsafe_seed_is_rejected_before_any_cli_command(sandbox, tmp_path, entry):
    value, gateway = sandbox
    seed, private = directories(tmp_path)
    if entry == "symlink":
        (seed / "outside").symlink_to(tmp_path, target_is_directory=True)
    else:
        os.mkfifo(seed / "pipe")
    with pytest.raises(IsolationError, match="readable regular files"):
        await value.start(seed, private)
    assert gateway.calls == []
    assert value.name is None


@pytest.mark.parametrize("change", ["global", "widened"])
async def test_unexpected_authority_prevents_upload_and_is_deleted(
    sandbox, tmp_path, change
):
    value, gateway = sandbox
    if change == "global":
        gateway.policy_source = "global"
    else:
        gateway.extra_write = "/etc"
    with pytest.raises(IsolationError, match="requested sandbox policy"):
        await value.start(*directories(tmp_path))
    assert not gateway.exists
    assert not any("upload" in call for call in gateway.calls)


async def test_ambiguous_create_recovers_only_owned_sandbox(sandbox, tmp_path):
    value, gateway = sandbox
    gateway.create_error = True
    with pytest.raises(IsolationError, match="command failed"):
        await value.start(*directories(tmp_path))
    assert value.id is None
    assert not gateway.exists


async def test_spawn_quotes_arguments_and_excludes_daemon_environment(
    sandbox, tmp_path, monkeypatch
):
    value, gateway = sandbox
    await value.start(*directories(tmp_path))
    monkeypatch.setenv("OPENENVD_SECRET", "must-not-cross-boundary")
    process = SimpleNamespace(returncode=None)
    execute = AsyncMock(return_value=process)
    monkeypatch.setattr("asyncio.create_subprocess_exec", execute)
    argv = [value.settings.python, "-S", "worker.py", "name;$(not-a-command)"]
    env = {"PATH": "/usr/bin:/bin", "OPTION": "a 'quoted' value\nline"}
    assert await value.spawn(argv, env) is process
    invocation = execute.call_args
    remote = invocation.args[-1]
    assert remote.startswith(f"cd {WORKSPACE} && exec ")
    assert shlex.split(remote.split(" && exec ", 1)[1]) == [
        "/usr/bin/env",
        "-i",
        "PATH=/usr/bin:/bin",
        "OPTION=" + env["OPTION"],
        *argv,
    ]
    assert "SetEnv=OPENSHELL_NO_LOGIN_SHELL=1" in invocation.args
    assert "-T" in invocation.args
    assert invocation.kwargs["limit"] == 16 * 1024 * 1024
    assert "OPENENVD_SECRET" not in invocation.kwargs["env"]
    assert "must-not-cross-boundary" not in remote


def archive_bytes(*entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, kind, mode, content in entries:
            member = tarfile.TarInfo(name)
            member.type, member.mode = kind, mode
            member.size = len(content)
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "/outside/staging"
            archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


async def test_download_validates_archive_and_preserves_content_and_safe_modes(
    sandbox, tmp_path
):
    value, gateway = sandbox
    await value.start(*directories(tmp_path))
    destination = tmp_path / "download"
    destination.mkdir()
    archive = archive_bytes(
        (".", tarfile.DIRTYPE, 0o1750, b""),
        ("./nested/", tarfile.DIRTYPE, 0o750, b""),
        ("./nested/result.bin", tarfile.REGTYPE, 0o4751, b"result\0\xff"),
    )
    value._run_bytes = AsyncMock(return_value=archive)
    await value.download(destination)
    assert (destination / "nested/result.bin").read_bytes() == b"result\0\xff"
    assert (destination / "nested/result.bin").stat().st_mode & 0o7777 == 0o751
    assert destination.stat().st_mode & 0o7777 == 0o750
    assert (destination / "nested").stat().st_mode & 0o777 == 0o750
    invocation = value._run_bytes.call_args
    assert invocation.args[0][-1] == (
        f"cd {WORKSPACE} && exec /usr/bin/env -i /bin/tar -cf - ."
    )
    assert invocation.kwargs["output_limit"] == 64 * 1024 * 1024
    assert value._processes == []
    assert not any("download" in call for call in gateway.calls)


@pytest.mark.parametrize(
    "unsafe",
    [
        [("escape", tarfile.SYMTYPE, 0o777, b"")],
        [("escape", tarfile.LNKTYPE, 0o777, b"")],
        [("pipe", tarfile.FIFOTYPE, 0o600, b"")],
        [("sparse", tarfile.GNUTYPE_SPARSE, 0o600, b"")],
        [("../escape", tarfile.REGTYPE, 0o600, b"bad")],
        [("/escape", tarfile.REGTYPE, 0o600, b"bad")],
        [("./safe", tarfile.REGTYPE, 0o600, b"duplicate")],
        [("safe/nested", tarfile.REGTYPE, 0o600, b"file-parent")],
    ],
)
async def test_download_rejects_all_unsafe_members_before_writing(
    sandbox, tmp_path, unsafe
):
    value, _ = sandbox
    await value.start(*directories(tmp_path))
    destination = tmp_path / "download"
    destination.mkdir()
    archive = archive_bytes(("safe", tarfile.REGTYPE, 0o600, b"valid"), *unsafe)
    value._run_bytes = AsyncMock(return_value=archive)
    with pytest.raises(IsolationError, match="unsafe|conflict"):
        await value.download(destination)
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize(
    "pax_headers",
    [{"GNU.sparse.test": "unsafe"}, {"path": "bad\0path"}],
)
async def test_download_rejects_sparse_pax_metadata_and_null_names(
    sandbox, tmp_path, pax_headers
):
    value, _ = sandbox
    await value.start(*directories(tmp_path))
    destination = tmp_path / "download"
    destination.mkdir()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        member = tarfile.TarInfo("file")
        member.pax_headers = pax_headers
        archive.addfile(member)
    value._run_bytes = AsyncMock(return_value=output.getvalue())
    with pytest.raises(IsolationError, match="unsafe"):
        await value.download(destination)
    assert list(destination.iterdir()) == []


async def test_download_rejects_archive_limit_before_extraction(
    sandbox, tmp_path, monkeypatch
):
    value, _ = sandbox
    await value.start(*directories(tmp_path))
    destination = tmp_path / "download"
    destination.mkdir()
    monkeypatch.setattr(
        "openenv.core.openenvd.backends.openshell._WORKSPACE_ARCHIVE_LIMIT", 128
    )
    value._run_bytes = AsyncMock(return_value=b"x" * 129)
    with pytest.raises(IsolationError, match="archive limit"):
        await value.download(destination)
    assert list(destination.iterdir()) == []


async def test_download_rejects_member_limit_before_extraction(
    sandbox, tmp_path, monkeypatch
):
    value, _ = sandbox
    await value.start(*directories(tmp_path))
    destination = tmp_path / "download"
    destination.mkdir()
    monkeypatch.setattr(
        "openenv.core.openenvd.backends.openshell._WORKSPACE_MEMBER_LIMIT", 1
    )
    value._run_bytes = AsyncMock(
        return_value=archive_bytes(
            ("one", tarfile.REGTYPE, 0o600, b"1"),
            ("two", tarfile.REGTYPE, 0o600, b"2"),
        )
    )
    with pytest.raises(IsolationError, match="unsafe"):
        await value.download(destination)
    assert list(destination.iterdir()) == []


async def test_delete_waits_for_inventory_confirmation(sandbox, tmp_path, monkeypatch):
    value, gateway = sandbox
    await value.start(*directories(tmp_path))
    value._owned = AsyncMock(side_effect=[gateway.metadata(), gateway.metadata(), None])
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    await value.close()
    assert value._owned.await_count == 3
    assert value._closed
    assert value.id is None
    assert [call[8] for call in gateway.calls[-2:]] == ["stop", "delete"]


async def test_delete_failure_retains_ownership_for_retry(sandbox, tmp_path):
    value, gateway = sandbox
    await value.start(*directories(tmp_path))
    identity = value.name, value.id
    value._owned = AsyncMock(
        side_effect=[gateway.metadata(), IsolationError("gateway unavailable")]
    )
    with pytest.raises(IsolationError, match="gateway unavailable"):
        await value.close()
    assert (value.name, value.id) == identity
    assert not value._closed
    value._owned = AsyncMock(return_value=None)
    await value.close()
    assert value._closed


async def test_cleanup_refuses_a_different_sandbox_with_same_name(sandbox, tmp_path):
    value, gateway = sandbox
    await value.start(*directories(tmp_path))
    metadata = gateway.metadata()
    metadata["id"] = "replacement-sandbox"
    value._json = AsyncMock(
        return_value={"sandboxes": [metadata], "next_page_token": ""}
    )
    with pytest.raises(IsolationError, match="ownership"):
        await value.close()
    assert gateway.exists
    assert not any("delete" in call for call in gateway.calls)


async def test_command_timeout_kills_and_reaps_local_transport(settings):
    value = OpenShellSandbox(settings, settings.policy, timeout_s=0.1)
    original = value._create_process
    processes = []

    async def create(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        return process

    value._create_process = create
    with pytest.raises(IsolationError, match="timed out"):
        await value._run([sys.executable, "-c", "import time; time.sleep(30)"])
    assert len(processes) == 1
    assert processes[0].returncode is not None


async def test_command_cancellation_reaps_local_transport(settings):
    value = OpenShellSandbox(settings, settings.policy)
    original = value._create_process
    started = asyncio.Event()
    processes = []

    async def create(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        started.set()
        return process

    value._create_process = create
    task = asyncio.create_task(
        value._run([sys.executable, "-c", "import time; time.sleep(30)"])
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert processes[0].returncode is not None


async def test_excessive_output_is_bounded_and_transport_is_reaped(settings):
    value = OpenShellSandbox(settings, settings.policy)
    command = [
        sys.executable,
        "-c",
        "import os; chunk=b'x'*65536\nwhile True: os.write(1, chunk)",
    ]
    with pytest.raises(IsolationError, match="output limit"):
        await asyncio.wait_for(value._run(command), 5)


async def test_exited_cli_parent_does_not_leave_proxy_child_holding_pipes(settings):
    value = OpenShellSandbox(settings, settings.policy, timeout_s=0.2)
    original = value._create_process
    processes = []

    async def create(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        return process

    value._create_process = create
    command = [
        sys.executable,
        "-c",
        "import os,time\nif os.fork() == 0: time.sleep(30)\nelse: os._exit(0)",
    ]
    try:
        with pytest.raises(IsolationError, match="timed out"):
            await asyncio.wait_for(value._run_bytes(command), 3)
        assert processes[0].returncode == 0
        assert processes[0].stdout.at_eof()
        assert processes[0].stderr.at_eof()
    finally:
        for process in processes:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass


async def test_workspace_export_caps_stream_and_leaves_staging_empty(
    sandbox, tmp_path, monkeypatch
):
    value, _ = sandbox
    await value.start(*directories(tmp_path))
    destination = tmp_path / "download"
    destination.mkdir()
    monkeypatch.setattr(
        "openenv.core.openenvd.backends.openshell._WORKSPACE_ARCHIVE_LIMIT", 128
    )
    value._ssh_command = lambda _: [
        sys.executable,
        "-c",
        "import os; chunk=b'x'*65536\nwhile True: os.write(1, chunk)",
    ]
    with pytest.raises(IsolationError, match="output limit"):
        await asyncio.wait_for(value.download(destination), 5)
    assert list(destination.iterdir()) == []


async def test_sandbox_name_fits_openshell_limit_and_label_keeps_full_id(
    sandbox, tmp_path
):
    # OpenShell 0.1.2 rejects names longer than 19 characters.
    value, gateway = sandbox
    await value.start(*directories(tmp_path))
    assert len(value.name) <= 19
    assert value.name == "openenvd-" + value._label[:10]
    create = next(call for call in gateway.calls if "create" in call)
    assert create[create.index("--label") + 1] == "openenv-session=" + value._label


async def test_gateway_cli_errors_reach_the_daemon_log_not_the_exception(
    settings, caplog
):
    value = OpenShellSandbox(settings, settings.policy, timeout_s=10)
    value._cli = sys.executable
    argv = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('name exceeds maximum length (41 > 19)'); sys.exit(1)",
    ]
    with caplog.at_level("WARNING"):
        with pytest.raises(IsolationError) as error:
            await value._run(argv)
    assert "maximum length" not in str(error.value)
    assert "name exceeds maximum length (41 > 19)" in caplog.text


async def test_transport_errors_are_never_logged(settings, caplog):
    value = OpenShellSandbox(settings, settings.policy, timeout_s=10)
    value._cli = "/usr/bin/openshell"
    argv = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('workload secret'); sys.exit(1)",
    ]
    with caplog.at_level("WARNING"):
        with pytest.raises(IsolationError):
            await value._run(argv)
    assert "workload secret" not in caplog.text
