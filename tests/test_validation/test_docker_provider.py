import json
import sys
import time
import urllib.error
from pathlib import Path

import pytest
from openenv.validation.manifest import (
    ExecutionDeclaration,
    NetworkPolicy,
    ResourceDeclaration,
)
from openenv.validation.providers import (
    docker,
    ProviderError,
    StartupError,
    UnsupportedCapability,
)
from openenv.validation.runtime.contracts import LaunchSpec


IMAGE = "sha256:" + "a" * 64


def launch_spec(**kwargs):
    return LaunchSpec(
        image_ref=IMAGE,
        run_id="test-run",
        network=kwargs.pop("network", NetworkPolicy()),
        resources=kwargs.pop(
            "resources",
            ResourceDeclaration(
                cpu=0.5,
                memory_mb=128,
                disk_mb=32,
                episode_timeout_s=5,
            ),
        ),
        **kwargs,
    )


def details(name="test", run_id="test-run", mounts=None):
    return {
        "Id": name,
        "Image": IMAGE,
        "Config": {
            "Labels": {docker._LABEL: run_id},
            "User": "65532:65532",
            "Env": ["TOKEN=private"],
        },
        "State": {"Running": True, "OOMKilled": False},
        "HostConfig": {"Memory": 128 * 1024**2, "ReadonlyRootfs": True},
        "Mounts": mounts or [],
        "NetworkSettings": {
            "Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "49231"}]}
        },
    }


class Healthy:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def open(self, *args, **kwargs):
        return self


@pytest.fixture
def commands(monkeypatch):
    calls = []
    removed = set()

    def run(argv, timeout_s, max_bytes=docker._MAX_OUTPUT):
        calls.append(argv)
        if argv[1] == "inspect":
            if argv[-1] in removed:
                return 1, "", "No such container"
            if "--format" in argv:
                return 0, "test-run", ""
            return 0, json.dumps([details(argv[-1])]), ""
        if argv[1] == "rm":
            removed.add(argv[-1])
        return 0, "ok", ""

    monkeypatch.setattr(docker, "_command", run)
    monkeypatch.setattr(docker.urllib.request, "build_opener", lambda *a: Healthy())
    return calls


def test_start_is_hardened_and_does_not_forward_host_tokens(commands, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_not_forwarded_to_subject")
    subject = docker.DockerValidationProvider().start(
        launch_spec(env_vars={"MODE": "probe"})
    )
    create = commands[0]
    assert subject.base_url == "http://127.0.0.1:49231"
    for flag, value in {
        "--user": "65532:65532",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges",
        "--memory": "128m",
        "--memory-swap": "128m",
        "--cpus": "0.5",
        "--pids-limit": "256",
        "--publish": "127.0.0.1::8000",
        "--network": "bridge",
    }.items():
        assert create[create.index(flag) + 1] == value
    assert "--read-only" in create and "--init" in create
    assert "MODE=probe" in create
    assert not any("HF_TOKEN" in arg or "not_forwarded" in arg for arg in create)
    assert create[-1] == IMAGE
    tmpfs = create[create.index("--tmpfs") + 1]
    assert "size=16777216" in tmpfs
    assert int(create[create.index("--shm-size") + 1]) == 16777216
    subject.stop()
    subject.stop()
    assert sum(call[1] == "rm" for call in commands) == 1
    assert all("prune" not in call for call in commands)


@pytest.mark.parametrize(
    "network",
    [
        NetworkPolicy(mode="no-network"),
        NetworkPolicy(mode="allowlist", allowed_hosts=["example.org"]),
    ],
)
def test_unsupported_network_is_rejected_without_docker(commands, network):
    with pytest.raises(UnsupportedCapability):
        docker.DockerValidationProvider().start(launch_spec(network=network))
    assert not commands


def test_gpu_is_rejected_without_docker(commands):
    spec = launch_spec(
        resources=ResourceDeclaration(
            cpu=1, memory_mb=128, disk_mb=32, episode_timeout_s=5, gpus=1
        )
    )
    with pytest.raises(UnsupportedCapability, match="GPU"):
        docker.DockerValidationProvider().start(spec)
    assert not commands


def test_startup_timeout_removes_container(commands, monkeypatch):
    class Unhealthy:
        def open(self, *args, **kwargs):
            raise urllib.error.URLError("refused")

    monkeypatch.setattr(docker.urllib.request, "build_opener", lambda *a: Unhealthy())
    with pytest.raises(StartupError, match="deadline"):
        docker.DockerValidationProvider().start(launch_spec(startup_timeout_s=0.02))
    assert commands[-2][1:4] == ["rm", "--force", "--volumes"]


def test_image_volumes_are_rejected_before_execution(commands, monkeypatch):
    removed = False

    def run(argv, *args):
        nonlocal removed
        commands.append(argv)
        if argv[1] == "inspect":
            if removed:
                return 1, "", "No such container"
            if "--format" in argv:
                return 0, "test-run", ""
            return 0, json.dumps([details(mounts=[{"Type": "volume"}])]), ""
        if argv[1] == "rm":
            removed = True
        return 0, "ok", ""

    monkeypatch.setattr(docker, "_command", run)
    with pytest.raises(StartupError, match="mounts"):
        docker.DockerValidationProvider().start(launch_spec())
    assert not any(call[1] == "start" for call in commands)
    assert commands[-2][1] == "rm"


def test_create_failure_still_attempts_owned_cleanup(commands, monkeypatch):
    removed = False

    def run(argv, *args):
        nonlocal removed
        commands.append(argv)
        if argv[1] == "create":
            raise ProviderError("deadline")
        if argv[1] == "inspect":
            if removed:
                return 1, "", "No such container"
            if "--format" in argv:
                return 0, "test-run", ""
            return 0, json.dumps([details(argv[-1])]), ""
        if argv[1] == "rm":
            removed = True
        return 0, "", ""

    monkeypatch.setattr(docker, "_command", run)
    with pytest.raises(StartupError):
        docker.DockerValidationProvider().start(launch_spec())
    assert commands[-2][1] == "rm"
    assert commands[0][commands[0].index("--name") + 1] == commands[-1][-1]


def test_inspection_reports_effective_settings_not_environment(commands):
    subject = docker.DockerValidationProvider().start(
        launch_spec(env_vars={"OPTION": "1"})
    )
    observed = subject.inspect()
    assert observed["limits"]["Memory"] == 128 * 1024**2
    assert observed["limits"]["ReadonlyRootfs"] is True
    assert "TOKEN" not in json.dumps(observed)
    subject.stop()


def test_exec_timeout_destroys_subject_to_stop_descendants(commands, monkeypatch):
    subject = docker.DockerValidationProvider().start(launch_spec())
    original = docker._command

    def timeout_exec(argv, *args):
        if argv[1] == "exec":
            raise ProviderError("deadline")
        return original(argv, *args)

    monkeypatch.setattr(docker, "_command", timeout_exec)
    with pytest.raises(ProviderError, match="deadline"):
        subject.exec(["python", "-c", "while True: pass"], 0.1)
    assert commands[-2][1] == "rm"
    with pytest.raises(ProviderError, match="stopped"):
        subject.exec(["true"], 1)


def test_exec_timeout_is_preserved_when_cleanup_verification_fails(
    commands, monkeypatch
):
    subject = docker.DockerValidationProvider().start(launch_spec())
    original = docker._command
    timeout = ProviderError("Docker operation exceeded its deadline")

    def fail_exec_and_verification(argv, *args):
        if argv[1] == "exec":
            raise timeout
        if argv[1:4] == ["inspect", "--format", "{{.Id}}"]:
            return 1, "", "Docker daemon unavailable"
        return original(argv, *args)

    monkeypatch.setattr(docker, "_command", fail_exec_and_verification)
    with pytest.raises(ProviderError, match="independently verified") as error:
        subject.exec(["sleep", "20"], 0.1)
    assert error.value.__cause__ is timeout
    assert commands[-1][1:4] == ["rm", "--force", "--volumes"]
    assert not subject._stopped


def test_exec_preserves_argv_and_redacts_credentials(commands, monkeypatch):
    subject = docker.DockerValidationProvider().start(
        launch_spec(env_vars={"API_KEY": "private-value"})
    )

    def run(argv, *args):
        commands.append(argv)
        return 7, "private-value hf_abcdefgh12345", "token=bad-secret"

    monkeypatch.setattr(docker, "_command", run)
    result = subject.exec(["printf", "$(cat /host/key); anything"], 1)
    assert result.exit_code == 7
    assert commands[-1][-1] == "$(cat /host/key); anything"
    assert "private-value" not in result.stdout and "hf_" not in result.stdout
    assert "bad-secret" not in result.stderr


def test_build_uses_filtered_snapshot_and_returns_immutable_image(
    tmp_path, monkeypatch
):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "data.txt").write_text("source")
    for name in (".git", ".venv", "outputs"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "private").write_text("not copied")
    (tmp_path / ".env").write_text("HF_TOKEN=private")
    (tmp_path / "key.pem").write_text("private")
    for name in ("secret", "secrets", "credentials"):
        (tmp_path / name).write_text("private")
    (tmp_path / "secret.py").write_text("VALUE = 'source'\n")
    (tmp_path / "credentials.py").write_text("VALUE = 'source'\n")

    def build(argv, timeout_s):
        assert timeout_s <= 600
        context = Path(argv[-1])
        assert set(path.name for path in context.iterdir()) == {
            "Dockerfile",
            "credentials.py",
            "data.txt",
            "secret.py",
        }
        assert context != tmp_path
        Path(argv[argv.index("--iidfile") + 1]).write_text(IMAGE)
        return 0, "", ""

    monkeypatch.setattr(docker, "_command", build)
    assert (
        docker.DockerValidationProvider().build(tmp_path, ExecutionDeclaration())
        == IMAGE
    )


def test_build_rejects_symlink_escape_without_invoking_docker(tmp_path, commands):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ProviderError, match="symbolic"):
        docker.DockerValidationProvider().build(tmp_path, ExecutionDeclaration())
    assert not commands


def test_build_rejects_oversized_snapshot(tmp_path, commands):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    with pytest.raises(ProviderError, match="budget"):
        docker.DockerValidationProvider(max_context_bytes=1).build(
            tmp_path, ExecutionDeclaration()
        )
    assert not commands


def test_build_failure_is_bounded_and_sanitized(tmp_path, monkeypatch):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    monkeypatch.setattr(
        docker, "_command", lambda *a: (1, "", "x" * 8000 + " hf_1234567890123456")
    )
    with pytest.raises(ProviderError) as error:
        docker.DockerValidationProvider().build(tmp_path, ExecutionDeclaration())
    assert len(str(error.value)) < 4200
    assert "hf_123456" not in str(error.value)


def test_command_drains_large_stdout_and_stderr_with_bounded_memory():
    code, stdout, stderr = docker._command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('a'*200000); sys.stderr.write('b'*200000)",
        ],
        5,
        max_bytes=1024,
    )
    assert code == 0 and stdout == "a" * 1024 and stderr == "b" * 1024


def test_command_kills_timed_out_process_group():
    started = time.monotonic()
    with pytest.raises(ProviderError, match="deadline"):
        docker._command([sys.executable, "-c", "import time; time.sleep(20)"], 0.05)
    assert time.monotonic() - started < 3


def test_readiness_never_follows_redirects():
    handler = docker._NoRedirects()
    assert handler.redirect_request(None, None, 302, "", {}, "http://private/") is None


def test_cleanup_refuses_a_container_with_another_owner(commands, monkeypatch):
    subject = docker.DockerValidationProvider().start(launch_spec())
    monkeypatch.setattr(
        docker,
        "_command",
        lambda *a: (0, json.dumps([details(run_id="other-run")]), ""),
    )
    with pytest.raises(ProviderError, match="another owner"):
        subject.stop()
    assert not any(call[1] == "rm" for call in commands)


def test_logs_remain_byte_bounded_after_redaction_expands_short_values(
    commands, monkeypatch
):
    subject = docker.DockerValidationProvider().start(
        launch_spec(env_vars={"KEY": "x"})
    )
    monkeypatch.setattr(docker, "_command", lambda *a: (0, "x" * 32, ""))
    output = subject.logs(max_bytes=32)
    assert len(output.encode("utf-8")) <= 32
    assert "x" not in output


def test_large_image_metadata_cannot_prevent_cleanup_after_inspection_failure(
    monkeypatch,
):
    calls = []
    removed = False

    def run(argv, *args):
        nonlocal removed
        calls.append(argv)
        if argv[1] == "inspect":
            if removed:
                return 1, "", "No such container"
            if "--format" in argv:
                return 0, "test-run\n", ""
            # Simulate the retained tail after image metadata exceeds 64 KiB.
            return 0, "x" * docker._MAX_OUTPUT, ""
        if argv[1] == "rm":
            removed = True
        return 0, "created", ""

    monkeypatch.setattr(docker, "_command", run)
    with pytest.raises(StartupError, match="invalid inspection evidence"):
        docker.DockerValidationProvider().start(launch_spec())
    assert removed
    assert calls[-1][1:4] == ["inspect", "--format", "{{.Id}}"]
