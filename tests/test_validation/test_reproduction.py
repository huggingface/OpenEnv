"""Regression checks for the shared acceptance harness."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/validation" / f"{name}.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_linux_wheelhouse_markers_ignore_host_platform(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "jeepney==0.9.0 ; sys_platform == 'linux' \\\n    --hash=sha256:aaa\n"
        "pyobjc==11.0 ; sys_platform == 'darwin' \\\n    --hash=sha256:bbb\n"
    )
    module("reproduce").select_linux_requirements(requirements, "aarch64")
    assert requirements.read_text() == "jeepney==0.9.0 \\\n    --hash=sha256:aaa\n"


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize("minor", [31, 34, 39, 40])
def test_wheel_download_uses_target_image_compatibility(
    tmp_path, monkeypatch, arch, minor
):
    # The pinned runtime lab supplies pip and fails if any test is skipped.
    pytest.importorskip(
        "pip", reason="Wheel resolution requires the pinned runtime lab"
    )
    reproduction = module("reproduce")
    available = tmp_path / "available"
    available.mkdir()
    wheel = available / f"native_probe-1.0-cp312-cp312-manylinux_2_{minor}_{arch}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "native_probe-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: native-probe\nVersion: 1.0\n",
        )
        archive.writestr(
            "native_probe-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
            f"Tag: cp312-cp312-manylinux_2_{minor}_{arch}\n",
        )
        archive.writestr("native_probe-1.0.dist-info/RECORD", "")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        f"native-probe==1.0 --hash=sha256:{reproduction.digest(wheel)}\n"
    )
    # Only Docker is faked: real pip must resolve the hash-pinned native wheel
    # offline, including intermediate tags and a tag newer than the old list.
    target_tags = [f"manylinux_2_{version}_{arch}" for version in range(39, 16, -1)]
    target_tags += [f"manylinux2014_{arch}", f"linux_{arch}"]
    docker_platform = "linux/amd64" if arch == "x86_64" else "linux/arm64"
    base_image = "python:3.12-slim@sha256:target"
    run = reproduction.run

    def fake_docker(argv, **kwargs):
        if argv[0] == "docker":
            assert argv[argv.index("--platform") + 1] == docker_platform
            assert base_image in argv
            if argv[1] == "pull":
                return "Pulling layers...\nStatus: Downloaded newer image"
            assert "--pull=never" in argv
            assert "--network=none" in argv
            return json.dumps(target_tags)
        return run(argv, **kwargs)

    monkeypatch.setattr(reproduction, "run", fake_docker)
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(available))
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    wheelhouse = tmp_path / "wheelhouse"
    arguments = (requirements, wheelhouse, base_image, docker_platform)
    if minor == 40:
        with pytest.raises(RuntimeError, match="No matching distribution"):
            reproduction.download_linux_wheels(
                *arguments, log=tmp_path / "download.log"
            )
        assert not list(wheelhouse.glob("*.whl"))
    else:
        reproduction.download_linux_wheels(*arguments, log=tmp_path / "download.log")
        assert (wheelhouse / wheel.name).read_bytes() == wheel.read_bytes()


def evidence(tmp_path, skipped=0):
    reproduction = module("reproduce")
    (tmp_path / "run-manifest.json").write_text(json.dumps({"success": True}))
    (tmp_path / "junit.xml").write_text(
        f'<testsuites><testsuite tests="1" skipped="{skipped}"/></testsuites>'
    )
    entries = reproduction.hashes(tmp_path)
    (tmp_path / "SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in entries.items())
    )


def test_artifact_verification_rejects_missing_or_added_evidence(tmp_path):
    evidence(tmp_path)
    verify = module("verify_artifacts").verify
    assert verify(tmp_path) == 1
    (tmp_path / "unrecorded.txt").write_text("unrecorded")
    with pytest.raises(ValueError, match="inventory"):
        verify(tmp_path)
    (tmp_path / "unrecorded.txt").unlink()
    (tmp_path / "junit.xml").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path)


def test_required_acceptance_cannot_silently_skip(tmp_path):
    evidence(tmp_path, skipped=1)
    with pytest.raises(ValueError, match="skips"):
        module("verify_artifacts").verify(tmp_path)


def test_nested_case_checksum_is_itself_covered_by_bundle(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "SHA256SUMS").write_text("case-owned checksums")
    evidence(tmp_path)
    assert module("verify_artifacts").verify(tmp_path) == 1


def test_command_drains_large_output_without_unbounded_retention(tmp_path):
    reproduction = module("reproduce")
    log = tmp_path / "command.log"
    result = reproduction.run(
        [sys.executable, "-c", "import sys; sys.stdout.write('x' * 3000000)"],
        log=log,
    )
    assert len(result) == reproduction.MAX_LOG_BYTES
    assert log.stat().st_size == reproduction.MAX_LOG_BYTES


def test_command_digest_hashes_complete_unsanitized_bytes():
    reproduction = module("reproduce")
    payload = b" leading\n" + b"x" * 1_100_000 + b" ghp_abcdefgh\n"
    observed = reproduction.command_digest(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.stdout.buffer.write("
                "b' leading\\n' + b'x' * 1_100_000 + b' ghp_abcdefgh\\n')"
            ),
        ]
    )
    assert observed == hashlib.sha256(payload).hexdigest()


def test_command_timeout_retains_partial_log(tmp_path):
    log = tmp_path / "command.log"
    with pytest.raises(subprocess.TimeoutExpired):
        module("reproduce").run(
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(10)",
            ],
            timeout=0.2,
            log=log,
        )
    assert log.read_text() == "started\n"
