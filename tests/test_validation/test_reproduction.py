"""Regression checks for the shared acceptance harness."""

import hashlib
import importlib.util
import json
import subprocess
import sys
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
