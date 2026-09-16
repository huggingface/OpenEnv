"""Installed-wheel CLI acceptance: real builds, wire faults, reports and cleanup."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import jsonschema
import pytest


pytestmark = pytest.mark.docker

IMPLEMENTED = {
    "runtime.startup",
    "runtime.reward_well_formed",
    "runtime.observation_schema",
    "runtime.state_contract",
}
PENDING = {
    "runtime.trajectory_record",
    "runtime.tool_declaration_accuracy",
    "runtime.seed_control",
    "runtime.episode_determinism",
    "runtime.network_policy",
    "runtime.host_containment",
    "runtime.resource_bounds",
    "runtime.episode_isolation",
    "runtime.oracle_containment",
}
NOT_APPLICABLE = {
    "runtime.rubric_introspectable",
    "runtime.reward_attribution",
    "runtime.task_declaration_accuracy",
}


def _copy_asset(source, destination):
    # Wheel bytes are immutable inputs; hardlinks avoid copying the locked
    # wheelhouse for every fault case. Fall back across filesystem boundaries.
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


@pytest.fixture
def cli_context(tmp_path):
    root = os.environ.get("OPENENV_VALIDATION_CONTEXT")
    if not root:
        if os.environ.get("OPENENV_REQUIRE_DOCKER") == "1":
            pytest.fail("Required CLI acceptance needs OPENENV_VALIDATION_CONTEXT")
        pytest.skip("Run the validation lab to provide its offline build context")
    context = tmp_path / "subject"
    shutil.copytree(root, context, copy_function=_copy_asset)
    # Fault injection must never modify the shared context's Dockerfile inode.
    dockerfile = context / "Dockerfile"
    content = dockerfile.read_bytes()
    dockerfile.unlink()
    dockerfile.write_bytes(content)
    return context


def _container_ids():
    checked = subprocess.run(
        [
            "docker",
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            "label=org.openenv.validation.run",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return set(checked.stdout.split())


def _verify_bundle(bundle, report):
    schema = json.loads(
        files("openenv.validation")
        .joinpath("schemas/report-v2.schema.json")
        .read_text()
    )
    jsonschema.validate(report, schema)
    assert report["report_schema_version"] == "2"
    assert report["manifest"]["manifest_schema_version"] == "2"
    assert report["policy_version"] == "v2"
    assert report["lane"] == "local"
    assert len(report["source_digest"]) == 64
    artifact_report = json.loads((bundle / "report.json").read_text())
    assert artifact_report == report
    recorded = set()
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name not in recorded
        recorded.add(name)
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest
    assert recorded == {
        path.name for path in bundle.iterdir() if path.name != "SHA256SUMS"
    }
    return {name: json.loads((bundle / name).read_text()) for name in recorded}


def _invoke_cli(context, tmp_path, case, *, skip_build=False):
    artifact_root = Path(os.environ.get("OPENENV_VALIDATION_ARTIFACTS", tmp_path))
    work = artifact_root / "cli" / case
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / "report.json"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    marker = tmp_path / "unexpected-docker-call"
    if skip_build:
        binary_dir = tmp_path / "bin"
        binary_dir.mkdir()
        docker = binary_dir / "docker"
        docker.write_text(
            f"#!{sys.executable}\nfrom pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('called')\nraise SystemExit(91)\n"
        )
        docker.chmod(0o755)
        environment["PATH"] = str(binary_dir) + os.pathsep + environment["PATH"]
    command = [
        sys.executable,
        "-m",
        "openenv.cli",
        "validate",
        str(context),
        "--level",
        "runtime",
        "--local",
        "--policy",
        "v2",
        "--json",
        "--output",
        str(report_path),
    ]
    if skip_build:
        command.append("--skip-build")
    before = _container_ids()
    try:
        result = subprocess.run(
            command,
            cwd=work,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
    finally:
        remaining = _container_ids() - before
        (work / "cleanup-external.json").write_text(
            json.dumps({"remaining_container_ids": sorted(remaining)}) + "\n"
        )
        assert not remaining, "CLI validation leaked a container"
    (work / "stdout.json").write_text(result.stdout)
    (work / "stderr.log").write_text(result.stderr)
    assert report_path.is_file(), result.stderr
    report = json.loads(report_path.read_text())
    assert json.loads(result.stdout) == report
    artifacts = _verify_bundle(work / "report.artifacts", report)
    assert not marker.exists(), "--skip-build invoked Docker"
    checks = {row["check_id"]: row for row in report["results"]}
    assert len(checks) == len(report["results"]), "Report contains duplicate check IDs"
    assert {
        key for key in checks if key.startswith("runtime.")
    } == IMPLEMENTED | PENDING
    assert not NOT_APPLICABLE & checks.keys()
    assert all(checks[key]["status"] == "skip" for key in PENDING)
    assert all(checks[key]["evidence"] for key in PENDING)
    assert (
        set(artifacts["coverage.json"]["requested_runtime_checks"])
        == IMPLEMENTED | PENDING
    )
    assert artifacts["cleanup.json"]["completed"] is True
    return result, report, checks, artifacts


@pytest.mark.parametrize(
    "mode,failed_check",
    [
        ("good", None),
        ("bad_reward", "runtime.reward_well_formed"),
        ("bad_observation", "runtime.observation_schema"),
        ("bad_state", "runtime.state_contract"),
    ],
)
def test_cli_runtime_contract_findings(cli_context, tmp_path, mode, failed_check):
    with (cli_context / "Dockerfile").open("a") as stream:
        stream.write(f"\nENV VALIDATION_FAULT={mode}\n")
    result, report, checks, artifacts = _invoke_cli(cli_context, tmp_path, mode)
    assert report["levels_run"] == [1, 2]
    assert checks["static.manifest"]["status"] == "pass"
    assert checks["runtime.startup"]["status"] == "pass"
    assert checks["runtime.startup"]["measured"]["image_ref"].startswith("sha256:")
    assert artifacts["cleanup.json"]["required"] is True
    assert artifacts["run-manifest.json"]["provider"]["container_id"]
    assert artifacts["run-manifest.json"]["source_digest"] == report["source_digest"]
    trace = artifacts["collector-trace.json"]
    assert [row["operation"] for row in trace] == [
        "reset",
        "state",
        "step",
        "state",
        "step",
        "state",
    ]
    if failed_check:
        assert result.returncode == 1
        assert report["verdict"] == "fail"
        assert checks[failed_check]["status"] == "fail"
        assert checks[failed_check]["evidence"]
    else:
        assert result.returncode == 0
        assert report["verdict"] == "warn", (
            "Missing later graders must keep the result incomplete"
        )
        assert all(checks[key]["status"] == "pass" for key in IMPLEMENTED)
        assert set(artifacts["coverage.json"]["incomplete"]) == PENDING


def test_cli_startup_failure_is_a_finding_and_leaves_no_container(
    cli_context, tmp_path
):
    with (cli_context / "Dockerfile").open("a") as stream:
        stream.write("\nENV VALIDATION_FAULT=startup_failure\n")
    result, report, checks, artifacts = _invoke_cli(
        cli_context, tmp_path, "startup_failure"
    )
    assert result.returncode == 1
    assert report["verdict"] == "fail"
    assert report["levels_run"] == [1, 2]
    assert checks["runtime.startup"]["status"] == "fail"
    assert all(
        checks[key]["status"] == "skip" for key in IMPLEMENTED - {"runtime.startup"}
    )
    assert artifacts["collector-trace.json"] == []


def test_cli_skip_build_does_not_invoke_docker(cli_context, tmp_path):
    result, report, checks, artifacts = _invoke_cli(
        cli_context, tmp_path, "skip_build", skip_build=True
    )
    assert result.returncode == 0
    assert report["verdict"] == "warn"
    assert report["levels_run"] == [1]
    assert all(checks[key]["status"] == "skip" for key in IMPLEMENTED | PENDING)
    assert "--skip-build" in " ".join(checks["runtime.startup"]["evidence"])
    assert artifacts["cleanup.json"]["required"] is False
    assert artifacts["run-manifest.json"]["provider"] == {}
    assert artifacts["collector-trace.json"] == []
