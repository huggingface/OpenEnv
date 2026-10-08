import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import FIXTURES
from openenv.validation.report import ValidationReport, ValidationReportV2

REPO_ROOT = Path(__file__).parent.parent.parent


def _validate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "openenv.cli", "validate", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_valid_package_exits_zero():
    result = _validate(
        str(FIXTURES / "served_min_pass"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verdict: PASS" in result.stdout


def test_echo_env_exits_zero():
    result = _validate(
        str(REPO_ROOT / "envs" / "echo_env"),
        "--level",
        "static",
        "--skip-build",
        "--json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = ValidationReport.model_validate_json(result.stdout)
    assert report.verdict.value == "pass"
    assert any(r.check_id == "static.manifest" for r in report.results)


def test_json_report_is_schema_valid():
    result = _validate(
        str(FIXTURES / "served_min_pass"), "--level", "static", "--skip-build", "--json"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = ValidationReport.model_validate_json(result.stdout)
    assert report.verdict.value == "pass"
    assert report.manifest.openenvd.enabled is False


@pytest.mark.parametrize(
    "config",
    [
        None,
        {"surfaces": {"agent": {"tools": ["grader.*"]}}},
        {"surfaces": {"agent": {"allow_lifecycle": True}}},
        {"surfaces": {"agent": {"fs_read": ["/openenvd/assets/**"]}}},
    ],
)
def test_invalid_openenvd_policy_fails_static_manifest(tmp_path, config):
    src = (FIXTURES / "served_min_pass" / "openenv.yaml").read_text()
    (tmp_path / "openenv.yaml").write_text(src + yaml.safe_dump({"openenvd": config}))
    result = _validate(str(tmp_path), "--level", "static", "--skip-build", "--json")
    assert result.returncode == 1, result.stdout + result.stderr
    report = ValidationReport.model_validate_json(result.stdout)
    failure = next(r for r in report.results if r.check_id == "static.manifest")
    assert failure.status.value == "fail"
    assert any("openenvd" in evidence for evidence in failure.evidence)
    assert "openenvd" in failure.remediation


@pytest.mark.parametrize("schema_version", ["1", "2"])
def test_json_report_preserves_openenvd_policy_without_importing_app(
    tmp_path, schema_version
):
    source = yaml.safe_load((FIXTURES / "served_min_pass" / "openenv.yaml").read_text())
    config = {
        "enabled": True,
        "openshell": {
            "image": "openenv-echo:latest",
            "gateway": "training",
            "workspace": "episodes",
            "python": "/opt/venv/bin/python",
            "policy": {
                "version": 1,
                "filesystem_policy": {
                    "include_workdir": False,
                    "read_only": ["/usr", "/opt"],
                    "read_write": ["/sandbox", "/tmp", "/dev/null"],
                },
                "landlock": {"compatibility": "hard_requirement"},
                "process": {"run_as_user": "1000", "run_as_group": "1000"},
                "network_policies": {},
            },
        },
        "surfaces": {
            "agent": {"tools": ["env.*"]},
            "grader": {"tools": ["grader.*"], "allow_privileged_exec": True},
            "orchestrator": {"allow_lifecycle": True},
            "observer": {"stream": ["process", "fs_diff"]},
        },
        "privileged_assets": {"oracle": "solution/oracle.py"},
    }
    source["openenvd"] = config
    if schema_version == "2":
        source["validation"]["execution"] = {"kind": "openenv_ws"}
    (tmp_path / "openenv.yaml").write_text(yaml.safe_dump(source))
    marker = tmp_path / "imported"
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "raise RuntimeError('validation imported package code')\n"
    )
    result = _validate(str(tmp_path), "--level", "static", "--skip-build", "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    report_model = ValidationReportV2 if schema_version == "2" else ValidationReport
    report = report_model.model_validate_json(result.stdout)
    assert report.manifest.manifest_schema_version == schema_version
    normalized = report.manifest.openenvd.model_dump(mode="json")
    assert normalized["enabled"] is True
    assert {key: normalized["openshell"][key] for key in config["openshell"]} == config[
        "openshell"
    ]
    assert normalized["privileged_assets"] == config["privileged_assets"]
    for principal, policy in config["surfaces"].items():
        assert normalized["surfaces"][principal]["principal"] == principal
        for permission, value in policy.items():
            assert normalized["surfaces"][principal][permission] == value
    assert not marker.exists()


def test_output_writes_the_json_report(tmp_path):
    out = tmp_path / "report.json"
    result = _validate(
        str(FIXTURES / "served_min_pass"),
        "--level",
        "static",
        "--skip-build",
        "--output",
        str(out),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ValidationReport.model_validate_json(out.read_text())


def test_output_write_failure_is_an_internal_error(tmp_path):
    out = tmp_path / "missing" / "report.json"
    result = _validate(
        str(FIXTURES / "served_min_pass"),
        "--level",
        "static",
        "--skip-build",
        "--output",
        str(out),
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "Internal error:" in result.stderr


def test_failing_manifest_exits_one():
    result = _validate(
        str(FIXTURES / "broken_manifest"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "static.manifest" in result.stdout


def test_ambiguous_package_exits_two():
    result = _validate(
        str(FIXTURES / "ambiguous_package"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "ambiguous" in result.stderr


def test_unrecognized_package_exits_two():
    result = _validate(
        str(FIXTURES / "unrecognized_package"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "unrecognized" in result.stderr


def test_format_without_a_parser_exits_two():
    result = _validate(
        str(FIXTURES / "harbor_task_min"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "unrecognized" in result.stderr


def test_nonexistent_path_exits_two(tmp_path):
    result = _validate(str(tmp_path / "nope"))
    assert result.returncode == 2, result.stdout + result.stderr


def test_unknown_level_is_an_internal_error():
    result = _validate(str(FIXTURES / "served_min_pass"), "--level", "cosmic")
    assert result.returncode == 3, result.stdout + result.stderr


def test_unknown_policy_is_an_internal_error():
    result = _validate(str(FIXTURES / "served_min_pass"), "--policy", "nope")
    assert result.returncode == 3, result.stdout + result.stderr


def test_unpinned_judge_fails_static_manifest():
    result = _validate(
        str(FIXTURES / "unpinned_judge"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 1
    assert "judge pin" in result.stdout
