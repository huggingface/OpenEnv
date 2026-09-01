import hashlib
import shutil

import pytest
from conftest import FIXTURES
from openenv.validation.policy import load_policy
from openenv.validation.report import ValidationReport
from openenv.validation.runner import run_validation, source_digest
from openenv.validation.signature import SignatureError
from openenv.validation.types import CheckStatus, Lane, Level, Verdict


def test_valid_package_passes_static_level():
    report = run_validation(
        FIXTURES / "served_min_pass", max_level=Level.STATIC, skip_build=True
    )
    assert report.verdict is Verdict.PASS
    assert report.lane is Lane.LOCAL
    assert report.levels_run == [Level.STATIC]
    assert report.manifest is not None and report.manifest.name == "served-min-pass"
    by_id = {r.check_id: r for r in report.results}
    assert by_id["static.manifest"].status is CheckStatus.PASS


def test_report_round_trips_through_its_schema():
    report = run_validation(
        FIXTURES / "served_min_pass", max_level=Level.STATIC, skip_build=True
    )
    assert ValidationReport.model_validate_json(report.model_dump_json()) == report


def test_broken_manifest_fails_static_manifest_with_evidence():
    report = run_validation(
        FIXTURES / "broken_manifest", max_level=Level.STATIC, skip_build=True
    )
    assert report.verdict is Verdict.FAIL
    assert report.manifest is None
    (result,) = report.results
    assert result.check_id == "static.manifest"
    assert result.status is CheckStatus.FAIL
    assert result.evidence, "schema errors must surface as evidence"


def test_out_of_bounds_declaration_fails_static_manifest(tmp_path):
    src = (FIXTURES / "served_min_pass" / "openenv.yaml").read_text()
    (tmp_path / "openenv.yaml").write_text(
        src.replace("floor_margin: 0.5", "floor_margin: 0.01")
    )
    report = run_validation(tmp_path, max_level=Level.STATIC, skip_build=True)
    assert report.verdict is Verdict.FAIL
    (result,) = report.results
    assert result.status is CheckStatus.FAIL
    assert "floor_margin" in "\n".join(result.evidence)


def test_ambiguous_package_raises_signature_error():
    with pytest.raises(SignatureError, match="ambiguous"):
        run_validation(FIXTURES / "ambiguous_package", max_level=Level.STATIC)


def test_report_embeds_the_pinned_policy_version():
    policy = load_policy("v1")
    report = run_validation(
        FIXTURES / "served_min_pass",
        max_level=Level.STATIC,
        skip_build=True,
        policy=policy,
    )
    assert report.policy_version == policy.policy_version


def test_source_digest_is_deterministic_and_content_sensitive(tmp_path):
    copy = tmp_path / "pkg"
    shutil.copytree(FIXTURES / "served_min_pass", copy)
    first = source_digest(copy)
    assert first == source_digest(copy)
    assert len(first) == 64
    (copy / "extra.txt").write_text("changed\n")
    assert source_digest(copy) != first


def test_source_digest_uses_portable_relative_paths(tmp_path):
    package_root = tmp_path / "pkg"
    nested = package_root / "nested"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_bytes(b"contents")

    expected = hashlib.sha256(b"nested/file.txt\0contents\0").hexdigest()
    assert source_digest(package_root) == expected
