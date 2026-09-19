"""Runtime scheduling, capability preflight and lifetime regression tests."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from openenv.validation.policy import load_policy, PolicyError
from openenv.validation.providers import StartupError
from openenv.validation.report import CheckResult
from openenv.validation.runner import run_validation, source_digest
from openenv.validation.runtime.artifacts import write_runtime_bundle
from openenv.validation.runtime.scheduler import execute_graders, order_graders
from openenv.validation.types import CheckStatus, Level, ProviderCapability
from support.runtime import evidence, exchange, FakeRuntimeProvider


FIXTURE = Path(__file__).parents[1] / "fixtures/validation/runtime/served_probe"


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "subject"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("__pycache__"))
    return root


def measured_episode():
    rows = []
    for step in (0, 1):
        operation = "reset" if step == 0 else "step"
        rows.append(
            exchange(
                operation,
                {"data": {"episode_id": "validation-probe"}},
                {
                    "type": "observation",
                    "data": {
                        "observation": {"counter": step},
                        "reward": float(step),
                        "done": bool(step),
                    },
                },
            )
        )
        rows.append(
            exchange(
                "state",
                {"type": "state"},
                {
                    "type": "state",
                    "data": {"episode_id": "validation-probe", "step_count": step},
                },
            )
        )
    return evidence(
        *rows,
        observation_schema={
            "type": "object",
            "required": ["counter", "reward", "done"],
            "properties": {
                "counter": {"type": "integer"},
                "done": {"type": "boolean"},
                "reward": {"type": "number"},
            },
        },
    )


def test_runtime_collects_once_cleans_up_and_marks_remaining_work(
    package, monkeypatch, tmp_path
):
    provider = FakeRuntimeProvider()
    calls = []

    def collect(*args, **kwargs):
        calls.append((args, kwargs))
        return measured_episode()

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    bundle = tmp_path / "bundle"
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
    )
    results = {r.check_id: r.status for r in report.results}
    assert len(calls) == 1
    assert len(provider.builds) == len(provider.launches) == 1
    assert provider.subject.stopped
    assert report.report_schema_version == "2"
    assert report.levels_run == [Level.STATIC, Level.RUNTIME]
    assert report.verdict.value == "warn"
    for name in (
        "startup",
        "reward_well_formed",
        "observation_schema",
        "state_contract",
    ):
        assert results[f"runtime.{name}"] is CheckStatus.PASS
    assert results["runtime.network_policy"] is CheckStatus.SKIP
    assert json.loads((bundle / "cleanup.json").read_text())["completed"] is True
    assert json.loads((bundle / "runtime-plan.json").read_text())["reset"]["seed"] == 42


def test_skip_build_has_no_provider_side_effects(package):
    provider = FakeRuntimeProvider()
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, skip_build=True
    )
    assert not provider.builds and not provider.launches
    assert report.levels_run == [Level.STATIC]
    assert all(
        r.status is CheckStatus.SKIP
        for r in report.results
        if r.check_id.startswith("runtime.")
    )


@pytest.mark.parametrize("change", ["network", "gpu", "build"])
def test_unsupported_capability_is_refused_before_build(package, change):
    import yaml

    source = package / "openenv.yaml"
    data = yaml.safe_load(source.read_text())
    provider = FakeRuntimeProvider()
    if change == "network":
        data["validation"]["network"] = {"mode": "no-network"}
    elif change == "gpu":
        data["validation"]["resources"]["gpus"] = 1
    else:
        provider.capabilities = frozenset()
    source.write_text(yaml.safe_dump(data))
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    assert not provider.builds and not provider.launches
    assert (
        next(r for r in report.results if r.check_id == "runtime.startup").status
        is CheckStatus.SKIP
    )


def test_explicit_v1_rejected_before_runtime(package):
    provider = FakeRuntimeProvider()
    with pytest.raises(PolicyError, match="policy v2"):
        run_validation(
            package,
            max_level=Level.RUNTIME,
            policy=load_policy("v1"),
            provider=provider,
        )
    assert not provider.builds


def test_invalid_plan_is_visible_failure(package):
    (package / "validation/runtime.json").write_text('{"actions": []}')
    provider = FakeRuntimeProvider()
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    assert report.verdict.value == "fail"
    assert not provider.builds


def test_startup_failure_does_not_masquerade_as_skips(package):
    provider = FakeRuntimeProvider()

    def failed_build(*args):
        raise StartupError("subject build failed")

    provider.build = failed_build
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    result = next(r for r in report.results if r.check_id == "runtime.startup")
    assert result.status is CheckStatus.FAIL
    assert report.verdict.value == "fail"


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_collector_crash_or_cancel_always_tears_down(package, monkeypatch, failure):
    provider = FakeRuntimeProvider()

    def explode(*args, **kwargs):
        raise failure("token=must-not-appear")

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", explode)
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    assert provider.subject.stopped
    assert report.verdict.value == "fail"
    assert "must-not-appear" not in report.model_dump_json()


def test_bad_static_bounds_do_not_suppress_independent_runtime(package, monkeypatch):
    path = package / "openenv.yaml"
    path.write_text(path.read_text().replace("floor_margin: 0.5", "floor_margin: 0.01"))
    provider = FakeRuntimeProvider()
    monkeypatch.setattr(
        "openenv.validation.runner.collect_runtime_evidence",
        lambda *a, **k: measured_episode(),
    )
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    assert report.results[0].status is CheckStatus.FAIL
    assert (
        next(r for r in report.results if r.check_id == "runtime.state_contract").status
        is CheckStatus.PASS
    )


def test_semantic_ceiling_does_not_claim_semantic_execution(package):
    report = run_validation(package, max_level=Level.SEMANTIC, skip_build=True)
    assert report.levels_run == [Level.STATIC]
    assert any(
        r.check_id == "semantic.oracle_max" and r.status is CheckStatus.SKIP
        for r in report.results
    )


def grader(check_id, depends_on=(), *, status=CheckStatus.PASS, requires=frozenset()):
    return SimpleNamespace(
        check_id=check_id,
        depends_on=depends_on,
        requires_provider=requires,
        run=lambda _: CheckResult(check_id=check_id, status=status, duration_s=0),
    )


def test_scheduler_orders_dependencies_and_keeps_independent_checks():
    first = grader("runtime.z", status=CheckStatus.FAIL)
    dependent = grader("runtime.a", ("runtime.z",))
    independent = grader("runtime.other")
    assert [g.check_id for g in order_graders([dependent, first])] == [
        "runtime.z",
        "runtime.a",
    ]
    results = {
        r.check_id: r for r in execute_graders([dependent, first, independent], None)
    }
    assert results["runtime.a"].status is CheckStatus.SKIP
    assert "runtime.z" in results["runtime.a"].evidence[0]
    assert results["runtime.other"].status is CheckStatus.PASS


def test_scheduler_detects_cycle_and_missing_prerequisites():
    with pytest.raises(PolicyError, match="cycle"):
        order_graders(
            [grader("runtime.a", ("runtime.b",)), grader("runtime.b", ("runtime.a",))]
        )
    (result,) = execute_graders(
        [
            grader(
                "runtime.a",
                ("runtime.missing",),
                requires=frozenset({ProviderCapability.EXEC}),
            )
        ],
        None,
    )
    assert result.status is CheckStatus.SKIP
    assert "runtime.missing" in result.evidence[0]
    assert "exec" in result.evidence[1]


def test_grader_cannot_substitute_a_different_result_id():
    wrong = grader("runtime.wrong")
    wrong.check_id = "runtime.expected"
    (result,) = execute_graders([wrong], None)
    assert result.check_id == "runtime.expected"
    assert result.status is CheckStatus.ERROR


def test_digest_rejects_symlinks_without_reading_the_target(tmp_path):
    root = tmp_path / "subject"
    root.mkdir()
    (root / "link").symlink_to(tmp_path / "missing-secret")
    with pytest.raises(ValueError, match="symbolic"):
        source_digest(root)


def test_artifacts_encode_invalid_numbers_and_redact_secrets(package, tmp_path):
    report = run_validation(package, max_level=Level.RUNTIME, skip_build=True)
    raw = evidence(exchange("step", {}, {"token": "secret", "reward": float("nan")}))
    write_runtime_bundle(tmp_path / "bundle", report, evidence=raw)
    payload = (tmp_path / "bundle/collector-trace.json").read_text()
    assert "secret" not in payload
    parsed = json.loads(
        payload, parse_constant=lambda x: pytest.fail(f"invalid JSON number {x}")
    )
    assert parsed[0]["response_json"]["reward"] == {"invalid_number": "nan"}
