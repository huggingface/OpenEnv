"""Validation orchestration: parse → grade → apply policy → report."""

import hashlib
import os
import secrets
import stat
import time
import uuid
from dataclasses import replace
from pathlib import Path

from .graders import GraderRegistry, Subject
from .graders.runtime import (
    ObservationSchemaGrader,
    RewardWellFormedGrader,
    StateContractGrader,
)
from .graders.runtime.discovery import (
    RewardAttributionGrader,
    RubricIntrospectableGrader,
    TaskDeclarationAccuracyGrader,
    ToolDeclarationAccuracyGrader,
)
from .graders.runtime.repeatability import (
    EpisodeDeterminismGrader,
    SeedControlGrader,
    TrajectoryRecordGrader,
)
from .graders.static import StaticManifestGrader
from .manifest import ManifestError, NormalizedManifest, NormalizedManifestV2
from .parsers import ParserRegistry
from .parsers.openenv_yaml import OpenEnvYamlParser
from .policy import apply_policy, load_policy, PolicyError, SeverityPolicy
from .providers import ProviderError, StartupError, UnsupportedCapability
from .report import CheckResult, ValidationReport, ValidationReportV2
from .runtime.artifacts import write_runtime_bundle
from .runtime.collector import collect_runtime_evidence, RuntimeCollectionInterrupted
from .runtime.contracts import LaunchSpec, load_runtime_plan, RuntimePlanError
from .runtime.replay import collect_replays, REPLAY_BUDGET_SECONDS
from .runtime.scheduler import execute_graders
from .signature import detect_signature
from .types import CheckStatus, Lane, Level, ProviderCapability

REPORT_SCHEMA_VERSION = "1"

_DIGEST_EXCLUDED_DIRS = {".git", "__pycache__", ".venv", ".worktrees", "outputs"}


def source_digest(package_root: Path) -> str:
    """
    Deterministic sha256 over the package tree (relative paths + file contents).

    Args:
        package_root (`Path`):
            The package directory.

    Returns:
        `str`: a 64-character hex digest.
    """
    digest = hashlib.sha256()
    files = []
    for path in package_root.rglob("*"):
        relative_path = path.relative_to(package_root)
        if any(part in _DIGEST_EXCLUDED_DIRS for part in relative_path.parts):
            continue
        if path.is_symlink():
            raise ValueError("validation source may not contain symbolic links")
        if path.is_file():
            files.append((relative_path.as_posix(), path))

    for relative_path, path in sorted(files, key=lambda item: item[0]):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("validation source must contain regular files")
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def default_parser_registry() -> ParserRegistry:
    """The parsers shipped in this build."""
    registry = ParserRegistry()
    registry.register(OpenEnvYamlParser())
    return registry


def _outcome(check_id, status, reason, *, started=None, measured=None):
    return CheckResult(
        check_id=check_id,
        status=status,
        evidence=[reason],
        measured=measured or {},
        duration_s=time.monotonic() - started if started is not None else 0,
    )


def _applicable(check_id, manifest):
    if manifest is None:
        return True
    if check_id in {"runtime.rubric_introspectable", "runtime.reward_attribution"}:
        return manifest.capabilities.rubric_tree
    if check_id == "runtime.task_declaration_accuracy":
        return manifest.capabilities.task_api or bool(
            manifest.capabilities.declared_task_count
        )
    return True


def _runtime(subject, *, skip_build, provider):
    """Own build/start/collection/stop and retain failure evidence through teardown."""
    manifest = subject.manifest
    plan = None
    evidence = None
    running = None
    inspection = {}
    cleanup = {"required": False, "completed": True}
    started = time.monotonic()
    attempted = False
    result = None
    checks = []
    try:
        if skip_build:
            raise UnsupportedCapability(
                "--skip-build: build-dependent runtime checks skipped"
            )
        if not isinstance(manifest, NormalizedManifestV2):
            raise UnsupportedCapability(
                "missing validation.execution declaration and runtime plan"
            )
        plan = load_runtime_plan(subject.root, manifest.execution)
        if provider is None:
            from .providers.docker import DockerValidationProvider

            provider = DockerValidationProvider()
        if manifest.network.mode not in provider.supported_network_modes:
            raise UnsupportedCapability(
                f"provider cannot enforce network mode {manifest.network.mode}"
            )
        if (
            manifest.resources.gpus
            and ProviderCapability.GPU not in provider.capabilities
        ):
            raise UnsupportedCapability("missing provider capability: gpu")
        if ProviderCapability.IMAGE_BUILD not in provider.capabilities:
            raise UnsupportedCapability("missing provider capability: image_build")
        # Validate all launch constraints before any container or build is started.
        spec = LaunchSpec(
            image_ref="sha256:" + "0" * 64,
            resources=manifest.resources,
            network=manifest.network,
            run_id="validation-" + uuid.uuid4().hex,
            env_vars={"OPENENV_VALIDATION_TOKEN": secrets.token_urlsafe(32)},
        )
        attempted = True
        image_ref = provider.build(subject.root, manifest.execution)
        spec = LaunchSpec.model_validate({**spec.model_dump(), "image_ref": image_ref})
        running = provider.start(spec)
        cleanup = {"required": True, "completed": False}
        inspection = running.inspect()
        result = _outcome(
            "runtime.startup",
            CheckStatus.PASS,
            "subject built and reached its control endpoint",
            started=started,
            measured={"provider": provider.name, "image_ref": image_ref},
        )
        replay_deadline = time.monotonic() + REPLAY_BUDGET_SECONDS
        evidence = collect_runtime_evidence(
            running.base_url,
            plan,
            episode_timeout_s=min(
                manifest.resources.episode_timeout_s, REPLAY_BUDGET_SECONDS
            ),
            validation_token=spec.env_vars["OPENENV_VALIDATION_TOKEN"],
            collect_tools=True,
            task_env_name=manifest.name
            if _applicable("runtime.task_declaration_accuracy", manifest)
            else None,
        )
        # A health endpoint without a functioning protocol isn't a startup success.
        if not evidence.exchanges and evidence.failure_reason:
            result = _outcome(
                "runtime.startup",
                CheckStatus.FAIL,
                evidence.failure_reason,
                started=started,
            )
        evidence = collect_replays(
            provider,
            running,
            spec,
            plan,
            evidence,
            capabilities=manifest.capabilities,
            deadline=replay_deadline,
        )
        subject = replace(
            subject, image_ref=image_ref, running=running, runtime_evidence=evidence
        )
        runtime_graders = [
            RewardWellFormedGrader(),
            ObservationSchemaGrader(),
            StateContractGrader(),
            SeedControlGrader(),
            EpisodeDeterminismGrader(),
            TrajectoryRecordGrader(),
            ToolDeclarationAccuracyGrader(),
            TaskDeclarationAccuracyGrader(),
            RubricIntrospectableGrader(),
            RewardAttributionGrader(),
        ]
        checks = execute_graders(
            [grader for grader in runtime_graders if grader.applies_to(manifest)],
            subject,
            provider_capabilities=provider.capabilities,
            prior=[result],
        )
        # Cleanup failure cannot invalidate evidence already collected from a
        # healthy subject. Preserve its findings before failing the run closed.
        if any(replay.cleanup_complete is False for replay in evidence.replays):
            result.status = CheckStatus.ERROR
            result.evidence.append("replay subject teardown failed")
    except UnsupportedCapability as exc:
        result = _outcome(
            "runtime.startup", CheckStatus.SKIP, str(exc), started=started
        )
    except (RuntimePlanError, StartupError) as exc:
        result = _outcome(
            "runtime.startup",
            CheckStatus.FAIL,
            str(exc)[:4096],
            started=started,
        )
    except ProviderError as exc:
        result = _outcome(
            "runtime.startup",
            CheckStatus.ERROR,
            str(exc)[:4096],
            started=started,
        )
    except KeyboardInterrupt as exc:
        if isinstance(exc, RuntimeCollectionInterrupted):
            evidence = exc.evidence
        result = _outcome(
            "runtime.startup",
            CheckStatus.ERROR,
            "validation interrupted",
            started=started,
        )
    except Exception as exc:
        result = _outcome(
            "runtime.startup",
            CheckStatus.ERROR,
            f"runtime orchestration failed ({type(exc).__name__})",
            started=started,
        )
    finally:
        if running is not None:
            try:
                running.stop()
                cleanup["completed"] = not (
                    evidence
                    and any(
                        replay.cleanup_complete is False for replay in evidence.replays
                    )
                )
            except (Exception, KeyboardInterrupt):
                cleanup["completed"] = False
                if result is None:
                    result = _outcome(
                        "runtime.startup",
                        CheckStatus.ERROR,
                        "subject teardown failed",
                        started=started,
                    )
                else:
                    result.status = CheckStatus.ERROR
                    result.evidence.append("subject teardown failed")
                    result.duration_s = time.monotonic() - started
    return [result, *checks], attempted, plan, evidence, inspection, cleanup


def run_validation(
    target: Path,
    *,
    max_level: Level = Level.SEMANTIC,
    skip_build: bool = False,
    policy: SeverityPolicy | None = None,
    provider=None,
    artifacts_dir: Path | None = None,
) -> ValidationReport | ValidationReportV2:
    """
    Validate a package end to end and return the report.

    Raises [`~openenv.validation.signature.SignatureError`] for ambiguous or
    unrecognized packages and
    [`~openenv.validation.signature.UnsupportedPackageError`] for
    recognized-but-unsupported ones (CLI exit code 2). A package whose declarations
    fail the manifest schema yields a normal report with a `static.manifest` FAIL
    (exit code 1).

    Args:
        target (`Path`):
            The package directory.
        max_level ([`~openenv.validation.types.Level`], *optional*, defaults to `Level.SEMANTIC`):
            Level ceiling; graders above it are not selected.
        skip_build (`bool`, *optional*, defaults to `False`):
            Skip the image build; build-dependent checks SKIP with a reason.
        policy ([`~openenv.validation.policy.SeverityPolicy`], *optional*):
            `None` chooses v1 for static and v2 for runtime/semantic ceilings.
        provider ([`~openenv.validation.providers.ValidationProvider`], *optional*):
            Validation-only provider; defaults to Docker-local for runtime runs.
        artifacts_dir (`Path`, *optional*):
            Write a redacted reproduction bundle to this directory.

    Returns:
        [`~openenv.validation.report.ValidationReport`]: the completed report.
    """
    target = Path(target)
    wants_runtime = max_level >= Level.RUNTIME
    policy = policy or load_policy("v2" if wants_runtime else "v1")
    if wants_runtime and "runtime.startup" not in policy.entries_for_lane(Lane.LOCAL):
        raise PolicyError(
            "runtime validation requires policy v2; v1 supports --level static"
        )
    signature = detect_signature(target)
    try:
        digest_before = source_digest(target)
    except (ValueError, OSError):
        digest_before = ""

    parser = default_parser_registry().parser_for(signature)
    manifest: NormalizedManifest | None = None
    results: list[CheckResult] = []

    parse_started = time.monotonic()
    if not digest_before:
        results.append(
            _outcome(
                "static.manifest",
                CheckStatus.ERROR,
                "package source could not be verified before validation",
                started=parse_started,
            )
        )
    else:
        try:
            manifest = parser.parse(target)
        except ManifestError as exc:
            results.append(
                CheckResult(
                    check_id="static.manifest",
                    status=CheckStatus.FAIL,
                    measured={"schema_errors": len(exc.errors)},
                    evidence=exc.errors,
                    remediation=exc.remediation,
                    duration_s=time.monotonic() - parse_started,
                )
            )

    levels = [Level.STATIC]
    plan = evidence = None
    inspection = {}
    cleanup = {"required": False, "completed": True}
    if manifest is not None:
        graders = GraderRegistry()
        graders.register(StaticManifestGrader(policy.bounds))
        subject = Subject(
            root=target,
            manifest=manifest,
            image_ref=None,
            running=None,
            outputs_dir=target / "outputs",
        )
        results.extend(execute_graders(graders.select(manifest, Level.STATIC), subject))
        if wants_runtime:
            runtime_results, attempted, plan, evidence, inspection, cleanup = _runtime(
                subject, skip_build=skip_build, provider=provider
            )
            results.extend(runtime_results)
            if attempted:
                levels.append(Level.RUNTIME)

    if wants_runtime:
        # Policy IDs are an inventory, not evidence that a grader exists. Keep the
        # incomplete surface explicit throughout the staged implementation.
        present = {result.check_id for result in results}
        for entry in policy.entries_for_lane(Lane.LOCAL).values():
            if (
                entry.level in {Level.RUNTIME, Level.SEMANTIC}
                and entry.level <= max_level
                and entry.check_id not in present
                and _applicable(entry.check_id, manifest)
            ):
                reason = "grader not implemented in this build"
                if manifest is None:
                    reason = "unmet dependency: valid manifest"
                elif entry.check_id in {
                    "runtime.reward_well_formed",
                    "runtime.observation_schema",
                    "runtime.state_contract",
                    "runtime.seed_control",
                    "runtime.episode_determinism",
                    "runtime.trajectory_record",
                    "runtime.tool_declaration_accuracy",
                    "runtime.task_declaration_accuracy",
                    "runtime.rubric_introspectable",
                    "runtime.reward_attribution",
                }:
                    reason = "unmet dependency: runtime.startup"
                results.append(_outcome(entry.check_id, CheckStatus.SKIP, reason))
        source_problem = None
        if not digest_before:
            source_problem = "package source could not be verified before validation"
        else:
            try:
                digest_after = source_digest(target)
            except (ValueError, OSError):
                digest_after = None
            if digest_after != digest_before:
                source_problem = (
                    "package source changed during validation"
                    if digest_after is not None
                    else "package source could not be verified after validation"
                )
        if source_problem is not None:
            results = [
                _outcome(
                    r.check_id,
                    CheckStatus.SKIP,
                    f"unmet dependency: runtime.startup ({source_problem})",
                )
                if r.check_id
                in {
                    "runtime.reward_well_formed",
                    "runtime.observation_schema",
                    "runtime.state_contract",
                    "runtime.seed_control",
                    "runtime.episode_determinism",
                    "runtime.trajectory_record",
                    "runtime.tool_declaration_accuracy",
                    "runtime.task_declaration_accuracy",
                    "runtime.rubric_introspectable",
                    "runtime.reward_attribution",
                }
                else r
                for r in results
                if r.check_id != "runtime.startup"
            ]
            results.append(
                _outcome(
                    "runtime.startup",
                    CheckStatus.ERROR,
                    source_problem,
                )
            )

    report_type = (
        ValidationReportV2
        if wants_runtime or isinstance(manifest, NormalizedManifestV2)
        else ValidationReport
    )
    report = report_type(
        report_schema_version="2"
        if report_type is ValidationReportV2
        else REPORT_SCHEMA_VERSION,
        target=str(target),
        source_digest=digest_before,
        signature=signature,
        manifest=manifest,
        policy_version=policy.policy_version,
        lane=Lane.LOCAL,
        levels_run=levels,
        results=results,
        verdict=apply_policy(results, policy, Lane.LOCAL),
    )
    if artifacts_dir is not None:
        write_runtime_bundle(
            artifacts_dir,
            report,
            plan=plan,
            evidence=evidence,
            provider=inspection,
            cleanup=cleanup,
        )
    return report
