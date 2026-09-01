"""Validation orchestration: parse → grade → apply policy → report."""

import hashlib
import time
from pathlib import Path

from .graders import GraderRegistry, Subject
from .graders.static import StaticManifestGrader
from .manifest import ManifestError, NormalizedManifest
from .parsers import ParserRegistry
from .parsers.openenv_yaml import OpenEnvYamlParser
from .policy import apply_policy, load_policy, SeverityPolicy
from .report import CheckResult, ValidationReport
from .signature import detect_signature
from .types import CheckStatus, Lane, Level

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
        if path.is_file() and not any(
            part in _DIGEST_EXCLUDED_DIRS for part in relative_path.parts
        ):
            files.append((relative_path.as_posix(), path))

    for relative_path, path in sorted(files, key=lambda item: item[0]):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def default_parser_registry() -> ParserRegistry:
    """The parsers shipped in this build."""
    registry = ParserRegistry()
    registry.register(OpenEnvYamlParser())
    return registry


def _run_grader(grader, subject: Subject) -> CheckResult:
    started = time.monotonic()
    try:
        return grader.run(subject)
    except Exception as exc:
        return CheckResult(
            check_id=grader.check_id,
            status=CheckStatus.ERROR,
            evidence=[f"grader crashed: {exc!r}"],
            duration_s=time.monotonic() - started,
        )


def run_validation(
    target: Path,
    *,
    max_level: Level = Level.SEMANTIC,
    skip_build: bool = False,
    policy: SeverityPolicy | None = None,
) -> ValidationReport:
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
            Severity policy; `None` loads the committed default version.

    Returns:
        [`~openenv.validation.report.ValidationReport`]: the completed report.
    """
    del skip_build
    target = Path(target)
    policy = policy or load_policy()
    signature = detect_signature(target)

    parser = default_parser_registry().parser_for(signature)
    manifest: NormalizedManifest | None = None
    results: list[CheckResult] = []

    parse_started = time.monotonic()
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
        results.extend(
            _run_grader(grader, subject)
            for grader in graders.select(manifest, max_level)
        )

    return ValidationReport(
        report_schema_version=REPORT_SCHEMA_VERSION,
        target=str(target),
        source_digest=source_digest(target),
        signature=signature,
        manifest=manifest,
        policy_version=policy.policy_version,
        lane=Lane.LOCAL,
        levels_run=[Level.STATIC],
        results=results,
        verdict=apply_policy(results, policy, Lane.LOCAL),
    )
