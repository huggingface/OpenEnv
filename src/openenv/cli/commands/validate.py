# SPDX-License-Identifier: BSD-3-Clause

"""OpenEnv validate command."""

import json
from pathlib import Path
from typing import Annotated

import typer
from openenv.cli._validation import validate_running_environment
from openenv.validation import (
    Level,
    load_policy,
    PolicyError,
    run_validation,
    SignatureError,
    UnsupportedPackageError,
    ValidationReport,
    Verdict,
    write_report,
)


EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_UNSUPPORTED = 2
EXIT_INTERNAL = 3

_LEVELS = {
    "static": Level.STATIC,
    "runtime": Level.RUNTIME,
    "semantic": Level.SEMANTIC,
}


def _looks_like_url(value: str) -> bool:
    candidate = value.strip().lower()
    return candidate.startswith("http://") or candidate.startswith("https://")


def _render_report(report: ValidationReport) -> str:
    lines = [
        f"Validation report for {report.target} (signature: {report.signature.value})",
        f"  policy {report.policy_version} · levels run: "
        + ", ".join(level.name.lower() for level in report.levels_run),
    ]
    for result in report.results:
        lines.append(
            f"  {result.status.value.upper():5s} {result.check_id} ({result.duration_s:.2f}s)"
        )
        if result.status.value in ("fail", "error", "skip"):
            for line in result.evidence:
                lines.append(f"          {line}")
            if result.remediation:
                lines.append(f"          remediation: {result.remediation}")
    lines.append(f"Verdict: {report.verdict.value.upper()}")
    return "\n".join(lines)


def validate(
    target: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Path to the package directory (default: current directory) "
                "or a running OpenEnv URL (http://... or https://...)"
            ),
        ),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            help="Validate a running OpenEnv server by base URL (e.g. http://localhost:8000)",
        ),
    ] = None,
    level: Annotated[
        str,
        typer.Option(
            "--level",
            help="Validation level ceiling: static, runtime, or semantic",
        ),
    ] = "semantic",
    skip_build: Annotated[
        bool,
        typer.Option(
            "--skip-build",
            help="Skip the image build; build-dependent checks are SKIPped with a reason",
        ),
    ] = False,
    policy_version: Annotated[
        str,
        typer.Option("--policy", help="Severity policy version to apply"),
    ] = "v1",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the validation report as JSON"),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write the JSON report to a file"),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="HTTP timeout in seconds for --url runtime validation",
            min=0.1,
        ),
    ] = 5.0,
) -> None:
    """
    Validate a local package or a running OpenEnv server.

    Local validation detects the package format by its well-known file (the
    formats this build can parse; currently `openenv.yaml`), parses it into the
    normalized manifest, runs the applicable graders up to the requested level,
    applies the severity policy, and emits a report.

    Exit codes: 0 pass/warn · 1 fail · 2 unrecognized/unsupported package · 3
    internal error.

    Examples:

    ```bash
    # Validate the current directory up to the semantic level
    openenv validate

    # Fast inner loop: static checks only, no image build
    openenv validate envs/echo_env --level static --skip-build

    # Machine-readable report
    openenv validate envs/echo_env --json

    # Probe a running server (legacy runtime probe)
    openenv validate --url http://localhost:8000
    ```
    """
    runtime_target = url
    if (
        runtime_target is not None
        and target is not None
        and not _looks_like_url(target)
    ):
        typer.echo(
            "Error: Cannot combine a local path argument with --url runtime validation",
            err=True,
        )
        raise typer.Exit(EXIT_FAIL)

    if target is not None and _looks_like_url(target):
        if runtime_target is not None and runtime_target != target:
            typer.echo(
                "Error: Conflicting runtime targets provided via argument and --url",
                err=True,
            )
            raise typer.Exit(EXIT_FAIL)
        runtime_target = target

    if runtime_target is not None:
        try:
            report = validate_running_environment(runtime_target, timeout_s=timeout)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(EXIT_FAIL) from exc

        typer.echo(json.dumps(report, indent=2))
        if not report.get("passed", False):
            raise typer.Exit(EXIT_FAIL)
        return

    if level not in _LEVELS:
        typer.echo(
            f"Error: unknown level {level!r}; expected one of {sorted(_LEVELS)}",
            err=True,
        )
        raise typer.Exit(EXIT_INTERNAL)

    package_root = Path(target) if target is not None else Path.cwd()
    if not package_root.is_dir():
        typer.echo(f"Error: not a package directory: {package_root}", err=True)
        raise typer.Exit(EXIT_UNSUPPORTED)

    try:
        validation_report = run_validation(
            package_root,
            max_level=_LEVELS[level],
            skip_build=skip_build,
            policy=load_policy(policy_version),
        )
        report_json = write_report(validation_report, output)
    except (SignatureError, UnsupportedPackageError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_UNSUPPORTED) from exc
    except PolicyError as exc:
        typer.echo(f"Internal error: {exc}", err=True)
        raise typer.Exit(EXIT_INTERNAL) from exc
    except Exception as exc:
        typer.echo(f"Internal error: {exc}", err=True)
        raise typer.Exit(EXIT_INTERNAL) from exc

    if json_output:
        typer.echo(report_json)
    else:
        typer.echo(_render_report(validation_report))

    if validation_report.verdict is Verdict.FAIL:
        raise typer.Exit(EXIT_FAIL)
