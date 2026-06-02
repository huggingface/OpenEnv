# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv validate command.

This module provides the 'openenv validate' command to check if environments
are properly configured for multi-mode deployment.
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from openenv.cli._validation import (
    build_local_validation_json_report,
    format_validation_report,
    get_deployment_modes,
    validate_multi_mode_deployment,
    validate_running_environment,
)


def _looks_like_url(value: str) -> bool:
    """Return True when the value appears to be a URL target."""
    candidate = value.strip().lower()
    return candidate.startswith("http://") or candidate.startswith("https://")


def validate(
    target: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Path to the environment directory (default: current directory) "
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
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output local validation report as JSON (runtime validation is JSON by default)",
        ),
    ] = False,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="HTTP timeout in seconds for runtime validation",
            min=0.1,
        ),
    ] = 5.0,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed information")
    ] = False,
) -> None:
    """
    Validate local environments and running OpenEnv servers.

    Local validation checks if an environment is properly configured with:
    - Required files (pyproject.toml, openenv.yaml, server/app.py, etc.)
    - Docker deployment support
    - uv run server capability
    - python -m module execution

    Runtime validation checks if a live OpenEnv server conforms to the
    versioned runtime API contract and returns a criteria-based JSON report.

    Examples:
        # Validate current directory (recommended)
        $ cd my_env
        $ openenv validate

        # Validate a running environment and return JSON criteria
        $ openenv validate --url http://localhost:8000
        $ openenv validate https://my-env.hf.space

        # Validate with detailed output
        $ openenv validate --verbose

        # Validate specific environment
        $ openenv validate envs/echo_env
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
        raise typer.Exit(1)

    if target is not None and _looks_like_url(target):
        if runtime_target is not None and runtime_target != target:
            typer.echo(
                "Error: Conflicting runtime targets provided via argument and --url",
                err=True,
            )
            raise typer.Exit(1)
        runtime_target = target

    if runtime_target is not None:
        try:
            report = validate_running_environment(runtime_target, timeout_s=timeout)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

        typer.echo(json.dumps(report, indent=2))
        if not report.get("passed", False):
            raise typer.Exit(1)
        return

    # Determine environment path (default to current directory)
    if target is None:
        env_path_obj = Path.cwd()
    else:
        env_path_obj = Path(target)

    if not env_path_obj.exists():
        typer.echo(f"Error: Path does not exist: {env_path_obj}", err=True)
        raise typer.Exit(1)

    if not env_path_obj.is_dir():
        typer.echo(f"Error: Path is not a directory: {env_path_obj}", err=True)
        raise typer.Exit(1)

    # Check for openenv.yaml to confirm this is an environment directory
    openenv_yaml = env_path_obj / "openenv.yaml"
    if not openenv_yaml.exists():
        typer.echo(
            f"Error: Not an OpenEnv environment directory (missing openenv.yaml): {env_path_obj}",
            err=True,
        )
        typer.echo(
            "Hint: Run this command from the environment root directory or specify the path",
            err=True,
        )
        raise typer.Exit(1)

    env_name = env_path_obj.name
    if env_name.endswith("_env"):
        base_name = env_name[:-4]
    else:
        base_name = env_name

    # Run validation
    is_valid, issues = validate_multi_mode_deployment(env_path_obj)
    modes = get_deployment_modes(env_path_obj)

    if json_output:
        report = build_local_validation_json_report(
            env_name=base_name,
            env_path=env_path_obj,
            is_valid=is_valid,
            issues=issues,
            deployment_modes=modes if verbose else None,
        )
        typer.echo(json.dumps(report, indent=2))
        if not is_valid:
            raise typer.Exit(1)
        return

    # Show validation report
    report = format_validation_report(base_name, is_valid, issues)
    typer.echo(report)

    # Show deployment modes if verbose
    if verbose:
        typer.echo("\nSupported deployment modes:")
        for mode, supported in modes.items():
            status = "[YES]" if supported else "[NO]"
            typer.echo(f"  {status} {mode}")

        if is_valid:
            typer.echo("\nUsage examples:")
            typer.echo(f"  cd {env_path_obj.name} && uv run server")
            typer.echo(f"  cd {env_path_obj.name} && openenv build")
            typer.echo(f"  cd {env_path_obj.name} && openenv push")

    if not is_valid:
        raise typer.Exit(1)
