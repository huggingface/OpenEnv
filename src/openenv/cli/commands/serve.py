# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serve an OpenEnv environment locally (uvicorn, from ``openenv.yaml``)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .._cli_utils import console, validate_env_structure


def _find_repo_src_for_openenv(env_dir: Path) -> Path | None:
    """Return ``<repo>/src`` when ``env_dir`` is under an OpenEnv clone (for ``import openenv``)."""
    for parent in [env_dir, *env_dir.parents]:
        if (parent / "src" / "openenv").is_dir():
            return parent / "src"
    return None


def serve(
    env_path: Annotated[
        str | None,
        typer.Argument(
            help="Path to the environment directory (default: current directory)"
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind (default: ``port`` in openenv.yaml, else 8000)",
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option("--host", help="Host interface to bind"),
    ] = "0.0.0.0",
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable autoreload (development)"),
    ] = False,
) -> None:
    """
    Run the environment FastAPI app with uvicorn.

    Uses ``openenv.yaml`` fields ``app`` (e.g. ``server.app:app``), ``port``, and
    ``runtime`` (must be ``fastapi``). Matches ``uv run --project . server`` layout:
    the environment directory is the working directory and on ``sys.path``.

    For production or training, use Docker (``openenv build``) — this command runs
    on the host for local development only.
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise typer.BadParameter(
            "uvicorn is not installed. Run: pip install 'uvicorn>=0.24.0'"
        ) from exc

    env_path_obj = (
        Path.cwd().resolve() if env_path is None else Path(env_path).resolve()
    )

    try:
        validate_env_structure(env_path_obj)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"Not a valid OpenEnv environment: {exc}") from exc

    manifest_path = env_path_obj / "openenv.yaml"
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
    except OSError as exc:
        raise typer.BadParameter(f"Failed to read openenv.yaml: {exc}") from exc
    except yaml.YAMLError as exc:
        raise typer.BadParameter(f"Invalid YAML in openenv.yaml: {exc}") from exc

    if not isinstance(manifest, dict):
        raise typer.BadParameter("openenv.yaml must be a YAML dictionary")

    app_spec = manifest.get("app")
    if not app_spec or not isinstance(app_spec, str):
        raise typer.BadParameter(
            "openenv.yaml must contain a string 'app' field (e.g. server.app:app)"
        )
    if ":" not in app_spec:
        raise typer.BadParameter(
            f"openenv.yaml 'app' must look like 'module.path:attribute', got {app_spec!r}"
        )

    runtime = str(manifest.get("runtime", "fastapi")).lower()
    if runtime != "fastapi":
        raise typer.BadParameter(
            f"openenv serve only supports runtime 'fastapi' (got {runtime!r})"
        )

    raw_port = port if port is not None else manifest.get("port", 8000)
    try:
        listen_port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(
            f"Invalid port {raw_port!r}; expected an integer"
        ) from exc
    if not (1 <= listen_port <= 65535):
        raise typer.BadParameter(
            f"Invalid port {listen_port}; expected a value between 1 and 65535"
        )

    repo_src = _find_repo_src_for_openenv(env_path_obj)
    if repo_src is not None:
        repo_src_str = str(repo_src.resolve())
        if repo_src_str not in sys.path:
            sys.path.insert(0, repo_src_str)

    env_root = str(env_path_obj.resolve())
    if env_root not in sys.path:
        sys.path.insert(0, env_root)

    prev_cwd = os.getcwd()
    os.chdir(env_root)

    try:
        console.print(
            f"[bold green]Serving[/bold green] [cyan]{app_spec}[/cyan] on "
            f"[bold]http://{host}:{listen_port}/[/bold]  (cwd: {env_root})"
        )

        uvicorn.run(app_spec, host=host, port=listen_port, reload=reload)
    finally:
        try:
            os.chdir(prev_cwd)
        except OSError:
            pass
