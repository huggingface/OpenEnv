# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI utilities for OpenEnv command-line interface."""

from pathlib import Path
from typing import List

from rich.console import Console

# Create a console instance for CLI output
console = Console()


def validate_env_structure(env_dir: Path, strict: bool = False) -> List[str]:
    """
    Validate that the directory follows OpenEnv environment structure.

    Args:
        env_dir: Path to environment directory
        strict: If True, enforce all optional requirements

    Returns:
        List of validation warnings (empty if all checks pass)

    Raises:
        FileNotFoundError: If required files are missing
    """
    warnings = []

    # Required files
    required_files = [
        "openenv.yaml",
        "__init__.py",
        "client.py",
        "models.py",
        "README.md",
    ]

    for file in required_files:
        if not (env_dir / file).exists():
            raise FileNotFoundError(f"Required file missing: {file}")

    # Dockerfile: must exist in server/ or at env root
    has_root_dockerfile = (env_dir / "Dockerfile").exists()
    has_server_dockerfile = (env_dir / "server" / "Dockerfile").exists()

    if not has_root_dockerfile and not has_server_dockerfile:
        raise FileNotFoundError(
            "Required file missing: server/Dockerfile or Dockerfile at env root"
        )

    # When no root Dockerfile, require the traditional server/ layout
    if not has_root_dockerfile:
        server_dir = env_dir / "server"
        if not server_dir.exists() or not server_dir.is_dir():
            raise FileNotFoundError("Required directory missing: server/")

        for file in ["server/__init__.py", "server/app.py"]:
            if not (env_dir / file).exists():
                raise FileNotFoundError(f"Required file missing: {file}")

    # Check for dependency management (pyproject.toml required)
    has_pyproject = (env_dir / "pyproject.toml").exists()

    if not has_pyproject:
        raise FileNotFoundError(
            "No dependency specification found. 'pyproject.toml' is required."
        )

    # Warnings for recommended structure

    if not (env_dir / "outputs").exists():
        warnings.append("Recommended directory missing: outputs/")

    return warnings
