"""Tests for the Hugging Face deployment shell helper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_prepare_hf_deployment_repo_id_override(tmp_path: Path) -> None:
    """An exact repo override should target the canonical repo and README URLs."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "prepare_hf_deployment.sh"
    staging_dir = tmp_path / "hf-staging"

    env = os.environ.copy()
    env["OPENENV_VERSION"] = "main"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--env",
            "repl_env",
            "--repo-id",
            "openenv/repl",
            "--dry-run",
            "--skip-collection",
            "--staging-dir",
            str(staging_dir),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[dry-run] Would create/update space: openenv/repl" in result.stdout

    generated_readme = staging_dir / "openenv" / "repl" / "README.md"
    assert generated_readme.exists()
    readme_text = generated_readme.read_text()
    assert "https://huggingface.co/spaces/openenv/repl" in readme_text
    assert "https://huggingface.co/spaces/openenv/repl_env" not in readme_text


def test_prepare_hf_deployment_overrides_disabled_web_interface(tmp_path: Path) -> None:
    """Staged Hub Dockerfiles should force Gradio web interface on."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "prepare_hf_deployment.sh"
    staging_dir = tmp_path / "hf-staging"

    env = os.environ.copy()
    env["OPENENV_VERSION"] = "0.2.3"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--env",
            "chat_env",
            "--dry-run",
            "--skip-collection",
            "--staging-dir",
            str(staging_dir),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    staged_dockerfiles = [
        path
        for path in staging_dir.rglob("Dockerfile")
        if path.parent.name == "chat_env-0.2.3"
    ]
    assert len(staged_dockerfiles) == 1
    staged_dockerfile = staged_dockerfiles[0]
    dockerfile_text = staged_dockerfile.read_text()
    assert "ENV ENABLE_WEB_INTERFACE=true" in dockerfile_text
    assert "ENV ENABLE_WEB_INTERFACE=false" not in dockerfile_text


def test_prepare_hf_deployment_sets_textarena_alias_env_id(tmp_path: Path) -> None:
    """TextArena canonical aliases should inject the correct game id into the Dockerfile."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "prepare_hf_deployment.sh"
    staging_dir = tmp_path / "hf-staging"

    env = os.environ.copy()
    env["OPENENV_VERSION"] = "0.2.3"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--env",
            "textarena_env",
            "--repo-id",
            "openenv/sudoku",
            "--dry-run",
            "--skip-collection",
            "--staging-dir",
            str(staging_dir),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    staged_dockerfiles = [
        path for path in staging_dir.rglob("Dockerfile") if path.parent.name == "sudoku"
    ]
    assert len(staged_dockerfiles) == 1
    dockerfile_text = staged_dockerfiles[0].read_text()
    assert "ENV ENABLE_WEB_INTERFACE=true" in dockerfile_text
    assert "ENV TEXTARENA_ENV_ID=Sudoku-v0" in dockerfile_text
