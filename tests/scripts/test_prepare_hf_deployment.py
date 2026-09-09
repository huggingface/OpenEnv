"""Test staging behavior for the Hugging Face deployment shell helper.

The suite verifies repository naming, web-interface rewriting, aliases, and
explicit opt-out markers without contacting the Hugging Face Hub.
"""

from __future__ import annotations

import os
import shutil
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


def test_prepare_hf_deployment_all_honors_opt_out_but_manual_selection_works(
    tmp_path: Path,
) -> None:
    """Automatic discovery should honor the generic marker without blocking --env."""
    repo_root = tmp_path / "openenv"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    source_script = Path(__file__).resolve().parents[2] / "scripts"
    script_path = scripts_dir / "prepare_hf_deployment.sh"
    shutil.copy2(source_script / script_path.name, script_path)
    (repo_root / "pyproject.toml").write_text(
        '[project]\nversion = "0.4.2"\n',
        encoding="utf-8",
    )
    (repo_root / "src").mkdir()
    (repo_root / "src" / "placeholder.txt").write_text("source\n", encoding="utf-8")

    for env_name in ("ordinary_env", "thinkingbox_env"):
        env_dir = repo_root / "envs" / env_name
        env_dir.mkdir(parents=True)
        (env_dir / "Dockerfile").write_text(
            "FROM python:3.12-slim\n",
            encoding="utf-8",
        )
        (env_dir / "README.md").write_text(
            f"# {env_name}\n",
            encoding="utf-8",
        )
    (repo_root / "envs" / "thinkingbox_env" / "SKIP_HF_DEPLOYMENT").write_text(
        "Requires external services.\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update({"HF_NAMESPACE": "test", "OPENENV_VERSION": "main"})
    automatic_staging = tmp_path / "automatic-staging"
    automatic = subprocess.run(
        [
            "bash",
            str(script_path),
            "--all",
            "--dry-run",
            "--skip-collection",
            "--staging-dir",
            str(automatic_staging),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert automatic.returncode == 0, automatic.stderr
    assert "Would create/update space: test/ordinary_env-main" in automatic.stdout
    assert (
        "Would create/update space: test/thinkingbox_env-main" not in automatic.stdout
    )
    assert "SKIP_HF_DEPLOYMENT opts out of --all" in automatic.stderr
    assert (automatic_staging / "test" / "ordinary_env-main").is_dir()
    assert not (automatic_staging / "test" / "thinkingbox_env-main").exists()

    explicit_staging = tmp_path / "explicit-staging"
    explicit = subprocess.run(
        [
            "bash",
            str(script_path),
            "--env",
            "thinkingbox_env",
            "--repo-id",
            "test/external",
            "--dry-run",
            "--skip-collection",
            "--staging-dir",
            str(explicit_staging),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert explicit.returncode == 0, explicit.stderr
    assert "Would create/update space: test/external" in explicit.stdout
    assert (explicit_staging / "test" / "external").is_dir()
