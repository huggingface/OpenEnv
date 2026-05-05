# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Commands to manage OpenEnv CLI skills for AI assistants."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Annotated

import typer

DEFAULT_SKILL_ID = "openenv-cli"

_SKILL_YAML_PREFIX = """\
---
name: openenv-cli
description: "OpenEnv CLI (`openenv`) for scaffolding, validating, building, and pushing OpenEnv environments."
---

Install: `pip install openenv-core`

The OpenEnv CLI command `openenv` is available.
Use `openenv --help` to view available commands.
"""

_SKILL_TIPS = """
## Tips

- Start with `openenv init <env_name>` to scaffold a new environment
- Validate projects with `openenv validate`
- Build and deploy with `openenv build` and `openenv push`
- Use `openenv <command> --help` for command-specific options
"""

CENTRAL_LOCAL = Path(".agents/skills")
CENTRAL_GLOBAL = Path("~/.agents/skills")

GLOBAL_TARGETS = {
    "codex": Path("~/.codex/skills"),
    "claude": Path("~/.claude/skills"),
    "cursor": Path("~/.cursor/skills"),
    "opencode": Path("~/.config/opencode/skills"),
}

LOCAL_TARGETS = {
    "codex": Path(".codex/skills"),
    "claude": Path(".claude/skills"),
    "cursor": Path(".cursor/skills"),
    "opencode": Path(".opencode/skills"),
}

app = typer.Typer(help="Manage OpenEnv skills for AI assistants")


def _build_skill_md() -> str:
    """Generate SKILL.md content for the OpenEnv CLI skill."""
    from openenv import __version__

    lines = _SKILL_YAML_PREFIX.splitlines()
    lines.append("")
    lines.append(
        f"Generated with `openenv-core v{__version__}`. Run `openenv skills add --force` to regenerate."
    )
    lines.extend(_SKILL_TIPS.splitlines())
    return "\n".join(lines).strip() + "\n"


def _remove_existing(path: Path, force: bool) -> None:
    """Remove existing file/directory/symlink if force is True, else fail."""
    if not (path.exists() or path.is_symlink()):
        return
    if not force:
        raise typer.Exit(code=1)

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _install_to(skills_dir: Path, force: bool) -> Path:
    """Install the OpenEnv skill in a skills directory."""
    skills_dir = skills_dir.expanduser().resolve()
    skills_dir.mkdir(parents=True, exist_ok=True)
    dest = skills_dir / DEFAULT_SKILL_ID

    if dest.exists() or dest.is_symlink():
        if not force:
            typer.echo(
                f"Skill already exists at {dest}. Re-run with --force to overwrite."
            )
            raise typer.Exit(code=1)
        _remove_existing(dest, force=True)

    dest.mkdir()
    (dest / "SKILL.md").write_text(_build_skill_md(), encoding="utf-8")
    return dest


def _create_symlink(
    agent_skills_dir: Path, central_skill_path: Path, force: bool
) -> Path:
    """Create a relative symlink from agent directory to central skill location."""
    agent_skills_dir = agent_skills_dir.expanduser().resolve()
    agent_skills_dir.mkdir(parents=True, exist_ok=True)
    link_path = agent_skills_dir / DEFAULT_SKILL_ID

    if link_path.exists() or link_path.is_symlink():
        if not force:
            typer.echo(
                f"Skill already exists at {link_path}. Re-run with --force to overwrite."
            )
            raise typer.Exit(code=1)
        _remove_existing(link_path, force=True)

    link_path.symlink_to(os.path.relpath(central_skill_path, agent_skills_dir))
    return link_path


@app.command("preview")
def skills_preview() -> None:
    """Print generated SKILL.md content."""
    typer.echo(_build_skill_md())


@app.command("add")
def skills_add(
    claude: Annotated[
        bool,
        typer.Option("--claude", help="Install for Claude."),
    ] = False,
    codex: Annotated[
        bool,
        typer.Option("--codex", help="Install for Codex."),
    ] = False,
    cursor: Annotated[
        bool,
        typer.Option("--cursor", help="Install for Cursor."),
    ] = False,
    opencode: Annotated[
        bool,
        typer.Option("--opencode", help="Install for OpenCode."),
    ] = False,
    global_: Annotated[
        bool,
        typer.Option(
            "--global",
            "-g",
            help=(
                "Install globally (user-level) instead of in the current project directory."
            ),
        ),
    ] = False,
    dest: Annotated[
        Path | None,
        typer.Option(help="Install into a custom destination (skills directory path)."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing skills in the destination."),
    ] = False,
) -> None:
    """Install OpenEnv CLI skill for AI assistants."""
    if dest:
        if claude or codex or cursor or opencode or global_:
            typer.echo(
                "--dest cannot be combined with --claude, --codex, --cursor, --opencode, or --global."
            )
            raise typer.Exit(code=1)
        skill_dest = _install_to(dest, force)
        typer.echo(f"Installed '{DEFAULT_SKILL_ID}' to {skill_dest}")
        return

    central_path = CENTRAL_GLOBAL if global_ else CENTRAL_LOCAL
    central_skill_path = _install_to(central_path, force)
    typer.echo(
        f"Installed '{DEFAULT_SKILL_ID}' to central location: {central_skill_path}"
    )

    targets = GLOBAL_TARGETS if global_ else LOCAL_TARGETS
    agent_targets: list[Path] = []

    if claude:
        agent_targets.append(targets["claude"])
    if codex:
        agent_targets.append(targets["codex"])
    if cursor:
        agent_targets.append(targets["cursor"])
    if opencode:
        agent_targets.append(targets["opencode"])

    for agent_target in agent_targets:
        link_path = _create_symlink(agent_target, central_skill_path, force)
        typer.echo(f"Created symlink: {link_path}")
