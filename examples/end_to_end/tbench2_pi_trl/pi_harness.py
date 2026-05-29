#!/usr/bin/env python3
"""Pi-style context loading for the Terminus async GRPO example."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Message = dict[str, Any]

DEFAULT_PI_SYSTEM_PROMPT = """You are running inside a Pi-compatible environment session.

Follow the project AGENTS.md and available skill instructions. Use the available
tools exactly as described by their schemas. When the task is complete, submit
the environment's final tool call.
"""


@dataclass
class PiContext:
    """Load model-facing project instructions and skills."""

    project_root: Path
    agents_md_path: str = "AGENTS.md"
    skills_dir: str = ".pi/skills"
    max_context_chars: int = 6000
    extra_system_prompts: list[str] = field(default_factory=list)

    def messages(self) -> list[Message]:
        parts = [DEFAULT_PI_SYSTEM_PROMPT.strip(), *self.extra_system_prompts]
        for label, path, text in self.context_files():
            parts.append(f"{label} ({path}):\n{text}")
        return [{"role": "system", "content": "\n\n".join(parts)}]

    def loaded_files(self) -> list[str]:
        return [path for _, path, _ in self.context_files()]

    def context_files(self) -> list[tuple[str, str, str]]:
        files = [
            ("Project instructions", self._resolve(self.agents_md_path)),
            *(
                (f"Skill: {path.parent.name}", path)
                for path in _iter_skill_files(self._resolve(self.skills_dir))
            ),
        ]
        loaded = []
        for label, path in files:
            if text := _read_bounded(path, self.max_context_chars):
                loaded.append((label, self._display(path), text))
        return loaded

    def _resolve(self, path: str) -> Path:
        resolved = Path(path)
        return resolved if resolved.is_absolute() else self.project_root / resolved

    def _display(self, path: Path) -> str:
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()


def _iter_skill_files(skills_path: Path) -> list[Path]:
    if not skills_path.exists():
        return []
    if skills_path.is_file():
        return [skills_path]
    skill_files = set(skills_path.rglob("SKILL.md"))
    skill_files.update(path for path in skills_path.glob("*.md") if path.is_file())
    return sorted(skill_files, key=lambda path: path.as_posix())


def _read_bounded(path: Path, max_chars: int) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[truncated by PiContext]"


__all__ = ["PiContext"]
