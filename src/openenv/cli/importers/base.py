"""Shared importer registry types."""

from __future__ import annotations

import fnmatch
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

_EXCLUDED_FILE_SUFFIXES = {
    ".key",
    ".p12",
    ".pfx",
    ".pem",
    ".pyc",
    ".pyo",
}

_EXCLUDED_FILE_NAMES = {
    ".env",
    ".netrc",
    "credentials.json",
    "secrets.json",
    "secrets.toml",
    "secrets.yaml",
    "secrets.yml",
}

_EXCLUDED_FILE_PATTERNS = {
    ".env.*",
    "*_secret.*",
    "*_secrets.*",
    "id_ed25519*",
    "id_rsa*",
}


def _is_excluded(path: Path) -> bool:
    if any(part in _EXCLUDED_DIRS for part in path.parts):
        return True
    name = path.name
    return (
        name in _EXCLUDED_FILE_NAMES
        or path.suffix in _EXCLUDED_FILE_SUFFIXES
        or any(fnmatch.fnmatch(name, pattern) for pattern in _EXCLUDED_FILE_PATTERNS)
    )


def iter_python_files(source: Path) -> list[Path]:
    return [
        path
        for path in sorted(source.rglob("*.py"))
        if not _is_excluded(path.relative_to(source))
    ]


def module_path(source: Path, file_path: Path) -> str:
    rel = file_path.relative_to(source).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def safe_vendor_dir_name(source: Path) -> str:
    name = source.name.strip().replace("-", "_")
    return name if name.isidentifier() else "source"


def copy_source_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Vendored source path already exists: {destination}")

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if _is_excluded(Path(name))}

    shutil.copytree(source, destination, ignore=ignore)


def ensure_vendor_package(vendor_dir: Path) -> None:
    for path in (vendor_dir.parent, vendor_dir):
        init_file = path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8", newline="\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def _dependency_name(requirement: str) -> str:
    requirement = requirement.strip()
    for marker in ("[", "<", ">", "=", "!", "~", ";", " "):
        if marker in requirement:
            requirement = requirement.split(marker, 1)[0]
    return requirement.lower().replace("_", "-")


def append_dependency_files(
    env_dir: Path,
    env_name: str,
    dependencies: list[str],
) -> None:
    requirements = env_dir / "server" / "requirements.txt"
    if requirements.exists():
        content = requirements.read_text(encoding="utf-8")
        existing = {
            _dependency_name(line)
            for line in content.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        missing = [
            dependency
            for dependency in dependencies
            if _dependency_name(dependency) not in existing
        ]
        if missing:
            requirements.write_text(
                content.rstrip() + "\n" + "\n".join(missing) + "\n",
                encoding="utf-8",
                newline="\n",
            )

    pyproject = env_dir / "pyproject.toml"
    if not pyproject.exists():
        return

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.setdefault("project", {})
    project_dependencies = list(project.get("dependencies") or [])
    existing = {_dependency_name(dependency) for dependency in project_dependencies}
    for dependency in dependencies:
        if _dependency_name(dependency) not in existing:
            project_dependencies.append(dependency)
            existing.add(_dependency_name(dependency))
    project["dependencies"] = project_dependencies

    tool = data.setdefault("tool", {})
    setuptools = tool.setdefault("setuptools", {})
    package_data = setuptools.setdefault("package-data", {})
    vendor_data = list(package_data.get(env_name) or [])
    if "vendor/**/*" not in vendor_data:
        vendor_data.append("vendor/**/*")
    package_data[env_name] = vendor_data

    pyproject.write_text(tomli_w.dumps(data), encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class DetectedEnvironment:
    """A source environment class detected without importing user code."""

    source_type: str
    class_name: str
    module_path: str
    file_path: Path

    @property
    def qualified_name(self) -> str:
        return f"{self.module_path}:{self.class_name}"


class EnvironmentImporter(Protocol):
    source_type: str

    def detect(self, source: Path) -> list[DetectedEnvironment]:
        """Return environments supported by this importer."""
        ...

    def generate(
        self,
        *,
        source: Path,
        destination: Path,
        env_name: str,
        detected: DetectedEnvironment,
    ) -> None:
        """Generate an OpenEnv wrapper package."""
        ...


class ImporterRegistry:
    """Registry of deterministic environment importers."""

    def __init__(self, importers: list[EnvironmentImporter]):
        self._importers = importers

    @property
    def supported_types(self) -> list[str]:
        return [importer.source_type for importer in self._importers]

    def get(self, source_type: str) -> EnvironmentImporter:
        for importer in self._importers:
            if importer.source_type == source_type:
                return importer
        supported = ", ".join(self.supported_types)
        raise ValueError(
            f"Unsupported source type {source_type!r}. Supported: {supported}"
        )

    def detect(
        self,
        source: Path,
        source_type: str | None = None,
    ) -> list[tuple[EnvironmentImporter, DetectedEnvironment]]:
        importers = [self.get(source_type)] if source_type else self._importers
        matches: list[tuple[EnvironmentImporter, DetectedEnvironment]] = []
        for importer in importers:
            for detected in importer.detect(source):
                matches.append((importer, detected))
        return matches
