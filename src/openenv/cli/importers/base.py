"""Shared importer registry types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


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
        raise ValueError(f"Unsupported source type {source_type!r}. Supported: {supported}")

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
