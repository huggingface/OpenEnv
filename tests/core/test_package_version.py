"""Tests for package version resolution."""

from __future__ import annotations

from importlib import metadata

import openenv


def test_load_package_version_prefers_openenv_core(monkeypatch) -> None:
    calls: list[str] = []

    def fake_version(distribution_name: str) -> str:
        calls.append(distribution_name)
        if distribution_name == "openenv-core":
            return "0.2.3"
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(openenv.metadata, "version", fake_version)

    assert openenv._load_package_version() == "0.2.3"
    assert calls == ["openenv-core"]


def test_load_package_version_falls_back_to_openenv(monkeypatch) -> None:
    def fake_version(distribution_name: str) -> str:
        if distribution_name == "openenv-core":
            raise metadata.PackageNotFoundError
        if distribution_name == "openenv":
            return "0.2.0"
        raise AssertionError(f"Unexpected distribution name: {distribution_name}")

    monkeypatch.setattr(openenv.metadata, "version", fake_version)

    assert openenv._load_package_version() == "0.2.0"


def test_load_package_version_returns_zero_when_uninstalled(monkeypatch) -> None:
    monkeypatch.setattr(
        openenv.metadata,
        "version",
        lambda distribution_name: (_ for _ in ()).throw(metadata.PackageNotFoundError),
    )

    assert openenv._load_package_version() == "0.0.0"
