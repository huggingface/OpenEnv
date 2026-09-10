import subprocess
import time
from types import SimpleNamespace

import pytest

from openenv.core.containers.runtime import LocalDockerProvider


def make_provider(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="container-id\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    provider = LocalDockerProvider()
    monkeypatch.setattr(
        provider, "_generate_container_name", lambda _: "test-container"
    )
    return provider, commands


def test_start_container_forwards_volume_mounts(monkeypatch):
    provider, commands = make_provider(monkeypatch)

    base_url = provider.start_container(
        "retro-env:latest",
        port=8123,
        volumes={
            "/host/roms": {
                "bind": "/roms",
                "mode": "ro",
            },
            "game-data": {
                "bind": "/data",
            },
        },
    )

    assert base_url == "http://localhost:8123"
    assert commands[0] == ["docker", "version"]
    assert commands[1] == [
        "docker",
        "run",
        "-d",
        "--name",
        "test-container",
        "-p",
        "8123:8000",
        "--volume",
        "/host/roms:/roms:ro",
        "--volume",
        "game-data:/data",
        "retro-env:latest",
    ]


def test_start_container_rejects_volume_without_bind(monkeypatch):
    provider, _ = make_provider(monkeypatch)

    with pytest.raises(
        ValueError,
        match="Volume config for '/host/roms' must include a non-empty 'bind'",
    ):
        provider.start_container(
            "retro-env:latest",
            port=8123,
            volumes={"/host/roms": {}},
        )


def test_start_container_rejects_unsupported_kwargs(monkeypatch):
    provider, _ = make_provider(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported kwargs for LocalDockerProvider"):
        provider.start_container(
            "retro-env:latest",
            port=8123,
            privileged=True,
        )
