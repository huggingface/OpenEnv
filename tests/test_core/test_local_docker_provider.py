import subprocess
import time
from types import SimpleNamespace

from openenv.core.containers.runtime import LocalDockerProvider


def test_start_container_forwards_volume_mounts(monkeypatch):
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

    base_url = provider.start_container(
        "retro-env:latest",
        port=8123,
        volumes={
            "/host/roms": {
                "bind": "/roms",
                "mode": "ro",
            }
        },
    )

    assert base_url == "http://localhost:8123"
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
        "retro-env:latest",
    ]
