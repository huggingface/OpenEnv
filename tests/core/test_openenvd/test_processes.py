# SPDX-License-Identifier: BSD-3-Clause
"""Declared processes: workload sidecars in the episode sandbox, privileged runs apart."""

import asyncio
import json

import pytest
from openenv.core.openenvd.backends.local import LocalBackend
from openenv.core.openenvd.policy import OpenEnvDConfig, ProcessSpec
from openenv.core.openenvd.runtime import Runtime
from openenv.core.openenvd.surfaces import create_surface_app
from pydantic import ValidationError

LOCAL = {"enabled": True, "enforcement": {"backend": "local"}}


@pytest.mark.parametrize(
    "process",
    [
        {"trust": "workload"},
        {"trust": "workload", "argv": ["relative/harness"]},
        {"trust": "workload", "argv": ["/bin/h"], "asset": "oracle"},
        {"trust": "privileged"},
        {"trust": "privileged", "asset": "oracle", "argv": ["/bin/h"]},
        {"trust": "workload", "argv": ["/bin/h"], "env": {"OPENENVD_TOKEN": "x"}},
        {"trust": "workload", "argv": ["/bin/h"], "env": {"BAD-NAME": "x"}},
    ],
)
def test_process_shapes_are_validated(process):
    with pytest.raises(ValidationError):
        ProcessSpec.model_validate(process)


@pytest.mark.parametrize(
    ("name", "process", "message"),
    [
        ("worker", {"trust": "workload", "argv": ["/bin/h"]}, "invalid process name"),
        ("Harness", {"trust": "workload", "argv": ["/bin/h"]}, "invalid process name"),
        ("rubric", {"trust": "privileged", "asset": "rubric"}, "undeclared asset"),
    ],
)
def test_process_names_and_assets_are_checked(name, process, message):
    with pytest.raises(ValidationError, match=message):
        OpenEnvDConfig.model_validate({**LOCAL, "processes": {name: process}})


def test_oracle_asset_is_an_implicit_privileged_process():
    config = OpenEnvDConfig.model_validate(
        {
            **LOCAL,
            "privileged_assets": {"oracle": "grade.sh", "rubric": "rubric.sh"},
            "processes": {
                "rubric": {"trust": "privileged", "asset": "rubric", "args": ["-v"]},
                "harness": {"trust": "workload", "argv": ["/usr/bin/harness"]},
            },
        }
    )
    assert config.privileged_process("oracle").asset == "oracle"
    assert config.privileged_process("rubric").args == ("-v",)
    assert config.privileged_process("harness") is None
    assert config.privileged_process("missing") is None


# --- end to end on the local backend --------------------------------------


@pytest.fixture
def local_python(worker_python, monkeypatch):
    python, _ = worker_python
    monkeypatch.setattr(LocalBackend, "python", property(lambda self: str(python)))
    return str(python)


async def _wait_for(predicate, timeout=10):
    deadline = asyncio.get_running_loop().time() + timeout
    while not await predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError
        await asyncio.sleep(0.2)


async def test_local_episode_runs_sidecar_and_privileged_processes(
    tmp_path, local_python
):
    workspace = tmp_path / "workspace-seed"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("42")
    assets = tmp_path / "assets"
    assets.mkdir(mode=0o700)
    for name, body in (("oracle.sh", "echo oracle-ok"), ("rubric.sh", 'cat "$1"')):
        script = assets / name
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o700)
    config = OpenEnvDConfig.model_validate(
        {
            **LOCAL,
            "surfaces": {
                "orchestrator": {"allow_lifecycle": True},
                "agent": {"tools": ["echo_*"]},
                "grader": {
                    "tools": ["grader.*"],
                    "fs_read": ["/workspace/**"],
                    "allow_privileged_exec": True,
                },
            },
            "privileged_assets": {"oracle": "oracle.sh", "rubric": "rubric.sh"},
            "processes": {
                "sidecar": {
                    "trust": "workload",
                    "argv": [
                        local_python,
                        "-c",
                        "import os, pathlib; pathlib.Path('sidecar.txt')"
                        ".write_text(os.environ['MARK'])",
                    ],
                    "env": {"MARK": "sidecar-ran"},
                },
                "rubric": {
                    "trust": "privileged",
                    "asset": "rubric",
                    "args": ["answer.txt"],
                },
            },
        }
    )
    runtime = Runtime(
        config,
        "echo_env.server.echo_environment:EchoEnvironment",
        "openenv.core.env_server.mcp_types:CallToolAction",
        workspace,
        asset_root=assets,
        timeout_s=30,
    )
    await runtime.start()
    try:
        await runtime.reset({})
        tools = await runtime.request(
            "mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        assert any(t["name"] == "echo_message" for t in tools["result"]["tools"])

        async def sidecar_wrote():
            changes = await runtime.fs_diff()
            return any(c["path"] == "/workspace/sidecar.txt" for c in changes)

        await _wait_for(sidecar_wrote)
        assert await runtime.read_file("/workspace/sidecar.txt") == "sidecar-ran"

        rubric = await runtime.run_process("rubric")
        assert rubric == {"returncode": 0, "stdout": "42", "stderr": ""}
        oracle = await runtime.run_oracle()
        assert oracle["stdout"].strip() == "oracle-ok"
        with pytest.raises(PermissionError):
            await runtime.run_process("sidecar")

        events = runtime.collector.trajectory()
        assert {"kind": "spawn", "process": "sidecar"} in [e["data"] for e in events]
        assert not (workspace / "sidecar.txt").exists()
        assert runtime.grader_backend is None
    finally:
        await runtime.close()


async def test_grader_tools_list_run_process_only_with_privileged_exec(
    tmp_path, local_python
):
    from fastapi.testclient import TestClient

    workspace = tmp_path / "seed"
    workspace.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir(mode=0o700)
    for allowed in (True, False):
        config = OpenEnvDConfig.model_validate(
            {
                **LOCAL,
                "surfaces": {
                    "grader": {
                        "tools": ["grader.*"],
                        "allow_privileged_exec": allowed,
                    }
                },
            }
        )
        runtime = Runtime(
            config,
            "echo_env.server.echo_environment:EchoEnvironment",
            "openenv.core.env_server.mcp_types:CallToolAction",
            workspace,
            asset_root=assets,
            timeout_s=30,
        )
        app = create_surface_app(runtime, {"grader": "grader-token"})
        with TestClient(app) as client:
            response = client.post(
                "/mcp/grader",
                headers={"Authorization": "Bearer grader-token"},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
        names = {tool["name"] for tool in response.json()["result"]["tools"]}
        assert ("grader.run_process" in names) is allowed
        assert ("grader.run_oracle" in names) is allowed
        assert json.dumps(response.json())
