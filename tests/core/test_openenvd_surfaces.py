# SPDX-License-Identifier: BSD-3-Clause
"""Public boundary tests with one shared episode and distinct principals."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from openenv.core.openenvd.observation import Collector, Workspace
from openenv.core.openenvd.policy import ObservationEventType, OpenEnvDConfig, Principal
from openenv.core.openenvd.runtime import Runtime
from openenv.core.openenvd.surfaces import create_surface_app


@pytest.fixture
def runtime(tmp_path):
    config = OpenEnvDConfig.model_validate(
        {
            "enabled": True,
            "surfaces": {
                "agent": {"tools": ["echo"]},
                "grader": {
                    "tools": ["echo", "grader.*"],
                    "fs_read": [str(tmp_path / "workspace") + "/**"],
                },
                "orchestrator": {"allow_lifecycle": True},
                "observer": {"stream": ["process"]},
            },
        }
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("answer")
    instance = Runtime(
        config,
        "unused:factory",
        "unused:action",
        workspace,
        uid=12345,
        gid=12345,
        asset_root=tmp_path,
    )
    instance.workspace = Workspace(workspace, tmp_path / "snapshot")
    instance.workspace.capture()
    instance.directory = tmp_path
    instance.start = AsyncMock()
    instance.close = AsyncMock()
    instance.proc = SimpleNamespace(returncode=None)
    instance.reset = AsyncMock(return_value={"done": False, "reward": 0})

    async def request(operation, data=None):
        if operation == "state":
            return {"step_count": 2}
        if data["method"] == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": data["id"],
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo",
                            "inputSchema": {"type": "object"},
                        },
                        {"name": "grader.leak"},
                        {"name": "reset"},
                    ]
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": data["id"],
            "result": {"content": [{"type": "text", "text": "hello"}]},
        }

    instance.request = AsyncMock(side_effect=request)
    return instance


def client_for(runtime):
    return TestClient(
        create_surface_app(
            runtime,
            {
                Principal.GRADER: "grade-secret",
                Principal.OBSERVER: "observe-secret",
                Principal.ORCHESTRATOR: "control-secret",
            },
        )
    )


def test_agent_surface_excludes_privileges_and_preserves_domain_results(runtime):
    with client_for(runtime) as client:
        result = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        ).json()
        assert [tool["name"] for tool in result["result"]["tools"]] == ["echo"]
        for name in ("grader.read_file", "reset", "unknown"):
            result = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": name},
                },
            ).json()
            assert result["error"]["code"] == -32602
        result = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "echo"},
            },
        ).json()
        assert result == {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {"content": [{"type": "text", "text": "hello"}]},
        }


def test_grader_requires_its_own_credential(runtime):
    with client_for(runtime) as client:
        for token in (None, "observe-secret", "control-secret"):
            headers = {"Authorization": "Bearer " + token} if token else {}
            assert (
                client.post("/mcp/grader", headers=headers, json={}).status_code == 401
            )
        response = client.post(
            "/mcp/grader",
            headers={"Authorization": "Bearer grade-secret"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        ).json()
        names = [tool["name"] for tool in response["result"]["tools"]]
        assert "grader.get_full_state" in names
        assert "grader.run_oracle" not in names
        assert "grader.leak" not in names


def test_websocket_grader_reads_same_episode(runtime):
    with client_for(runtime) as client:
        with client.websocket_connect(
            "/mcp/grader", headers={"Authorization": "Bearer grade-secret"}
        ) as ws:
            ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "grader.get_full_state"},
                }
            )
            assert (
                '"step_count": 2' in ws.receive_json()["result"]["content"][0]["text"]
            )
        with client.websocket_connect(
            "/ws", headers={"Authorization": "Bearer control-secret"}
        ) as ws:
            ws.send_json({"type": "reset", "data": {"seed": 4}})
            assert ws.receive_json()["type"] == "observation"
        runtime.reset.assert_awaited_once_with({"seed": 4})


def test_observer_emits_only_declared_stream(runtime):
    runtime.collector.record(ObservationEventType.FS_CHANGE, {"path": "hidden"})
    runtime.collector.record(ObservationEventType.PROCESS, {"kind": "spawn"})
    with client_for(runtime) as client:
        with client.websocket_connect(
            "/observe", headers={"Authorization": "Bearer observe-secret"}
        ) as ws:
            event = ws.receive_json()
            assert event["type"] == "process"
            assert event["seq"] == 1


def test_read_file_rejects_symlinks_and_path_traversal(runtime, tmp_path):
    assert runtime.read_file(str(runtime.workspace_path / "answer.txt")) == "answer"
    (runtime.workspace_path / "escape").symlink_to(tmp_path)
    for path in (
        runtime.workspace_path / "escape" / "secret",
        runtime.workspace_path / ".." / "secret",
        Path("/etc/passwd"),
    ):
        with pytest.raises((OSError, PermissionError)):
            runtime.read_file(str(path))


def test_workspace_restore_and_diff_do_not_follow_links(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    (workspace / "file").write_text("initial")
    snapshot = Workspace(workspace, tmp_path / "snapshot")
    snapshot.capture()
    (workspace / "file").write_text("changed")
    (workspace / "escape").symlink_to(outside)
    assert {item["kind"] for item in snapshot.diff()} == {"create", "modify"}
    snapshot.restore()
    assert (workspace / "file").read_text() == "initial"
    assert not (workspace / "escape").exists()
    assert outside.read_text() == "untouched"


def test_tokens_cannot_be_shared(runtime):
    with pytest.raises(ValueError, match="distinct"):
        create_surface_app(runtime, {principal: "same" for principal in Principal})


def test_collector_copies_payload():
    collector = Collector()
    data = {"nested": {"value": 1}}
    collector.record(ObservationEventType.PROCESS, data)
    data["nested"]["value"] = 2
    assert collector.trajectory()[0]["data"]["nested"]["value"] == 1


def test_http_mcp_sessions_do_not_reset_episode(runtime):
    with client_for(runtime) as client:
        create = client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "openenv/session/create"}
        ).json()
        session_id = create["result"]["session_id"]
        result = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {"session_id": session_id},
            },
        ).json()
        assert result["result"]["tools"][0]["name"] == "echo"
        closed = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "openenv/session/close",
                "params": {"session_id": session_id},
            },
        ).json()
        assert closed["result"]["closed"]
        runtime.reset.assert_not_called()


@pytest.mark.asyncio
async def test_existing_http_mcp_client_works_without_new_headers(runtime):
    import httpx
    from openenv.core.mcp_client import MCPToolClient

    client = MCPToolClient(base_url="http://test")
    client.use_production_mode = True
    client._http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=create_surface_app(
                runtime,
                {
                    Principal.GRADER: "grade-secret",
                    Principal.OBSERVER: "observe-secret",
                    Principal.ORCHESTRATOR: "control-secret",
                },
            )
        )
    )
    try:
        tools = await client.list_tools()
        assert [tool.name for tool in tools] == ["echo"]
    finally:
        await client.close()


def test_invalid_request_id_does_not_escape_error_handler(runtime):
    with client_for(runtime) as client:
        response = client.post("/mcp", json={"id": {"bad": "id"}})
        assert response.status_code == 200
        assert response.json()["id"] is None
        assert "error" in response.json()


def test_snapshot_never_silently_ignores_large_file_contents(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "large").write_bytes(b"a" * 17)
    snapshot = Workspace(workspace, tmp_path / "snapshot", max_file_bytes=16)
    with pytest.raises(RuntimeError, match="observation limit"):
        snapshot.scan()


@pytest.mark.asyncio
async def test_shared_mcp_validation_keeps_sessions_bound_to_their_surface():
    from unittest.mock import AsyncMock

    from openenv.core.openenvd.mcp import mcp_handler
    from openenv.core.openenvd.policy import SurfacePolicy

    dispatch = AsyncMock(
        return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": [{"name": "echo"}, {"name": "private"}]},
        }
    )
    agent = mcp_handler(SurfacePolicy(principal="agent", tools=["echo"]), dispatch)
    grader = mcp_handler(SurfacePolicy(principal="grader", tools=["echo"]), dispatch)

    def request(method, **params):
        return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

    session = (await agent(request("openenv/session/create")))["result"]["session_id"]
    for rpc, data in [
        (grader, request("tools/call", name="echo", session_id=session)),
        (agent, request("tools/call", name="private")),
        (agent, request("tools/call", name="echo", arguments=[])),
        (agent, request("tools/call", name="echo", session_id=[])),
        (agent, request("reset")),
    ]:
        assert (await rpc(data))["error"]
    dispatch.assert_not_awaited()
    listed = await agent(request("tools/list", session_id=session))
    assert listed["result"]["tools"] == [{"name": "echo"}]
    await agent(request("openenv/session/close", session_id=session))
    assert (await agent(request("tools/list", session_id=session)))["error"]
    # Worker errors are preserved, not mistaken for successful tool lists.
    dispatch.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32603, "message": "failed"},
        "result": None,
    }
    assert (await agent(request("tools/list")))["error"]["code"] == -32603
