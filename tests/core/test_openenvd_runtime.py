# SPDX-License-Identifier: BSD-3-Clause
"""Episode ownership and fail-closed OpenShell lifecycle."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openenv.core.openenvd.isolation import IsolationError
from openenv.core.openenvd.observation import Workspace
from openenv.core.openenvd.policy import OpenEnvDConfig
from openenv.core.openenvd.runtime import Runtime


def make_runtime(tmp_path, **overrides):
    config = OpenEnvDConfig.model_validate(
        {
            "enabled": True,
            "openshell": {"image": "openenv-test:latest", "gateway": "test"},
            **overrides,
        }
    )
    return Runtime(
        config, "unused:factory", "unused:action", tmp_path, asset_root=tmp_path
    )


def test_runtime_requires_openshell_and_does_not_fall_back(tmp_path):
    with pytest.raises(ValueError, match="openshell"):
        Runtime(
            OpenEnvDConfig(enabled=True),
            "unused",
            "unused",
            tmp_path,
            asset_root=tmp_path,
        )


@pytest.mark.parametrize("stream", ["network", "resource"])
def test_runtime_rejects_unimplemented_telemetry(tmp_path, stream):
    with pytest.raises(ValueError, match="not yet available"):
        make_runtime(tmp_path, surfaces={"observer": {"stream": [stream]}})


async def test_reset_refuses_new_sandbox_until_teardown_verified(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime._stop = AsyncMock(side_effect=IsolationError("delete not confirmed"))
    runtime._spawn = AsyncMock()
    with pytest.raises(IsolationError, match="not confirmed"):
        await runtime.reset({})
    runtime._spawn.assert_not_awaited()


async def test_failed_reset_stops_new_sandbox(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime._stop = AsyncMock()
    runtime._spawn = AsyncMock(side_effect=IsolationError("startup failed"))
    with pytest.raises(IsolationError, match="startup failed"):
        await runtime.reset({})
    assert runtime._stop.await_count == 2


async def test_reset_starts_new_backend_and_monitor(tmp_path):
    runtime = make_runtime(tmp_path)
    previous = runtime.backend
    runtime._stop = AsyncMock()
    runtime._spawn = AsyncMock()
    runtime._request = AsyncMock(return_value={"done": False})
    runtime._monitor = AsyncMock()
    assert await runtime.reset({"seed": 42}) == {"done": False}
    assert runtime.backend is not previous
    await runtime.monitor
    runtime._request.assert_awaited_once_with("reset", {"seed": 42})


async def test_snapshot_failure_still_deletes_sandbox(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.backend.id = "sandbox-id"
    runtime.workspace = object()
    runtime._sync_workspace = AsyncMock(side_effect=IsolationError("cannot download"))
    runtime.backend.close = AsyncMock()
    with pytest.raises(IsolationError, match="snapshot failed"):
        await runtime.stop()
    runtime.backend.close.assert_awaited_once()


async def test_failed_cleanup_keeps_private_state_for_retry(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.directory = tmp_path / "private"
    runtime.directory.mkdir()
    (runtime.directory / "recovery").write_text("owned sandbox identity")
    runtime.backend.close = AsyncMock(side_effect=IsolationError("still deleting"))
    with pytest.raises(IsolationError):
        await runtime.close()
    assert runtime.directory.exists()
    runtime.backend.close = AsyncMock()
    await runtime.close()
    assert runtime.directory is None


async def test_start_copies_seed_without_mutating_source(tmp_path):
    seed = tmp_path / "source"
    seed.mkdir()
    (seed / "initial").write_text("baseline")
    runtime = make_runtime(seed)
    runtime._spawn = AsyncMock()
    runtime._monitor = AsyncMock()
    try:
        await runtime.start()
        assert runtime.workspace.snapshot.name == "workspace"
        (seed / "initial").write_text("later local change")
        assert (runtime.workspace.snapshot / "initial").read_text() == "baseline"
        runtime._stop = AsyncMock()
        runtime._request = AsyncMock(return_value={})
        await runtime.reset({})
        assert (seed / "initial").read_text() == "later local change"
    finally:
        await runtime.close()


async def test_seed_never_follows_links_to_private_assets(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("private")
    (seed / "link").symlink_to(secret)
    runtime = make_runtime(seed)
    runtime._spawn = AsyncMock()
    with pytest.raises(ValueError, match="regular files"):
        await runtime.start()
    runtime._spawn.assert_not_awaited()


async def test_control_response_and_events_are_separate(tmp_path):
    runtime = make_runtime(tmp_path)
    reader = asyncio.StreamReader()
    runtime.proc = SimpleNamespace(stdout=reader)
    runtime.pending = asyncio.get_running_loop().create_future()
    reader.feed_data(json.dumps({"event": {"kind": "tool_call"}}).encode() + b"\n")
    reader.feed_data(b'{"result": {"ready": true}}\n')
    reader.feed_eof()
    await runtime._read_frames()
    assert runtime.pending.result() == {"result": {"ready": True}}
    assert runtime.collector.events[0].data["source"] == "workload"


async def test_protocol_failure_cannot_leave_request_hanging(tmp_path):
    runtime = make_runtime(tmp_path)
    reader = asyncio.StreamReader()
    runtime.proc = SimpleNamespace(stdout=reader)
    runtime.pending = asyncio.get_running_loop().create_future()
    reader.feed_data(b"not a control frame\n")
    await runtime._read_frames()
    with pytest.raises(IsolationError, match="invalid data"):
        await runtime.pending


async def test_grader_snapshot_uses_logical_paths_and_rejects_symlinks(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "answer").write_text("baseline")
    runtime = make_runtime(seed, surfaces={"grader": {"fs_read": ["/workspace/**"]}})
    runtime.workspace = Workspace(seed, tmp_path / "snapshot")
    runtime.workspace.capture()
    runtime._snapshot_valid = True
    (seed / "answer").write_text("changed")
    assert await runtime.read_file("/workspace/answer") == "changed"
    assert await runtime.fs_diff() == [{"path": "/workspace/answer", "kind": "modify"}]
    (seed / "escape").symlink_to(tmp_path / "snapshot" / "answer")
    with pytest.raises(OSError):
        await runtime.read_file("/workspace/escape")
    with pytest.raises(PermissionError):
        await runtime.read_file("/workspace/../snapshot/answer")


async def test_cancelled_stop_still_confirms_sandbox_deletion(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.backend.id = "live"
    runtime.workspace = object()
    started, release = asyncio.Event(), asyncio.Event()

    async def snapshot():
        started.set()
        await release.wait()

    runtime._sync_workspace = snapshot
    runtime.backend.close = AsyncMock()
    stop = asyncio.create_task(runtime.stop())
    await started.wait()
    stop.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await stop
    runtime.backend.close.assert_awaited_once()


async def test_timed_out_request_deletes_without_snapshot(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.proc = SimpleNamespace(returncode=None)
    runtime._request = AsyncMock(side_effect=asyncio.TimeoutError)
    runtime._stop = AsyncMock()
    with pytest.raises(asyncio.TimeoutError):
        await runtime.request("step", {})
    runtime._stop.assert_awaited_once_with(capture=False)


async def test_oracle_output_limit_reaps_process_without_hanging(tmp_path):
    runtime = make_runtime(tmp_path)
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "while true; do printf 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'; done",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    with pytest.raises(RuntimeError, match="exceeds"):
        await asyncio.wait_for(runtime._oracle_result(proc), 5)
    assert proc.returncode is not None


async def test_oracle_gets_separate_sandbox_and_never_changes_agent_seed(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer").write_text("candidate")
    runtime = make_runtime(
        workspace, surfaces={"grader": {"allow_privileged_exec": True}}
    )
    runtime.workspace = SimpleNamespace(path=workspace)
    runtime._snapshot_valid = True
    runtime.directory = tmp_path / "daemon"
    runtime.directory.mkdir()
    oracle = tmp_path / "oracle"
    oracle.write_text("private oracle")
    runtime.assets = {"oracle": oracle}
    captured = {}

    async def start(seed, directory):
        captured["files"] = {
            str(p.relative_to(seed)): p.read_text()
            for p in seed.rglob("*")
            if p.is_file()
        }

    grader = SimpleNamespace(
        start=AsyncMock(side_effect=start), spawn=AsyncMock(), close=AsyncMock()
    )
    monkeypatch.setattr(
        "openenv.core.openenvd.runtime.OpenShellSandbox", lambda *a, **kw: grader
    )
    runtime._oracle_result = AsyncMock(return_value={"returncode": 0})
    assert await runtime.run_oracle() == {"returncode": 0}
    assert captured["files"]["answer"] == "candidate"
    secret = next(name for name in captured["files"] if name.endswith("/oracle"))
    assert captured["files"][secret] == "private oracle"
    assert grader.spawn.await_args.args[0] == ["/sandbox/workspace/" + secret]
    assert set(p.name for p in workspace.iterdir()) == {"answer"}
    grader.close.assert_awaited_once()
    assert runtime.grader_backend is None


async def test_forced_teardown_never_grades_stale_workspace(tmp_path):
    runtime = make_runtime(
        tmp_path, surfaces={"grader": {"fs_read": ["/workspace/**"]}}
    )
    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        await runtime.read_file("/workspace/answer")


async def test_failed_refresh_invalidates_previous_grading_copy(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.directory = tmp_path / "daemon"
    runtime.directory.mkdir()
    runtime._snapshot_valid = True
    runtime.backend.download = AsyncMock(side_effect=IsolationError("transfer failed"))
    with pytest.raises(IsolationError):
        await runtime._sync_workspace()
    assert not runtime._snapshot_valid
    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        await runtime._refresh_snapshot()


def test_private_snapshot_cleanup_handles_readonly_dirs_without_following_links(
    tmp_path,
):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("preserved")
    outside.chmod(0o555)
    snapshot = tmp_path / "snapshot"
    readonly = snapshot / "readonly"
    readonly.mkdir(parents=True)
    (readonly / "answer").write_text("answer")
    (readonly / "link").symlink_to(outside, target_is_directory=True)
    readonly.chmod(0o555)
    snapshot.chmod(0o555)
    Runtime._remove_private_tree(snapshot)
    assert not snapshot.exists()
    assert (outside / "keep").read_text() == "preserved"
    assert outside.stat().st_mode & 0o777 == 0o555
    outside.chmod(0o700)
