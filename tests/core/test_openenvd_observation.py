# SPDX-License-Identifier: BSD-3-Clause
"""Filesystem observation and grading use different comparison points."""

import asyncio
import shutil
from unittest.mock import patch

import pytest
from openenv.core.openenvd.observation import Workspace
from openenv.core.openenvd.policy import OpenEnvDConfig
from openenv.core.openenvd.runtime import Runtime


@pytest.mark.asyncio
async def test_observer_tracks_samples_while_grader_tracks_baseline(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("modified", "removed", "unchanged"):
        (workspace / name).write_text("initial")
    runtime = Runtime(
        OpenEnvDConfig(enabled=True, surfaces={"observer": {"stream": ["fs_diff"]}}),
        "unused:factory",
        "unused:action",
        workspace,
        uid=65530,
        gid=65530,
        asset_root=tmp_path,
    )
    runtime.workspace = Workspace(workspace, tmp_path / "snapshot")
    runtime.workspace.capture()
    runtime.cgroup.path = tmp_path / "cgroup"
    runtime.cgroup.path.mkdir()
    (runtime.cgroup.path / "cgroup.procs").write_text("")
    grader_diffs = []

    async def change_workspace(_):
        if not grader_diffs:
            (workspace / "created").write_text("new")
            (workspace / "modified").write_text("changed")
            (workspace / "removed").unlink()
        elif len(grader_diffs) == 1:
            (workspace / "created").unlink()
            for name in ("modified", "removed"):
                shutil.copy2(runtime.workspace.snapshot / name, workspace / name)
        else:
            raise asyncio.CancelledError
        grader_diffs.append(runtime.workspace.diff())

    with (
        patch("openenv.core.openenvd.runtime.asyncio.sleep", change_workspace),
        patch.object(runtime, "_reap_orphans"),
        pytest.raises(asyncio.CancelledError),
    ):
        await runtime._observe_loop()

    assert grader_diffs == [
        [
            {"path": str(workspace / "created"), "kind": "create"},
            {"path": str(workspace / "modified"), "kind": "modify"},
            {"path": str(workspace / "removed"), "kind": "delete"},
        ],
        [],
    ]
    assert [event.data for event in runtime.collector.events] == [
        {"path": "created", "kind": "create"},
        {"path": "modified", "kind": "modify"},
        {"path": "removed", "kind": "delete"},
        {"path": "created", "kind": "delete"},
        {"path": "modified", "kind": "modify"},
        {"path": "removed", "kind": "create"},
    ]


@pytest.mark.asyncio
async def test_disabled_observation_does_not_scan_workspace(tmp_path):
    from unittest.mock import AsyncMock, Mock

    runtime = Runtime(
        OpenEnvDConfig(enabled=True),
        "unused",
        "unused",
        tmp_path,
        uid=65530,
        gid=65530,
        asset_root=tmp_path,
    )
    runtime.workspace = Mock(baseline={})
    with (
        patch.object(runtime, "_reap_orphans"),
        patch(
            "openenv.core.openenvd.runtime.asyncio.sleep",
            AsyncMock(side_effect=[None, asyncio.CancelledError]),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await runtime._observe_loop()
    runtime.workspace.scan.assert_not_called()
    runtime.workspace.disk_usage.assert_not_called()
    assert runtime.collector.events == []


@pytest.mark.asyncio
async def test_slow_scan_does_not_block_reset_or_publish_across_episodes(tmp_path):
    import threading
    from unittest.mock import AsyncMock, Mock

    from openenv.core.openenvd.observation import Collector

    runtime = Runtime(
        OpenEnvDConfig(enabled=True, surfaces={"observer": {"stream": ["fs_diff"]}}),
        "unused",
        "unused",
        tmp_path,
        uid=65530,
        gid=65530,
        asset_root=tmp_path,
    )
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def scan():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return {"previous-episode": "content"}

    runtime.workspace = Mock(baseline={}, scan=scan)
    with (
        patch.object(runtime, "_reap_orphans"),
        patch(
            "openenv.core.openenvd.runtime.asyncio.sleep",
            AsyncMock(side_effect=[None, asyncio.CancelledError]),
        ),
    ):
        monitor = asyncio.create_task(runtime._observe_loop())
        try:
            await asyncio.wait_for(started.wait(), 2)
            await asyncio.wait_for(runtime.lock.acquire(), 0.2)
            try:
                runtime.collector = Collector()  # Reset's episode boundary.
            finally:
                runtime.lock.release()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await monitor
            assert runtime.collector.events == []
        finally:
            release.set()
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)


def test_reset_snapshot_does_not_require_observation_fingerprints(tmp_path):
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    file = workspace_path / "large-for-observation"
    file.write_text("original")
    workspace = Workspace(workspace_path, tmp_path / "snapshot", max_file_bytes=4)
    with patch.object(
        workspace, "scan", side_effect=AssertionError("unexpected hashing")
    ):
        workspace.capture()
        file.write_text("changed")
        with patch("os.chown"):
            workspace.restore()
    assert file.read_text() == "original"
    # Fingerprinting still enforces its limit when actually requested.
    with pytest.raises(RuntimeError, match="observation limit"):
        workspace.diff()


def test_first_grader_diff_uses_original_snapshot_after_workload_changes(tmp_path):
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    file = workspace_path / "file"
    file.write_text("original")
    workspace = Workspace(workspace_path, tmp_path / "snapshot")
    workspace.capture()
    assert "baseline" not in workspace.__dict__
    file.write_text("changed")
    assert workspace.diff() == [{"path": str(file), "kind": "modify"}]
