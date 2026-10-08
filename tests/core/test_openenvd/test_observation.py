# SPDX-License-Identifier: BSD-3-Clause
"""Downloaded observations compare content, and streams remain explicit."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from openenv.core.openenvd.observation import Workspace
from openenv.core.openenvd.policy import OpenEnvDConfig
from openenv.core.openenvd.runtime import Runtime


def runtime_for(tmp_path, streams=()):
    return Runtime(
        OpenEnvDConfig.model_validate(
            {
                "enabled": True,
                "openshell": {"image": "test:latest", "gateway": "test"},
                "surfaces": {"observer": {"stream": list(streams)}},
            }
        ),
        "unused",
        "unused",
        tmp_path,
        asset_root=tmp_path,
    )


def test_transfer_mtime_does_not_make_unchanged_files_different(tmp_path):
    seed = tmp_path / "workspace"
    seed.mkdir()
    file = seed / "file"
    file.write_text("same bytes")
    workspace = Workspace(seed, tmp_path / "snapshot")
    workspace.capture()
    os.utime(file, (1, 1))
    assert workspace.diff() == []
    file.write_text("different bytes")
    assert workspace.diff() == [{"path": str(file), "kind": "modify"}]


async def test_unrequested_filesystem_observation_never_downloads(tmp_path):
    runtime = runtime_for(tmp_path)
    runtime.proc = SimpleNamespace(returncode=None)
    runtime.started_at = float("inf")
    runtime.event_task = Mock(done=lambda: False)
    runtime._sync_workspace = AsyncMock()
    with patch(
        "openenv.core.openenvd.runtime.asyncio.sleep",
        AsyncMock(side_effect=[None, asyncio.CancelledError]),
    ):
        with pytest.raises(asyncio.CancelledError):
            await runtime._observe_loop()
    runtime._sync_workspace.assert_not_awaited()


async def test_filesystem_observer_uses_fresh_downloads(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "file").write_text("initial")
    runtime = runtime_for(seed, ["fs_diff"])
    runtime.workspace = Workspace(seed, tmp_path / "snapshot")
    runtime.workspace.capture()
    runtime.proc = SimpleNamespace(returncode=None)
    runtime.started_at = float("inf")
    runtime.event_task = Mock(done=lambda: False)
    runtime._sync_workspace = AsyncMock(
        side_effect=lambda: (seed / "file").write_text("changed")
    )
    with patch(
        "openenv.core.openenvd.runtime.asyncio.sleep",
        AsyncMock(side_effect=[None, asyncio.CancelledError]),
    ):
        with pytest.raises(asyncio.CancelledError):
            await runtime._observe_loop()
    runtime._sync_workspace.assert_awaited_once()
    assert [e.data for e in runtime.collector.events] == [
        {"path": "file", "kind": "modify"}
    ]


async def test_disconnected_worker_deletes_sandbox(tmp_path):
    runtime = runtime_for(tmp_path)
    runtime.proc = SimpleNamespace(returncode=1)
    runtime._stop = AsyncMock()
    with patch("openenv.core.openenvd.runtime.asyncio.sleep", AsyncMock()):
        await runtime._observe_loop()
    runtime._stop.assert_awaited_once_with(capture=False)
