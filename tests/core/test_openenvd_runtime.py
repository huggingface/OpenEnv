# SPDX-License-Identifier: BSD-3-Clause
"""Runtime lifecycle failures and Linux structural isolation."""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from openenv.core.openenvd.isolation import (
    detect_capabilities,
    IsolationCapabilities,
    IsolationError,
)
from openenv.core.openenvd.policy import OpenEnvDConfig
from openenv.core.openenvd.runtime import Runtime


@pytest.mark.asyncio
async def test_runtime_refuses_missing_isolation(tmp_path):
    runtime = Runtime(
        OpenEnvDConfig(enabled=True),
        "unused:factory",
        "unused:action",
        tmp_path,
        uid=65530,
        gid=65530,
        asset_root=tmp_path,
    )
    with patch(
        "openenv.core.openenvd.runtime.detect_capabilities",
        return_value=IsolationCapabilities(False, False),
    ):
        with pytest.raises(IsolationError):
            await runtime.start()
    assert runtime.directory is None
    assert runtime.proc is None


@pytest.mark.asyncio
async def test_failed_reset_stops_worker(tmp_path):
    runtime = Runtime(
        OpenEnvDConfig(enabled=True),
        "unused:factory",
        "unused:action",
        tmp_path,
        uid=65530,
        gid=65530,
        asset_root=tmp_path,
    )
    runtime.workspace = type("Workspace", (), {"restore": lambda self: None})()
    runtime._stop = AsyncMock()
    runtime._spawn = AsyncMock(side_effect=RuntimeError("startup failed"))
    with patch("os.chown"):
        with pytest.raises(RuntimeError, match="startup failed"):
            await runtime.reset({})
    assert runtime._stop.await_count == 2


@pytest.mark.asyncio
async def test_linux_runtime_reset_restores_workspace_and_replaces_process(tmp_path):
    if sys.platform != "linux" or os.geteuid() != 0:
        pytest.skip("Linux root required")
    if not detect_capabilities().can_unshare_net:
        pytest.skip("network namespace capability required")
    if not os.access("/sys/fs/cgroup", os.W_OK):
        pytest.skip("writable cgroup v2 required")
    # Keep importable environment code outside pytest's private temporary root.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="openenvd-linux-", dir="/tmp") as directory:
        root = Path(directory)
        root.chmod(0o755)
        workspace = root / "workspace"
        workspace.mkdir(mode=0o755)
        os.chown(workspace, 65530, 65530)
        (workspace / "initial").write_text("baseline")
        assets = root / "assets"
        assets.mkdir(mode=0o700)
        runtime = Runtime(
            OpenEnvDConfig(enabled=True),
            "echo_env.server.echo_environment:EchoEnvironment",
            "openenv.core.env_server.mcp_types:CallToolAction",
            workspace,
            uid=65530,
            gid=65530,
            asset_root=assets,
            timeout_s=30,
        )
        try:
            await runtime.start()
            pid = runtime.proc.pid
            network = runtime.network
            network_pid = network.process.pid
            assert (
                Path(f"/proc/{pid}/ns/net").readlink()
                != Path("/proc/self/ns/net").readlink()
            )
            assert "Uid:\t65530" in Path(f"/proc/{pid}/status").read_text()
            (workspace / "initial").write_text("changed")
            (workspace / "new").write_text("remove")
            await runtime.reset({"seed": 42})
            assert runtime.proc.pid != pid
            assert runtime.network.process.pid != network_pid
            assert network.process.returncode is not None
            assert (workspace / "initial").read_text() == "baseline"
            assert not (workspace / "new").exists()
            assert (workspace / "initial").stat().st_uid == 0
            assert workspace.stat().st_uid == 65530
            runtime.network.process.kill()
            for _ in range(100):
                if runtime.proc is None:
                    break
                await asyncio.sleep(0.05)
            assert runtime.proc is None, "helper failure must stop the workload"
        finally:
            await runtime.close()


@pytest.mark.asyncio
async def test_reset_restarts_monitor_after_observation_failure(tmp_path):
    from unittest.mock import Mock

    runtime = Runtime(
        OpenEnvDConfig(enabled=True),
        "unused",
        "unused",
        tmp_path,
        uid=65530,
        gid=65530,
        asset_root=tmp_path,
    )
    runtime.workspace = Mock()
    runtime._stop = AsyncMock()
    runtime._spawn = AsyncMock()
    runtime._request = AsyncMock(return_value={"done": False})
    runtime._monitor = AsyncMock()
    runtime.monitor = asyncio.create_task(asyncio.sleep(0))
    await runtime.monitor
    old_monitor = runtime.monitor
    await runtime.reset({})
    assert runtime.monitor is not old_monitor
    await runtime.monitor
    runtime._monitor.assert_awaited_once()
