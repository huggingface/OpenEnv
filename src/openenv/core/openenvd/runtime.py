# SPDX-License-Identifier: BSD-3-Clause
"""Episode coordination around an OpenShell-owned sandbox."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import stat
import tempfile
import time
import uuid
from pathlib import Path

from .isolation import IsolationError
from .observation import Collector, snapshot_changes, Workspace
from .openshell import OpenShellSandbox
from .policy import ObservationEventType, OpenEnvDConfig, Principal


class Runtime:
    """Keep principal surfaces and assets outside an OpenShell workload sandbox.

    The local workspace is a seed, never a host mount. Each episode receives a
    new sandbox and a copy of that seed. Grading operates on downloaded snapshots;
    oracles run in a separate OpenShell sandbox with private assets. Their
    changes are not copied back into the running workload.
    """

    def __init__(
        self,
        config: OpenEnvDConfig,
        factory: str,
        action_class: str,
        workspace: Path,
        *,
        asset_root: Path,
        timeout_s: float = 300,
    ):
        if not config.enabled:
            raise ValueError("openenvd is not enabled")
        if config.openshell is None:
            raise ValueError("openenvd requires an openshell image and gateway")
        if timeout_s <= 0 or not float(timeout_s) < float("inf"):
            raise ValueError("a finite positive timeout is required")
        self.config = OpenEnvDConfig.model_validate(config.model_dump())
        observer = self.config.surfaces.get(Principal.OBSERVER)
        if observer and set(observer.stream) & {"network", "resource"}:
            raise ValueError(
                "OpenShell network/resource telemetry is not yet available through "
                "the observer surface; remove these streams"
            )
        self.factory, self.action_class = factory, action_class
        self.workspace_path = workspace.resolve(strict=True)
        self.asset_root = asset_root.resolve(strict=True)
        self.timeout_s = timeout_s
        self.collector = Collector()
        self.backend = OpenShellSandbox(self.config.openshell, timeout_s=timeout_s)
        self.proc = None
        self.directory = None
        self.workspace = None
        self.assets = {}
        self.grader_backend = None
        self.grader_directory = None
        self._snapshot_valid = False
        self.lock = asyncio.Lock()
        self.monitor = None
        self.event_task = None
        self.stderr_task = None
        self.pending = None
        self.teardown = None
        self.started_at = 0.0
        self._closed = False

    async def start(self):
        if self._closed or self.directory is not None:
            raise RuntimeError("runtime is already started or closed")
        self._validate_seed()
        self.directory = Path(tempfile.mkdtemp(prefix="openenvd-"))
        self.directory.chmod(0o700)
        try:
            assets = self.directory / "assets"
            assets.mkdir(mode=0o700)
            for name, relative in self.config.privileged_assets.items():
                source = (self.asset_root / relative).resolve(strict=True)
                if not source.is_relative_to(self.asset_root):
                    raise ValueError("privileged asset escapes asset_root")
                if source.is_dir():
                    shutil.copytree(source, assets / name, symlinks=True)
                else:
                    shutil.copy2(source, assets / name)
                self.assets[name] = assets / name
            seed_parent = self.directory / "seed"
            seed_parent.mkdir(mode=0o700)
            self.workspace = Workspace(self.workspace_path, seed_parent / "workspace")
            self.workspace.capture()
            # The seed and asset copies stay daemon-owned. Only the seed is uploaded.
            await self._spawn()
            self.monitor = asyncio.create_task(self._monitor())
        except BaseException:
            await self.close()
            raise

    def _validate_seed(self):
        if not self.workspace_path.is_dir() or self.workspace_path == Path("/"):
            raise ValueError("workspace must be a dedicated directory")
        if self.config.privileged_assets and (
            self.asset_root.is_relative_to(self.workspace_path)
            or self.workspace_path.is_relative_to(self.asset_root)
        ):
            raise ValueError("privileged assets and workspace must have separate roots")
        for root, dirs, files in os.walk(self.workspace_path, followlinks=False):
            for name in dirs + files:
                mode = (Path(root) / name).lstat().st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError(
                        "workspace seed may contain only regular files and directories"
                    )

    async def _spawn(self):
        await self.backend.start(self.workspace.snapshot, self.directory)
        bootstrap = Path(__file__).with_name("_worker_bootstrap.py").read_text()
        self.proc = await self.backend.spawn(
            [self.config.openshell.python, "-I", "-S", "-c", bootstrap],
            self._workload_env(),
        )
        self.started_at = time.monotonic()
        self.event_task = asyncio.create_task(self._read_frames())
        self.stderr_task = asyncio.create_task(self._drain_stderr())
        policy = self.config.surfaces.get(Principal.AGENT)
        await self._exchange(
            {
                "factory": self.factory,
                "action_class": self.action_class,
                "agent_policy": policy.model_dump(mode="json") if policy else None,
            }
        )
        self.collector.record(
            ObservationEventType.PROCESS,
            {"kind": "spawn", "sandbox_id": self.backend.id},
        )

    def _workload_env(self):
        return {
            "PATH": str(Path(self.config.openshell.python).parent) + ":/usr/bin:/bin",
            "HOME": "/sandbox",
            "TMPDIR": "/tmp",
        }

    async def _drain_stderr(self):
        # Environment logs may contain user data. Drain without retaining or
        # copying them into API errors, and without allowing a full pipe to hang.
        while await self.proc.stderr.read(65536):
            pass

    async def _read_frames(self):
        try:
            while line := await self.proc.stdout.readline():
                frame = json.loads(line)
                if not isinstance(frame, dict):
                    raise IsolationError("invalid worker response")
                if set(frame) == {"event"} and isinstance(frame["event"], dict):
                    self.collector.record(
                        ObservationEventType.HARNESS_EVENT,
                        {"source": "workload", "event": frame["event"]},
                    )
                elif set(frame) in ({"result"}, {"error"}) and self.pending is not None:
                    if self.pending.done():
                        raise IsolationError("unexpected worker response")
                    self.pending.set_result(frame)
                else:
                    raise IsolationError("unexpected worker response")
            raise IsolationError("OpenShell worker disconnected")
        except asyncio.CancelledError:
            raise
        except Exception:
            if self.pending is not None and not self.pending.done():
                self.pending.set_exception(
                    IsolationError("OpenShell worker disconnected or sent invalid data")
                )
            # The monitor notices completion and tears down the complete sandbox.

    async def _exchange(self, payload):
        future = asyncio.get_running_loop().create_future()
        self.pending = future
        try:
            if self.event_task.done():
                raise IsolationError("OpenShell worker disconnected")
            self.proc.stdin.write(json.dumps(payload).encode() + b"\n")
            await asyncio.wait_for(self.proc.stdin.drain(), self.timeout_s)
            response = await asyncio.wait_for(future, self.timeout_s)
            if "error" in response:
                raise RuntimeError("environment operation failed")
            return response["result"]
        finally:
            self.pending = None
            if not future.done():
                future.cancel()

    async def _request(self, operation: str, data: dict | None = None):
        return await self._exchange({"operation": operation, "data": data or {}})

    async def request(self, operation: str, data: dict | None = None):
        async with self.lock:
            if self._closed or self.proc is None or self.proc.returncode is not None:
                raise RuntimeError("workload exited; reset is required")
            try:
                result = await self._request(operation, data)
            except (
                asyncio.TimeoutError,
                asyncio.CancelledError,
                IsolationError,
                OSError,
            ):
                await self._stop(capture=False)
                raise
            if operation == "step" or (
                operation == "mcp" and data and data.get("method") == "tools/call"
            ):
                self.collector.record(
                    ObservationEventType.HARNESS_EVENT,
                    {
                        "source": "daemon",
                        "operation": operation,
                        "request": data,
                        "response": result,
                    },
                )
            return result

    async def _sync_workspace(self):
        self._snapshot_valid = False
        staging = Path(tempfile.mkdtemp(prefix="view-", dir=self.directory))
        try:
            await self.backend.download(staging)
        except BaseException:
            self._remove_private_tree(staging)
            raise
        previous = self.workspace.path
        self.workspace.path = staging
        self._snapshot_valid = True
        if previous != self.workspace_path and previous.is_relative_to(self.directory):
            self._remove_private_tree(previous)

    @staticmethod
    def _remove_private_tree(path):
        # Snapshots preserve workload modes, including read-only directories.
        # Repair only the private copy, never a symlink target or the seed.
        if path.is_symlink():
            path.unlink()
            return
        path.chmod(0o700)
        for root, directories, _ in os.walk(path, followlinks=False):
            for name in directories:
                directory = Path(root) / name
                if not directory.is_symlink():
                    directory.chmod(0o700)
        shutil.rmtree(path)

    async def stop(self):
        async with self.lock:
            await self._stop()

    async def _stop(self, *, capture=True):
        # Keep cleanup owned even if the request/connection that initiated it is
        # cancelled. Subsequent reset/close calls await the same teardown first.
        if self.teardown is None or self.teardown.done():
            self.teardown = asyncio.create_task(self._teardown(capture=capture))
        try:
            await asyncio.shield(self.teardown)
        except asyncio.CancelledError:
            await asyncio.shield(self.teardown)
            raise

    async def _teardown(self, *, capture):
        capture_error = None
        if not capture:
            self._snapshot_valid = False
        try:
            await self._close_grader()
        except Exception as error:
            capture_error = error
        if capture and self.backend.id and self.workspace is not None:
            try:
                await self._sync_workspace()
            except Exception as error:
                capture_error = error
        # Only confirmed sandbox deletion permits another episode. Killing the
        # local SSH client by itself does not prove remote descendant cleanup.
        await self.backend.close()
        proc = self.proc
        if proc is not None:
            if proc.returncode is None:
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            self.proc = None
            self.collector.record(
                ObservationEventType.PROCESS,
                {"kind": "exit", "returncode": proc.returncode},
            )
        for task in (self.event_task, self.stderr_task):
            if task and task is not asyncio.current_task():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.event_task = self.stderr_task = None
        if capture_error:
            raise IsolationError(
                "workspace snapshot failed before sandbox teardown"
            ) from capture_error

    async def reset(self, data: dict):
        async with self.lock:
            if self._closed:
                raise RuntimeError("runtime is closed")
            await self._stop(capture=False)
            self.collector = Collector()
            self.backend = OpenShellSandbox(
                self.config.openshell, timeout_s=self.timeout_s
            )
            try:
                await self._spawn()
                result = await self._request("reset", data)
                if self.monitor is None or self.monitor.done():
                    self.monitor = asyncio.create_task(self._monitor())
                return result
            except BaseException:
                await self._stop(capture=False)
                raise

    async def _monitor(self):
        try:
            await self._observe_loop()
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self.lock:
                await self._stop(capture=False)

    async def _observe_loop(self):
        observer = self.config.surfaces.get(Principal.OBSERVER)
        watch_files = bool(observer and "fs_diff" in observer.stream)
        previous = self.workspace.baseline if watch_files else {}
        previous_collector = self.collector
        while True:
            await asyncio.sleep(1)
            async with self.lock:
                if self.proc is None:
                    continue
                if (
                    self.proc.returncode is not None
                    or self.event_task.done()
                    or time.monotonic() - self.started_at >= self.timeout_s
                ):
                    await self._stop(capture=False)
                    return
                if not watch_files:
                    continue
                if self.collector is not previous_collector:
                    previous = self.workspace.baseline
                    previous_collector = self.collector
                await self._sync_workspace()
                current = await asyncio.to_thread(self.workspace.scan)
                for change in snapshot_changes(previous, current):
                    self.collector.record(ObservationEventType.FS_CHANGE, change)
                previous = current

    async def _refresh_snapshot(self):
        if self.backend.id:
            await self._sync_workspace()
        elif not self._snapshot_valid:
            raise RuntimeError(
                "workspace snapshot unavailable after forced teardown; reset required"
            )

    async def fs_diff(self):
        async with self.lock:
            await self._refresh_snapshot()
            return [
                {**change, "path": str(Path("/workspace") / change["path"])}
                for change in snapshot_changes(
                    self.workspace.baseline, self.workspace.scan()
                )
            ]

    async def read_file(self, path: str) -> str:
        policy = self.config.surfaces[Principal.GRADER]
        logical = Path(path)
        if (
            not logical.is_absolute()
            or ".." in logical.parts
            or not policy.permits_read(logical)
        ):
            raise PermissionError("path is not permitted")
        async with self.lock:
            if logical.is_relative_to("/openenvd/assets"):
                relative = logical.relative_to("/openenvd/assets")
                root = self.directory / "assets"
            elif logical.is_relative_to("/workspace"):
                await self._refresh_snapshot()
                relative = logical.relative_to("/workspace")
                root = self.workspace.path
            else:
                raise PermissionError("path is outside managed roots")
            return await asyncio.to_thread(self._read_regular_file, root, relative)

    @staticmethod
    def _read_regular_file(root, relative):
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for index, part in enumerate(relative.parts):
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if index < len(relative.parts) - 1:
                    flags |= os.O_DIRECTORY
                next_fd = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                raise PermissionError("only regular files up to 1 MiB may be read")
            content = os.read(fd, 1024 * 1024 + 1)
            if len(content) > 1024 * 1024:
                raise PermissionError("file grew beyond 1 MiB")
            return content.decode("utf-8")
        finally:
            os.close(fd)

    async def _close_grader(self):
        if self.grader_backend is None:
            return
        await self.grader_backend.close()
        self.grader_backend = None
        self._remove_private_tree(self.grader_directory)
        self.grader_directory = None

    async def run_oracle(self):
        async with self.lock:
            policy = self.config.surfaces[Principal.GRADER]
            if not policy.allow_privileged_exec or "oracle" not in self.assets:
                raise PermissionError("oracle execution is not permitted")
            await self._close_grader()
            await self._refresh_snapshot()
            directory = Path(tempfile.mkdtemp(prefix="grader-", dir=self.directory))
            self.grader_directory = directory
            self.grader_backend = OpenShellSandbox(
                self.config.openshell, timeout_s=self.timeout_s
            )
            try:
                seed = directory / "workspace"
                shutil.copytree(self.workspace.path, seed)
                seed.chmod(stat.S_IMODE(seed.stat().st_mode) | 0o700)
                asset_name = ".openenvd-assets-" + uuid.uuid4().hex
                # Assets enter only this short-lived grading sandbox. No daemon
                # credentials or grader results enter the agent's sandbox.
                for name, source in self.assets.items():
                    target = seed / asset_name / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.is_dir():
                        shutil.copytree(source, target, symlinks=True)
                    else:
                        shutil.copy2(source, target)
                await self.grader_backend.start(seed, directory)
                proc = await self.grader_backend.spawn(
                    [f"/sandbox/workspace/{asset_name}/oracle"], self._workload_env()
                )
                return await self._oracle_result(proc)
            finally:
                cleanup = asyncio.create_task(self._close_grader())
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await asyncio.shield(cleanup)
                    raise

    async def _oracle_result(self, proc):
        async def read_bounded(stream):
            chunks, size = [], 0
            while chunk := await stream.read(65536):
                size += len(chunk)
                if size > 1024 * 1024:
                    raise RuntimeError("oracle output exceeds 1 MiB")
                chunks.append(chunk)
            return b"".join(chunks)

        # The oracle receives no interactive control channel.
        proc.stdin.close()
        readers = [
            asyncio.create_task(read_bounded(proc.stdout)),
            asyncio.create_task(read_bounded(proc.stderr)),
        ]
        try:
            stdout, stderr, _ = await asyncio.wait_for(
                asyncio.gather(*readers, proc.wait()), self.timeout_s
            )
            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
            }
        finally:
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            async def discard(stream):
                while await stream.read(65536):
                    pass

            await asyncio.wait_for(
                asyncio.gather(discard(proc.stdout), discard(proc.stderr), proc.wait()),
                5,
            )

    async def close(self):
        self._closed = True
        if self.monitor:
            self.monitor.cancel()
            await asyncio.gather(self.monitor, return_exceptions=True)
            self.monitor = None
        async with self.lock:
            await self._stop(capture=False)
        if self.directory:
            self._remove_private_tree(self.directory)
            self.directory = None
