# SPDX-License-Identifier: BSD-3-Clause
"""One isolated environment process and one trajectory per daemon."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import shutil
import signal
import socket
import stat
import sys
import tempfile
import time
from pathlib import Path

from .cgroup import WorkloadCgroup
from .isolation import detect_capabilities, IsolationError, spawn_task
from .models import TaskSpec
from .network import Network
from .observation import Collector, snapshot_changes, Workspace
from .policy import ObservationEventType, OpenEnvDConfig, Principal
from .telemetry import resource_sample


class Runtime:
    """Own an isolated worker, its episode workspace, and privileged assets.

    The environment factory is trusted framework code. Untrusted subprocesses
    stay inside its network namespace and receive no daemon credentials. Linux
    root and network namespace capability are required; startup fails closed.
    """

    def __init__(
        self,
        config: OpenEnvDConfig,
        factory: str,
        action_class: str,
        workspace: Path,
        *,
        uid: int,
        gid: int,
        asset_root: Path,
        timeout_s: float = 300,
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        python_path: Path | None = None,
    ):
        if not config.enabled:
            raise ValueError("openenvd is not enabled")
        if uid <= 0 or gid <= 0 or timeout_s <= 0:
            raise ValueError("positive worker identity and timeout are required")
        self.config = OpenEnvDConfig.model_validate(config.model_dump())
        self.python_path = python_path
        self.factory = factory
        self.action_class = action_class
        self.workspace_path = workspace.resolve(strict=True)
        self.asset_root = asset_root.resolve(strict=True)
        self.uid, self.gid = uid, gid
        self.timeout_s = timeout_s
        self.collector = Collector()
        self.proc = None
        self.directory = None
        self.workspace = None
        self.assets = {}
        self.lock = asyncio.Lock()
        self.monitor = None
        self.cgroup = WorkloadCgroup(cgroup_root)
        self.network = None
        self.network_monitor = None
        self.started_at = 0.0
        self.event_task = None
        self._closed = False

    async def start(self):
        if self._closed or self.directory is not None:
            raise RuntimeError("runtime is already started or closed")
        self.capabilities = detect_capabilities()
        if (
            sys.platform != "linux"
            or not self.capabilities.can_drop_uid
            or not self.capabilities.can_unshare_net
        ):
            raise IsolationError(
                "openenvd runtime requires Linux UID and network namespace isolation"
            )
        parent = self.workspace_path.parent.stat()
        if parent.st_uid != os.geteuid() or parent.st_mode & 0o022:
            raise IsolationError(
                "workspace parent must be daemon-owned and not writable by other identities"
            )
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise IsolationError("could not enable orphan process reaping")
        self.cgroup.create()
        self.directory = Path(tempfile.mkdtemp(prefix="openenvd-"))
        os.chmod(self.directory, 0o700)
        try:
            assets = self.directory / "assets"
            assets.mkdir(mode=0o700)
            for name, relative in self.config.privileged_assets.items():
                source = (self.asset_root / relative).resolve(strict=True)
                if not source.is_relative_to(self.asset_root) or source.is_relative_to(
                    self.workspace_path
                ):
                    raise ValueError(
                        "privileged asset sources must be outside the workload workspace"
                    )
                # Source copies must also be inaccessible to the workload.
                # Require a daemon-owned 0700 source root instead of leaving an
                # unprotected original behind after copying.
                info = self.asset_root.stat()
                if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                    raise ValueError("asset_root must be daemon-owned with mode 0700")
                target = assets / name
                if source.is_dir():
                    shutil.copytree(source, target, symlinks=True)
                else:
                    shutil.copy2(source, target)
                self.assets[name] = target
            self.workspace = Workspace(self.workspace_path, self.directory / "snapshot")
            self.workspace.capture()
            await self._spawn()
            self.monitor = asyncio.create_task(self._monitor())
        except BaseException:
            await self.close()
            raise

    async def _spawn(self):
        self.started_at = time.monotonic()
        observer = self.config.surfaces.get(Principal.OBSERVER)
        self.network = Network(
            self.config.network,
            self.collector,
            observe=bool(observer and "network" in observer.stream),
        )
        await self.network.start()
        await self.network.wait_ready()
        path = self.directory / "worker.sock"
        path.unlink(missing_ok=True)
        sock = socket.socket(socket.AF_UNIX)
        event_read, event_write = os.pipe()
        try:
            reader = asyncio.StreamReader(limit=65536)
            protocol = asyncio.StreamReaderProtocol(reader)
            await asyncio.get_running_loop().connect_read_pipe(
                lambda: protocol, os.fdopen(event_read, "rb")
            )
            self.event_task = asyncio.create_task(self._read_events(reader))
            sock.bind(str(path))
            sock.listen(16)
            spec = TaskSpec(
                name="workload",
                argv=[
                    sys.executable,
                    # The bootstrap seals the worker (PR_SET_DUMPABLE) before
                    # any package import; -S defers site setup until sealed.
                    "-S",
                    str(Path(__file__).with_name("_worker_bootstrap.py")),
                    str(sock.fileno()),
                    self.factory,
                    self.action_class,
                ],
                cwd=str(self.workspace_path),
                uid=self.uid,
                gid=self.gid,
                network_isolated=True,
            )
            self.proc = await spawn_task(
                spec,
                self.capabilities,
                env={
                    "PATH": os.defpath,
                    "OPENENVD_EVENT_FD": str(event_write),
                    "OPENENVD_AGENT_POLICY": self.config.surfaces.get(
                        Principal.AGENT
                    ).model_dump_json()
                    if Principal.AGENT in self.config.surfaces
                    else "",
                    **(
                        {"PYTHONPATH": str(self.python_path)}
                        if self.python_path
                        else {}
                    ),
                },
                pass_fds=(sock.fileno(), event_write),
                cgroup_path=str(self.cgroup.path),
                network_namespace=self.network.namespace,
            )
        finally:
            sock.close()
            os.close(event_write)
        self.network_monitor = asyncio.create_task(self._watch_network())
        self.collector.record(
            ObservationEventType.PROCESS, {"kind": "spawn", "pid": self.proc.pid}
        )
        # A roundtrip establishes readiness after environment imports complete.
        await self._request("ready")

    async def _watch_network(self):
        await self.network.failed.wait()
        async with self.lock:
            try:
                self.collector.record(
                    ObservationEventType.NETWORK,
                    {
                        "kind": "failure",
                        "reason": "network mediation or observation failed",
                    },
                )
            finally:
                await self._stop()

    async def _read_events(self, reader):
        try:
            while line := await reader.readline():
                try:
                    data = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(data, dict):
                    self.collector.record(
                        ObservationEventType.HARNESS_EVENT,
                        {"source": "workload", "event": data},
                    )
        except Exception:
            # An invalid or overflowing stream cannot silently disable collection.
            async with self.lock:
                await self._stop()

    async def stop(self):
        """Stop the episode while serializing against reset and oracle execution."""
        async with self.lock:
            await self._stop()

    async def _stop(self):
        if self.network_monitor and self.network_monitor is not asyncio.current_task():
            self.network_monitor.cancel()
            await asyncio.gather(self.network_monitor, return_exceptions=True)
        self.network_monitor = None
        network_error = None
        if self.network:
            try:
                await self.network.close()
                self.network = None
            except Exception as error:
                network_error = error
        if self.proc is None:
            if self.event_task:
                self.event_task.cancel()
                await asyncio.gather(self.event_task, return_exceptions=True)
                self.event_task = None
            if network_error:
                raise network_error
            return
        proc = self.proc
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), 2)
        except asyncio.TimeoutError:
            pass
        await self.cgroup.kill()
        await proc.wait()
        self.proc = None
        self._reap_orphans()
        if self.event_task and self.event_task is not asyncio.current_task():
            self.event_task.cancel()
            await asyncio.gather(self.event_task, return_exceptions=True)
            self.event_task = None
        self.collector.record(
            ObservationEventType.PROCESS,
            {"kind": "exit", "pid": proc.pid, "returncode": proc.returncode},
        )
        if network_error:
            raise network_error

    def _reap_orphans(self):
        children = Path(f"/proc/self/task/{os.getpid()}/children")
        if not children.exists():
            return
        managed = {self.proc.pid if self.proc else None}
        if self.network and self.network.process:
            managed.add(self.network.process.pid)
        for child in children.read_text().split():
            pid = int(child)
            if pid not in managed:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass

    async def _request(self, operation: str, data: dict | None = None):
        async def exchange():
            reader, writer = await asyncio.open_unix_connection(
                str(self.directory / "worker.sock"), limit=16 * 1024 * 1024
            )
            try:
                writer.write(
                    json.dumps({"operation": operation, "data": data or {}}).encode()
                    + b"\n"
                )
                await writer.drain()
                line = await reader.readline()
                if not line:
                    raise RuntimeError("environment worker disconnected")
                result = json.loads(line)
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result["result"]
            finally:
                writer.close()
                await writer.wait_closed()

        async def supervised_exchange():
            network = self.network
            if network is None:
                return await exchange()
            operation = asyncio.create_task(exchange())
            failure = asyncio.create_task(network.failed.wait())
            try:
                await asyncio.wait(
                    (operation, failure), return_when=asyncio.FIRST_COMPLETED
                )
                network.check()
                return await operation
            finally:
                operation.cancel()
                failure.cancel()
                await asyncio.gather(operation, failure, return_exceptions=True)

        return await asyncio.wait_for(supervised_exchange(), self.timeout_s)

    async def request(self, operation: str, data: dict | None = None):
        async with self.lock:
            if self._closed or self.proc is None or self.proc.returncode is not None:
                raise RuntimeError("workload exited; reset is required")
            if self.network:
                try:
                    self.network.check()
                except IsolationError:
                    await self._stop()
                    raise
            try:
                result = await self._request(operation, data)
            except (asyncio.TimeoutError, asyncio.CancelledError, IsolationError):
                await self._stop()
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

    async def reset(self, data: dict):
        async with self.lock:
            if self._closed:
                raise RuntimeError("runtime is closed")
            await self._stop()
            await asyncio.to_thread(self.workspace.restore)
            self.collector = Collector()
            try:
                await self._spawn()
                result = await self._request("reset", data)
                if self.monitor is None or self.monitor.done():
                    self.monitor = asyncio.create_task(self._monitor())
                return result
            except BaseException:
                await self._stop()
                raise

    async def _monitor(self):
        try:
            await self._observe_loop()
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self.lock:
                await self._stop()

    async def _observe_loop(self):
        observer = self.config.surfaces.get(Principal.OBSERVER)
        streams = set(observer.stream) if observer else set()
        previous = (
            await asyncio.to_thread(lambda: self.workspace.baseline)
            if "fs_diff" in streams
            else {}
        )
        previous_collector = self.collector
        processes = set()
        while True:
            await asyncio.sleep(1)
            async with self.lock:
                if self.proc and time.monotonic() - self.started_at >= self.timeout_s:
                    await self._stop()
                self._reap_orphans()
                if self.network:
                    self.network.check()
                if self.proc and self.proc.returncode is not None:
                    await self._stop()
                collector = self.collector

            # File hashing must not hold up actions or resets. A reset replaces
            # the collector, so discard any sample taken across that boundary.
            try:
                current = (
                    await asyncio.to_thread(self.workspace.scan)
                    if "fs_diff" in streams
                    else None
                )
                resources = None
                if "resource" in streams:
                    resources = {
                        **resource_sample(self.cgroup.path),
                        "disk_bytes": await asyncio.to_thread(
                            self.workspace.disk_usage
                        ),
                    }
            except Exception:
                async with self.lock:
                    if collector is not self.collector:
                        continue
                    await self._stop()
                return
            async with self.lock:
                if collector is not self.collector:
                    continue
                if collector is not previous_collector:
                    previous = self.workspace.baseline if "fs_diff" in streams else {}
                    processes = set()
                    previous_collector = collector
                if current is not None:
                    for change in snapshot_changes(previous, current):
                        collector.record(ObservationEventType.FS_CHANGE, change)
                    previous = current
                if resources is not None:
                    collector.record(ObservationEventType.RESOURCE, resources)
                if "process" in streams:
                    current_processes = set(
                        (self.cgroup.path / "cgroup.procs").read_text().split()
                    )
                    for pid in sorted(processes ^ current_processes):
                        collector.record(
                            ObservationEventType.PROCESS,
                            {
                                "pid": int(pid),
                                "kind": "spawn" if pid in current_processes else "exit",
                            },
                        )
                    processes = current_processes

    def read_file(self, path: str) -> str:
        policy = self.config.surfaces[Principal.GRADER]
        logical = Path(path)
        if (
            not logical.is_absolute()
            or ".." in logical.parts
            or not policy.permits_read(logical)
        ):
            raise PermissionError("path is not permitted")
        if logical.is_relative_to("/openenvd/assets"):
            relative = logical.relative_to("/openenvd/assets")
            root = self.directory / "assets"
        elif logical.is_relative_to(self.workspace_path):
            relative = logical.relative_to(self.workspace_path)
            root = self.workspace_path
        else:
            raise PermissionError("path is outside managed roots")
        # Walk descriptors with O_NOFOLLOW, including ancestors, to avoid symlink
        # and rename races against a concurrently running workload.
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

    async def run_oracle(self):
        async with self.lock:
            return await self._run_oracle()

    async def _run_oracle(self):
        policy = self.config.surfaces[Principal.GRADER]
        if not policy.allow_privileged_exec or "oracle" not in self.assets:
            raise PermissionError("oracle execution is not permitted")
        # Oracle is trusted operator code; inherit no daemon credentials.
        proc = await asyncio.create_subprocess_exec(
            str(self.assets["oracle"]),
            cwd=self.workspace_path,
            env={"PATH": os.defpath},
            stdin=asyncio.subprocess.DEVNULL,
            umask=0o077,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:

            async def read_bounded(stream):
                chunks = []
                size = 0
                while chunk := await stream.read(65536):
                    size += len(chunk)
                    if size > 1024 * 1024:
                        raise RuntimeError("oracle output exceeds 1 MiB")
                    chunks.append(chunk)
                return b"".join(chunks)

            stdout, stderr, _ = await asyncio.wait_for(
                asyncio.gather(
                    read_bounded(proc.stdout), read_bounded(proc.stderr), proc.wait()
                ),
                self.timeout_s,
            )
            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
            }
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()

    async def close(self):
        self._closed = True
        if self.monitor:
            self.monitor.cancel()
            await asyncio.gather(self.monitor, return_exceptions=True)
            self.monitor = None
        async with self.lock:
            await self._stop()
        if self.event_task:
            self.event_task.cancel()
            await asyncio.gather(self.event_task, return_exceptions=True)
            self.event_task = None
        if self.directory:
            shutil.rmtree(self.directory)
            self.directory = None
        self.cgroup.close()
