"""Bounded Docker-local execution for validation, independent of core providers.

Only public egress is supported in this increment. Runtime hardening is an execution
baseline; it does not certify containment or implement the later security graders.
"""

import json
import math
import os
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ..manifest import ExecutionDeclaration
from ..runtime.contracts import LaunchSpec
from ..types import ProviderCapability
from . import ExecResult, ProviderError, StartupError, UnsupportedCapability


_LABEL = "org.openenv.validation.run"
_MAX_OUTPUT = 65536
_EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    ".ssh",
    ".aws",
    ".azure",
    ".config",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "outputs",
    ".validation",
}
_SECRET_NAME = re.compile(
    r"^(?:\.env(?:\..*)?|\.netrc|.*_secrets?\..*|"
    r"id_(?:rsa|ed25519).*|id_ecdsa(?:\..*)?|"
    r"credentials(?:\.json)?|secrets?|secrets\.(?:json|toml|ya?ml))$|"
    r"\.(?:pem|key|p12|pfx)$",
    re.I | re.S,
)
_TOKEN = re.compile(
    r"(?:hf_[A-Za-z0-9]{8,}|(?:sk|ghp|github_pat)[-_][A-Za-z0-9_-]{8,}|"
    r"(?i:bearer)\s+\S+|(?i:password|token|api[_-]?key|secret)\s*[=:]\s*\S+)"
)


def _safe_text(value: str, secrets: tuple[str, ...] = ()) -> str:
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return (
        _TOKEN.sub("[REDACTED]", value)
        .encode("utf-8")[-_MAX_OUTPUT:]
        .decode("utf-8", "ignore")
    )


def _command(
    argv: list[str], timeout_s: float, max_bytes: int = _MAX_OUTPUT
) -> tuple[int, str, str]:
    """Drain both pipes while retaining bounded tails; kill the client process group."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ProviderError("Operation deadline elapsed")
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise ProviderError("Could not execute the Docker client") from exc
    buffers = [bytearray(), bytearray()]

    def drain(pipe, output):
        try:
            while chunk := pipe.read(8192):
                output.extend(chunk)
                if len(output) > max_bytes:
                    del output[:-max_bytes]
        finally:
            pipe.close()

    threads = [
        threading.Thread(target=drain, args=(pipe, output), daemon=True)
        for pipe, output in zip((process.stdout, process.stderr), buffers)
    ]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The process group can exit between wait() and killpg().
            pass
        process.wait()
        raise ProviderError("Docker operation exceeded its deadline") from None
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The process group can exit before this cleanup signal.
            pass
        process.wait()
        raise
    finally:
        for thread in threads:
            thread.join(timeout=1)
    return (
        process.returncode,
        buffers[0].decode("utf-8", "replace"),
        buffers[1].decode("utf-8", "replace"),
    )


def _contained(root: Path, relative: str) -> Path:
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ProviderError("Build paths must remain inside the package") from None
    if path.is_symlink():
        raise ProviderError("Build paths may not be symbolic links")
    return path


def _snapshot(root: Path, destination: Path, max_bytes: int, deadline: float) -> None:
    """Copy source bytes, excluding links, caches, known credential names and outputs."""
    total = 0
    files = 0

    def copy_tree(directory_fd: int, target_dir: Path, depth: int = 0):
        nonlocal total, files
        if depth > 64:
            raise ProviderError("Build snapshot exceeds its directory depth budget")
        # Descriptor-relative opens also reject a link swapped in during copying.
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                name = entry.name
                if name in _EXCLUDED or _SECRET_NAME.search(name):
                    continue
                if time.monotonic() >= deadline:
                    raise ProviderError("Build snapshot exceeded its deadline")
                if entry.is_symlink():
                    raise ProviderError("Build snapshots do not accept symbolic links")
                files += 1
                if files > 100000:
                    raise ProviderError("Build snapshot exceeds its file-count budget")
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                child_fd = os.open(name, flags, dir_fd=directory_fd)
                try:
                    info = os.fstat(child_fd)
                    target = target_dir / name
                    if stat.S_ISDIR(info.st_mode):
                        target.mkdir()
                        copy_tree(child_fd, target, depth + 1)
                    elif stat.S_ISREG(info.st_mode):
                        with target.open("wb") as output:
                            while chunk := os.read(child_fd, 1024 * 1024):
                                total += len(chunk)
                                if total > max_bytes:
                                    raise ProviderError(
                                        "Build snapshot exceeds its size budget"
                                    )
                                if time.monotonic() >= deadline:
                                    raise ProviderError(
                                        "Build snapshot exceeded its deadline"
                                    )
                                output.write(chunk)
                        target.chmod(info.st_mode & 0o777)
                    else:
                        raise ProviderError("Build snapshots accept regular files only")
                finally:
                    os.close(child_fd)

    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            copy_tree(root_fd, destination)
        finally:
            os.close(root_fd)
    except OSError:
        raise ProviderError(
            "Build snapshot could not safely read package files"
        ) from None


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DockerValidationProvider:
    """Build isolated source snapshots and launch owned, resource-bounded subjects."""

    name = "docker-local"
    capabilities = frozenset({ProviderCapability.IMAGE_BUILD, ProviderCapability.EXEC})
    supported_network_modes = frozenset({"public"})

    def __init__(self, *, build_timeout_s: float = 600, max_context_bytes: int = 2**30):
        if not math.isfinite(build_timeout_s) or build_timeout_s <= 0:
            raise ValueError("build_timeout_s must be positive and finite")
        if max_context_bytes <= 0:
            raise ValueError("max_context_bytes must be positive")
        self.build_timeout_s = build_timeout_s
        self.max_context_bytes = max_context_bytes

    def build(self, root: Path, execution: ExecutionDeclaration) -> str:
        deadline = time.monotonic() + self.build_timeout_s
        try:
            root = root.resolve(strict=True)
        except OSError:
            raise ProviderError("Package directory could not be read") from None
        _contained(root, execution.dockerfile)
        _contained(root, execution.context)
        with tempfile.TemporaryDirectory(
            prefix="openenv-validation-build-"
        ) as directory:
            staging = Path(directory) / "source"
            staging.mkdir()
            _snapshot(root, staging, self.max_context_bytes, deadline)
            dockerfile = _contained(staging, execution.dockerfile)
            context = _contained(staging, execution.context)
            if not dockerfile.is_file() or not context.is_dir():
                raise ProviderError(
                    "Build Dockerfile and context must exist in the snapshot"
                )
            iidfile = Path(directory) / "image-id"
            code, _, stderr = _command(
                [
                    "docker",
                    "build",
                    "--iidfile",
                    str(iidfile),
                    "--file",
                    str(dockerfile),
                    str(context),
                ],
                deadline - time.monotonic(),
            )
            if code != 0:
                raise StartupError("Image build failed: " + _safe_text(stderr[-4096:]))
            image = iidfile.read_text().strip() if iidfile.exists() else ""
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
                raise ProviderError("Docker did not return an immutable image ID")
            return image

    def start(self, spec: LaunchSpec) -> "DockerRunningSubject":
        if spec.network.mode not in self.supported_network_modes:
            raise UnsupportedCapability(
                f"docker-local does not enforce {spec.network.mode}"
            )
        if spec.resources.gpus or spec.resources.gpu_types:
            raise UnsupportedCapability("docker-local does not provide GPU isolation")
        if not math.isfinite(spec.resources.cpu):
            raise UnsupportedCapability("CPU budget must be finite")
        name = f"openenv-validation-{spec.run_id}-{uuid.uuid4().hex[:12]}"
        subject = DockerRunningSubject(name, spec)
        # tmpfs and /dev/shm share the declared aggregate writable-byte allowance.
        budget = spec.resources.disk_mb * 1024 * 1024
        shm_bytes = min(64 * 1024 * 1024, budget // 2)
        argv = [
            "docker",
            "create",
            "--name",
            name,
            "--label",
            f"{_LABEL}={spec.run_id}",
            "--init",
            "--user",
            "65532:65532",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--memory",
            f"{spec.resources.memory_mb}m",
            "--memory-swap",
            f"{spec.resources.memory_mb}m",
            "--cpus",
            str(spec.resources.cpu),
            "--network",
            "bridge",
            "--publish",
            "127.0.0.1::8000",
            "--shm-size",
            str(shm_bytes),
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={budget - shm_bytes},mode=1777",
            "--log-driver",
            "json-file",
            "--log-opt",
            "max-size=1m",
            "--log-opt",
            "max-file=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
        ]
        for key, value in sorted(spec.env_vars.items()):
            argv.extend(["--env", f"{key}={value}"])
        argv.append(spec.image_ref)
        deadline = time.monotonic() + spec.startup_timeout_s
        try:
            subject._run(argv, deadline - time.monotonic())
            details = subject._raw_inspect(deadline - time.monotonic())
            mounts = details.get("Mounts", [])
            if any(mount.get("Type") in {"bind", "volume"} for mount in mounts):
                raise StartupError("Image declares unsupported persistent mounts")
            subject._run(["docker", "start", name], deadline - time.monotonic())
            details = subject._raw_inspect(deadline - time.monotonic())
            ports = details.get("NetworkSettings", {}).get("Ports", {}).get("8000/tcp")
            if not ports or ports[0].get("HostIp") != "127.0.0.1":
                raise StartupError("Docker did not bind the API to host loopback")
            port = int(ports[0]["HostPort"])
            if not 0 < port < 65536:
                raise StartupError("Docker returned an invalid API port")
            subject.base_url = f"http://127.0.0.1:{port}"
            # Never send local validation requests through a host HTTP proxy.
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), _NoRedirects()
            )
            while time.monotonic() < deadline:
                try:
                    with opener.open(
                        subject.base_url + "/health",
                        timeout=min(1.0, deadline - time.monotonic()),
                    ) as response:
                        if response.status == 200:
                            return subject
                except (OSError, urllib.error.URLError):
                    # Connection failures are expected while the server starts.
                    pass
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            raise StartupError(
                "Subject did not return HTTP 200 from /health before its deadline"
            )
        except BaseException as exc:
            try:
                subject.stop()
            except ProviderError as cleanup:
                raise StartupError(
                    f"Startup failed; cleanup also failed: {cleanup}"
                ) from None
            if isinstance(exc, (KeyboardInterrupt, SystemExit, StartupError)):
                raise
            raise StartupError(_safe_text(str(exc), subject._secrets)) from None


class DockerRunningSubject:
    """An owned container; only this container is removed during cleanup."""

    def __init__(self, name: str, spec: LaunchSpec):
        self.name = name
        self.run_id = spec.run_id
        self.base_url = ""
        self._stopped = False
        self._secrets = tuple(spec.env_vars.values())

    def _run(self, argv: list[str], timeout_s: float, max_bytes: int = _MAX_OUTPUT):
        code, stdout, stderr = _command(argv, timeout_s, max_bytes)
        if code:
            raise ProviderError(
                "Docker operation failed: " + _safe_text(stderr[-4096:], self._secrets)
            )
        return stdout

    def _raw_inspect(self, timeout_s: float = 10) -> dict:
        output = self._run(["docker", "inspect", self.name], timeout_s)
        try:
            details = json.loads(output)[0]
            if details["Config"].get("Labels", {}).get(_LABEL) != self.run_id:
                raise ProviderError("Container ownership label does not match")
            return details
        except (ValueError, IndexError, KeyError, TypeError, AttributeError):
            raise ProviderError("Docker returned invalid inspection evidence") from None

    def inspect(self) -> dict:
        """Return selected effective settings without image env, command or host paths."""
        details = self._raw_inspect()
        host = details.get("HostConfig", {})
        return {
            "container_id": details.get("Id"),
            "image_id": details.get("Image"),
            "running": details.get("State", {}).get("Running", False),
            "oom_killed": details.get("State", {}).get("OOMKilled", False),
            "user": details.get("Config", {}).get("User"),
            "limits": {
                key: host.get(key)
                for key in (
                    "Memory",
                    "MemorySwap",
                    "NanoCpus",
                    "PidsLimit",
                    "ReadonlyRootfs",
                    "CapDrop",
                    "SecurityOpt",
                    "Init",
                    "Tmpfs",
                    "ShmSize",
                    "NetworkMode",
                )
            },
            "mounts": [
                {key: mount.get(key) for key in ("Type", "Destination", "RW")}
                for mount in details.get("Mounts", [])
            ],
        }

    def logs(self, max_bytes: int = _MAX_OUTPUT) -> str:
        if not 0 < max_bytes <= _MAX_OUTPUT:
            raise ValueError("max_bytes must be between 1 and 65536")
        code, stdout, stderr = _command(
            ["docker", "logs", "--tail", "1000", self.name], 10, max_bytes
        )
        if code:
            raise ProviderError("Could not read subject logs")
        return (
            _safe_text(stdout + stderr, self._secrets)
            .encode("utf-8")[-max_bytes:]
            .decode("utf-8", "ignore")
        )

    def exec(self, argv: list[str], timeout_s: float) -> ExecResult:
        if (
            not argv
            or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
            or not argv[0]
        ):
            raise ValueError(
                "exec requires a nonempty argument vector without null bytes"
            )
        if self._stopped:
            raise ProviderError("Subject has stopped")
        started = time.monotonic()
        try:
            code, stdout, stderr = _command(
                ["docker", "exec", "--", self.name, *argv], timeout_s
            )
        except ProviderError as exc:
            # Killing a docker exec client alone does not kill in-container children.
            try:
                self.stop()
            except ProviderError as cleanup:
                raise cleanup from exc
            raise
        return ExecResult(
            code,
            _safe_text(stdout, self._secrets),
            _safe_text(stderr, self._secrets),
            time.monotonic() - started,
        )

    def stop(self) -> None:
        if self._stopped:
            return
        # Ask Docker for only the owner label: arbitrary image ENV/labels can
        # exceed the bounded full-inspection output and must not block cleanup.
        code, output, stderr = _command(
            [
                "docker",
                "inspect",
                "--format",
                f'{{{{index .Config.Labels "{_LABEL}"}}}}',
                self.name,
            ],
            10,
        )
        if code:
            if "No such object" in stderr or "No such container" in stderr:
                self._stopped = True
                return
            raise ProviderError("Could not verify container ownership for cleanup")
        if output.strip() != self.run_id:
            raise ProviderError("Refusing cleanup of a container with another owner")
        code, _, stderr = _command(
            ["docker", "rm", "--force", "--volumes", self.name], 10
        )
        if code and "No such container" not in stderr:
            raise ProviderError(
                "Could not remove validation container: "
                + _safe_text(stderr[-4096:], self._secrets)
            )
        code, _, stderr = _command(
            ["docker", "inspect", "--format", "{{.Id}}", self.name], 10
        )
        if not code or not (
            "No such object" in stderr or "No such container" in stderr
        ):
            raise ProviderError("Container removal could not be independently verified")
        self._stopped = True
