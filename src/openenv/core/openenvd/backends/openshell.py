# SPDX-License-Identifier: BSD-3-Clause
"""The OpenShell enforcement backend.

Each sandbox is one NVIDIA OpenShell sandbox created from the environment image.
The openenvd contract becomes an OpenShell policy (Landlock filesystem rules, a
non-root identity, default-deny egress). After creation the backend checks that
the gateway admitted exactly that policy, then reaches the workload only over
the sandbox's authenticated SSH transport. Gateway credentials stay on the
daemon and are never forwarded into the sandbox.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import tarfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from packaging.version import InvalidVersion, Version

from ..isolation import IsolationError
from ..policy import Guarantee, OpenEnvDConfig, OpenShellConfig, SANDBOX_WORKSPACE
from .base import EnforcementBackend, EnforcementUnavailable

WORKSPACE = SANDBOX_WORKSPACE
SUPPORTED_VERSIONS = ">=0.1.2,<0.2"
_OUTPUT_LIMIT = 1024 * 1024
_WORKSPACE_ARCHIVE_LIMIT = 64 * 1024 * 1024
_WORKSPACE_MEMBER_LIMIT = 4096
_TERMINATION_TIMEOUT = 5

# Paths OpenShell adds to any policy with network rules
# (docs/how-it-works/policies/default-policy.mdx). Admission accepts them only
# in that case; privileged assets never enter the sandbox, so they expose
# nothing openenvd protects.
BASELINE_READ_ONLY = (
    "/usr",
    "/lib",
    "/etc",
    "/app",
    "/var/log",
    "/proc",
    "/dev/urandom",
)
BASELINE_READ_WRITE = ("/tmp", "/dev/null")


def render_policy(config: OpenEnvDConfig) -> dict[str, Any]:
    """
    The OpenShell policy that enforces `config`.

    A native `openshell.policy` is returned as declared (it was validated against
    openenvd's invariants). Otherwise the policy is rendered from the
    backend-neutral `workload` and `egress` declarations.

    Args:
        config ([`~openenv.core.openenvd.policy.OpenEnvDConfig`]):
            A validated config selecting the `openshell` backend.

    Returns:
        `dict`: an OpenShell policy document (`version: 1`).
    """
    settings = config.openshell
    if settings is None:
        raise ValueError("the openshell backend requires an openshell section")
    if settings.policy is not None:
        return json.loads(json.dumps(settings.policy))
    network = {}
    for index, rule in enumerate(config.egress.allow):
        network[f"egress_{index}"] = {
            "name": f"openenvd-egress-{index}",
            "endpoints": [{"host": rule.host, "port": rule.port, "protocol": "tcp"}],
            "binaries": [{"path": path} for path in rule.binaries or ("/**",)],
        }
    return {
        "version": 1,
        "filesystem_policy": {
            "include_workdir": False,
            "read_only": list(config.workload.read_only),
            "read_write": list(config.workload.read_write),
        },
        "landlock": {"compatibility": "hard_requirement"},
        "process": {
            "run_as_user": settings.run_as_user,
            "run_as_group": settings.run_as_group,
        },
        "network_policies": network,
        "network_middlewares": {},
    }


class OpenShellBackend(EnforcementBackend):
    """
    Enforces the openenvd contract with NVIDIA OpenShell sandboxes.

    Provides every [`~openenv.core.openenvd.policy.Guarantee`]:

    - `asset_isolation`: assets stay with the daemon and the sandbox's
      Landlock policy confines it to its own filesystem.
    - `egress_control`: the sandbox's network namespace only reaches declared
      destinations through OpenShell's policy proxy.
    - `privilege_drop`: the workload runs as a positive numeric uid/gid under
      OpenShell's seccomp filter.
    - `control_plane_isolation`: openenvd's surfaces run in the daemon,
      and OpenShell never authorizes loopback or link-local destinations. Do not
      add the daemon's own address to the egress allowlist.

    Host prerequisites, checked by `probe()`: the `openshell` CLI (stable
    `>=0.1.2,<0.2`), OpenSSH, and a registered gateway that answers a sandbox
    inventory query. The gateway must support Landlock ABI 3 or newer.

    Args:
        config ([`~openenv.core.openenvd.policy.OpenEnvDConfig`]):
            A validated config with an `openshell` section.
    """

    name = "openshell"
    guarantees = frozenset(Guarantee)

    def __init__(self, config: OpenEnvDConfig):
        super().__init__(config)
        self.policy = render_policy(config)
        self.settings: OpenShellConfig = config.openshell

    @property
    def python(self) -> str:
        return self.settings.python

    async def probe(self) -> None:
        cli, ssh = shutil.which("openshell"), shutil.which("ssh")
        if not cli or not ssh:
            raise EnforcementUnavailable(
                self.name, "the openshell CLI and OpenSSH must be installed"
            )
        probe = OpenShellSandbox(self.settings, self.policy, timeout_s=30)
        probe._cli = cli
        try:
            version_text = await probe._run([cli, "--version"])
            version = Version(version_text.strip().split()[-1])
        except (IsolationError, IndexError, InvalidVersion):
            raise EnforcementUnavailable(
                self.name, "could not determine the OpenShell version"
            ) from None
        if (
            version.is_prerelease
            or version.is_devrelease
            or not Version("0.1.2") <= version < Version("0.2")
        ):
            raise EnforcementUnavailable(
                self.name, f"OpenShell {SUPPORTED_VERSIONS} stable is required"
            )
        # `openshell status` exits 0 even for an unregistered gateway, so ask the
        # gateway for real work: a label-scoped, read-only inventory query.
        try:
            inventory = await probe._json(
                "sandbox",
                "list",
                "--selector",
                "openenv-probe=1",
                "--output",
                "json",
                "--page-token",
                "",
            )
        except IsolationError:
            inventory = None
        if not isinstance(inventory, dict) or not isinstance(
            inventory.get("sandboxes"), list
        ):
            raise EnforcementUnavailable(
                self.name,
                f"OpenShell gateway {self.settings.gateway!r} is not registered or "
                "not reachable; check `openshell gateway list`",
            )

    def sandbox(self, timeout_s: float) -> "OpenShellSandbox":
        return OpenShellSandbox(self.settings, self.policy, timeout_s=timeout_s)


class OpenShellSandbox:
    """Own one gateway sandbox and keep its credentials outside the workload."""

    def __init__(
        self,
        settings: OpenShellConfig,
        policy: dict[str, Any],
        *,
        timeout_s: float = 300,
    ):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("OpenShell timeout must be finite and positive")
        self.settings = settings
        self.policy = policy
        self.timeout_s = timeout_s
        self.name: str | None = None
        self.id: str | None = None
        self._label = uuid.uuid4().hex
        self._cli = None
        self._ssh = None
        self._ssh_config = None
        self._processes: list[asyncio.subprocess.Process] = []
        self._ready = False
        self._closed = False

    def _host_env(self):
        # The CLI reads gateway credentials from its own configuration. Neither
        # daemon credentials nor ambient OpenShell policy overrides are inherited.
        keys = (
            "PATH",
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME",
            "XDG_STATE_HOME",
            "TMPDIR",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
        )
        env = {key: os.environ[key] for key in keys if key in os.environ}
        timeout = str(max(1, int(self.timeout_s)))
        env.update(
            NO_COLOR="1",
            OPENSHELL_PROVISION_TIMEOUT=timeout,
            OPENSHELL_LIFECYCLE_TIMEOUT=timeout,
        )
        return env

    def _command(self, *args):
        return [
            self._cli,
            "--gateway",
            self.settings.gateway,
            "--workspace",
            self.settings.workspace,
            "--color",
            "never",
            *args,
        ]

    @staticmethod
    async def _terminate(process, *, drain=False):
        # The CLI may have exited while an SSH proxy still owns its pipes.
        # Every transport has its own session, so always kill that whole group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        async def discard(reader):
            while reader is not None and await reader.read(65536):
                pass

        pending = (
            asyncio.gather(
                discard(process.stdout), discard(process.stderr), process.wait()
            )
            if drain
            else process.wait()
        )
        try:
            await asyncio.wait_for(pending, _TERMINATION_TIMEOUT)
        except asyncio.TimeoutError:
            raise IsolationError(
                "OpenShell transport termination was not confirmed"
            ) from None

    async def _create_process(self, argv, **kwargs):
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *argv, env=self._host_env(), start_new_session=True, **kwargs
            )
        )
        try:
            return await asyncio.shield(creation)
        except asyncio.CancelledError:
            process = await creation
            await self._terminate(process, drain=True)
            raise
        except OSError:
            raise IsolationError("could not launch the OpenShell transport") from None

    async def _run(self, argv, *, timeout=None):
        output = await self._run_bytes(argv, timeout=timeout)
        try:
            return output.decode("utf-8", errors="strict")
        except UnicodeError:
            raise IsolationError("OpenShell returned invalid command output") from None

    async def _run_bytes(self, argv, *, timeout=None, output_limit=_OUTPUT_LIMIT):
        process = await self._create_process(
            argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def read_bounded(reader, limit):
            output = bytearray()
            while chunk := await reader.read(65536):
                output.extend(chunk)
                if len(output) > limit:
                    raise IsolationError("OpenShell command exceeded its output limit")
            return bytes(output)

        readers = [
            asyncio.create_task(read_bounded(process.stdout, output_limit)),
            asyncio.create_task(read_bounded(process.stderr, _OUTPUT_LIMIT)),
        ]
        try:
            stdout, _, _ = await asyncio.wait_for(
                asyncio.gather(*readers, process.wait()), timeout or self.timeout_s
            )
            if process.returncode:
                # CLI diagnostics can contain gateway credentials or workload
                # output. Keep them out of daemon errors and agent observations.
                raise IsolationError(
                    "OpenShell command failed; inspect the gateway diagnostics"
                )
            return stdout
        except asyncio.TimeoutError:
            raise IsolationError("OpenShell command timed out") from None
        finally:
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            await self._terminate(process, drain=True)

    async def _json(self, *args, timeout=None):
        try:
            return json.loads(await self._run(self._command(*args), timeout=timeout))
        except (TypeError, ValueError):
            raise IsolationError("OpenShell returned invalid metadata") from None

    @staticmethod
    def _policy_value(policy):
        """Normalize omitted empty sections without weakening path or rule checks."""
        result = json.loads(json.dumps(policy))
        if not isinstance(result, dict):
            raise IsolationError("OpenShell did not return an effective policy")
        for key in ("network_policies", "network_middlewares", "process"):
            result.setdefault(key, {})
        filesystem = result.get("filesystem_policy", {})
        if not isinstance(filesystem, dict):
            raise IsolationError("OpenShell returned an invalid filesystem policy")
        for key in ("read_only", "read_write"):
            filesystem[key] = sorted(set(filesystem.get(key, [])))
        filesystem.setdefault("include_workdir", False)
        result["filesystem_policy"] = filesystem
        return result

    def _admitted(self, effective) -> bool:
        """Whether the gateway applied exactly the requested policy.

        When the request has network rules, OpenShell may add its documented
        baseline paths; additions from that set, and only those, are accepted.
        """
        actual = self._policy_value(effective)
        expected = self._policy_value(self.policy)
        if actual == expected:
            return True
        if not expected["network_policies"]:
            return False
        baseline = {"read_only": BASELINE_READ_ONLY, "read_write": BASELINE_READ_WRITE}
        actual_fs = actual.pop("filesystem_policy")
        expected_fs = expected.pop("filesystem_policy")
        for key, allowed in baseline.items():
            have, want = set(actual_fs.pop(key)), set(expected_fs.pop(key))
            if not want <= have or not have - want <= set(allowed):
                return False
        return actual == expected and actual_fs == expected_fs

    async def start(self, seed: Path, directory: Path):
        if self.name is not None or self._closed:
            raise IsolationError("OpenShell sandbox has already been started")
        if seed.name != "workspace" or not seed.is_dir() or seed.is_symlink():
            raise IsolationError("OpenShell seed must be a workspace directory")
        try:
            pending = [seed]
            while pending:
                for child in pending.pop().iterdir():
                    mode = child.lstat().st_mode
                    if stat.S_ISDIR(mode):
                        pending.append(child)
                    elif not stat.S_ISREG(mode):
                        raise IsolationError(
                            "OpenShell seed contains symlinks or special files"
                        )
        except OSError:
            raise IsolationError(
                "OpenShell seed must contain readable regular files and directories"
            ) from None
        self._cli, self._ssh = shutil.which("openshell"), shutil.which("ssh")
        if not self._cli or not self._ssh:
            raise IsolationError("the openshell CLI and OpenSSH must be installed")

        policy_path = directory / "openshell-policy.yaml"
        policy_path.write_text(yaml.safe_dump(self.policy), encoding="utf-8")
        policy_path.chmod(0o600)
        self.name = "openenvd-" + self._label
        try:
            # Native ephemeral retention provides a daemon-crash backstop after
            # the main process exits. Allow provisioning and teardown grace in
            # addition to the episode deadline; normal cleanup remains explicit.
            await self._run(
                self._command(
                    "sandbox",
                    "create",
                    "--name",
                    self.name,
                    "--from",
                    self.settings.image,
                    "--policy",
                    str(policy_path),
                    "--detach",
                    "--no-keep",
                    "--no-tty",
                    "--no-auto-providers",
                    "--label",
                    "openenv-session=" + self._label,
                    "--",
                    "/bin/sleep",
                    str(max(1, math.ceil(4 * self.timeout_s + 60))),
                )
            )
            effective = await self._json(
                "sandbox", "get", self.name, "--output", "json"
            )
            self._check_identity(effective)
            self.id = effective["id"]
            if (
                effective.get("phase") != "Ready"
                or effective.get("policy_source") != "sandbox"
                or not self._admitted(effective.get("policy"))
            ):
                raise IsolationError(
                    "OpenShell did not admit the requested sandbox policy"
                )
            await self._run(
                self._command(
                    "sandbox",
                    "upload",
                    self.name,
                    str(seed),
                    "/sandbox",
                    "--no-git-ignore",
                )
            )
            config = await self._run(self._command("sandbox", "ssh-config", self.name))
            self._ssh_config = directory / "openshell-ssh.config"
            self._ssh_config.write_text(config, encoding="utf-8")
            self._ssh_config.chmod(0o600)
            self._ready = True
        except BaseException:
            await self.close()
            raise

    def _check_identity(self, value):
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("id"), str)
            or not value["id"]
            or value.get("name") != self.name
            or value.get("workspace") != self.settings.workspace
            or not isinstance(value.get("labels"), dict)
            or value["labels"].get("openenv-session") != self._label
            or (self.id is not None and value["id"] != self.id)
        ):
            raise IsolationError("OpenShell sandbox ownership could not be verified")

    def _ssh_command(self, command):
        return [
            self._ssh,
            "-F",
            str(self._ssh_config),
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "RequestTTY=no",
            "-o",
            "SetEnv=OPENSHELL_NO_LOGIN_SHELL=1",
            f"openshell-{self.name}.{self.settings.workspace}",
            "cd " + shlex.quote(WORKSPACE) + " && exec " + command,
        ]

    async def spawn(self, argv: list[str], env: dict[str, str]):
        if not self._ready or self._closed:
            raise IsolationError("OpenShell sandbox is not available for a process")
        if not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
            raise ValueError("invalid OpenShell worker command")
        if any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or not isinstance(value, str)
            or "\0" in value
            for key, value in env.items()
        ):
            raise ValueError("invalid OpenShell worker environment")
        command = shlex.join(
            [
                "/usr/bin/env",
                "-i",
                *(f"{key}={value}" for key, value in env.items()),
                *argv,
            ]
        )
        process = await self._create_process(
            self._ssh_command(command),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
        )
        self._processes.append(process)
        return process

    async def download(self, destination: Path):
        if not self._ready or self._closed:
            raise IsolationError("OpenShell sandbox is not available for download")
        if (
            destination.is_symlink()
            or not destination.is_dir()
            or any(destination.iterdir())
        ):
            raise IsolationError(
                "OpenShell downloads require an empty staging directory"
            )
        archive = await self._run_bytes(
            self._ssh_command("/usr/bin/env -i /bin/tar -cf - ."),
            output_limit=_WORKSPACE_ARCHIVE_LIMIT,
        )
        if len(archive) > _WORKSPACE_ARCHIVE_LIMIT:
            raise IsolationError("OpenShell workspace exceeded its archive limit")
        try:
            # Use uncompressed tar only. Parse and validate the complete bounded
            # archive before allowing any workload-controlled host writes.
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
                members = {}
                size = 0
                for member in source:
                    name = PurePosixPath(member.name)
                    if (
                        len(members) >= _WORKSPACE_MEMBER_LIMIT
                        or not member.name
                        or "\0" in member.name
                        or name.is_absolute()
                        or ".." in name.parts
                        or name in members
                        or member.type
                        not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE)
                        or member.sparse is not None
                        or any(
                            key.startswith("GNU.sparse") for key in member.pax_headers
                        )
                        or (name == PurePosixPath(".") and not member.isdir())
                        or member.size < 0
                        or member.offset_data + member.size > len(archive)
                    ):
                        raise IsolationError("OpenShell workspace archive is unsafe")
                    size += member.size
                    if size > _WORKSPACE_ARCHIVE_LIMIT:
                        raise IsolationError(
                            "OpenShell workspace exceeded its size limit"
                        )
                    members[name] = member
                for name in members:
                    for parent in name.parents:
                        if parent in members and not members[parent].isdir():
                            raise IsolationError("OpenShell workspace paths conflict")
                for name, member in members.items():
                    target = destination.joinpath(*name.parts)
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    if member.isdir():
                        target.mkdir(exist_ok=True, mode=0o700)
                    else:
                        with (
                            source.extractfile(member) as content,
                            target.open("xb") as output,
                        ):
                            shutil.copyfileobj(content, output, length=65536)
                        target.chmod(stat.S_IMODE(member.mode) & 0o777)
                # Apply directory permissions after all descendants are written.
                for name in sorted(
                    members, key=lambda path: len(path.parts), reverse=True
                ):
                    member = members[name]
                    if member.isdir():
                        destination.joinpath(*name.parts).chmod(
                            stat.S_IMODE(member.mode) & 0o777
                        )
        except IsolationError:
            raise
        except (tarfile.TarError, OSError, ValueError, RecursionError, KeyError):
            raise IsolationError(
                "OpenShell workspace export could not be read safely"
            ) from None

    async def _owned(self, timeout):
        page_token = ""
        seen = set()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise IsolationError("OpenShell sandbox inventory timed out")
            response = await self._json(
                "sandbox",
                "list",
                "--selector",
                "openenv-session=" + self._label,
                "--output",
                "json",
                "--page-token",
                page_token,
                timeout=remaining,
            )
            if (
                not isinstance(response, dict)
                or not isinstance(response.get("sandboxes"), list)
                or not isinstance(response.get("next_page_token"), str)
            ):
                raise IsolationError("OpenShell returned invalid sandbox inventory")
            if response["sandboxes"]:
                sandbox = response["sandboxes"][0]
                self._check_identity(sandbox)
                return sandbox
            next_token = response["next_page_token"]
            if not next_token:
                return None
            if next_token in seen:
                raise IsolationError("OpenShell repeated a sandbox inventory page")
            seen.add(next_token)
            page_token = next_token

    async def close(self):
        if self._closed or self.name is None:
            return
        self._ready = False
        for process in self._processes:
            await self._terminate(process)
        self._processes.clear()
        deadline = time.monotonic() + self.timeout_s
        owned = await self._owned(self.timeout_s)
        if owned is None:
            if self.id is None:
                raise IsolationError(
                    "OpenShell creation outcome is unknown; cleanup must be retried"
                )
            self._closed = True
            self.id = None
            return
        self.id = owned["id"]

        def remaining_timeout():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise IsolationError(
                    "OpenShell cleanup timed out; refusing workspace restore"
                )
            return remaining

        try:
            await self._run(
                self._command("sandbox", "stop", self.name), timeout=remaining_timeout()
            )
        except IsolationError:
            # Delete can still remove an unhealthy sandbox that cannot stop.
            pass
        await self._run(
            self._command("sandbox", "delete", self.name), timeout=remaining_timeout()
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise IsolationError(
                    "OpenShell deletion was not confirmed; refusing workspace restore"
                )
            if await self._owned(remaining) is None:
                self._closed = True
                self.id = None
                return
            await asyncio.sleep(min(0.2, remaining))
