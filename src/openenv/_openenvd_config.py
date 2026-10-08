# SPDX-License-Identifier: BSD-3-Clause
"""The openenvd contract, shared by the runtime and manifest validation.

The `openenvd:` block of `openenv.yaml` declares who may reach an environment
(principals and their surfaces), what must stay out of the workload (privileged
assets), what the workload may touch (filesystem and egress), and which
guarantees the environment depends on. A pluggable enforcement backend makes
those guarantees true or refuses to start.

This module imports no server code so validation and clients can load it cheaply.
"""

from __future__ import annotations

import re
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Principal(str, Enum):
    """Who is on the other end of a surface."""

    ORCHESTRATOR = "orchestrator"
    AGENT = "agent"
    GRADER = "grader"
    OBSERVER = "observer"


class ObservationEventType(str, Enum):
    """Kinds of events an observer surface can carry."""

    HARNESS_EVENT = "harness_event"
    FS_CHANGE = "fs_change"
    PROCESS = "process"
    NETWORK = "network"
    RESOURCE = "resource"


class Guarantee(str, Enum):
    """
    A property an enforcement backend makes structurally true for the workload.

    Environments list the guarantees they depend on under
    `openenvd.enforcement.require`. A backend that cannot provide every one of
    them refuses to start the runtime instead of running with weaker isolation.
    """

    ASSET_ISOLATION = "asset_isolation"
    """Privileged assets are unreachable from the workload's filesystem."""

    EGRESS_CONTROL = "egress_control"
    """Workload network egress is limited to the declared egress policy."""

    PRIVILEGE_DROP = "privilege_drop"
    """The workload runs unprivileged and cannot escalate."""

    CONTROL_PLANE_ISOLATION = "control_plane_isolation"
    """The workload cannot reach orchestrator, grader, or observer surfaces."""


RESERVED_TOOL_NAMES = frozenset({"reset", "step", "state", "close"})
_STREAMS = {"harness_events", "fs_diff", "process", "network", "resource"}
SANDBOX_WORKSPACE = "/sandbox/workspace"
DEFAULT_READ_ONLY = (
    "/bin",
    "/usr",
    "/lib",
    "/lib64",
    "/etc",
    "/proc",
    "/opt",
    "/dev/urandom",
)
DEFAULT_READ_WRITE = ("/sandbox", "/tmp", "/dev/null")


def _has_control_chars(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def _absolute_sandbox_path(value: Any) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value.startswith("//")
        or _has_control_chars(value)
        or ".." in PurePosixPath(value).parts
    ):
        raise ValueError("sandbox paths must be absolute without traversal")
    return PurePosixPath(value)


def _check_writable(read_write: Any) -> list[PurePosixPath]:
    if not isinstance(read_write, (list, tuple)):
        raise ValueError("read_write must be a list of paths")
    paths = []
    for value in read_write:
        path = _absolute_sandbox_path(value)
        if not (
            path.is_relative_to("/sandbox")
            or path.is_relative_to("/tmp")
            or path == PurePosixPath("/dev/null")
        ):
            raise ValueError(
                "writable paths must stay within /sandbox or /tmp, or be /dev/null"
            )
        paths.append(path)
    return paths


class SurfacePolicy(BaseModel):
    """
    An explicit allowlist for one principal; unspecified permissions are denied.

    Attributes:
        principal ([`~openenv.core.openenvd.policy.Principal`]):
            The principal this surface serves.
        tools (`tuple[str, ...]`):
            Tool names or patterns. Agent patterns allow only a trailing `*` and can
            never match lifecycle names or `grader.*`.
        fs_read (`tuple[str, ...]`):
            Absolute path globs a grader may read through grader tools.
        stream (`tuple[str, ...]`):
            Event streams an observer may subscribe to.
        allow_lifecycle (`bool`, *optional*, defaults to `False`):
            Whether the surface may reset, step, or close the episode.
        allow_privileged_exec (`bool`, *optional*, defaults to `False`):
            Whether a grader may run privileged processes such as the oracle.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    principal: Principal
    tools: tuple[str, ...] = ()
    fs_read: tuple[str, ...] = ()
    stream: tuple[str, ...] = ()
    allow_lifecycle: bool = False
    allow_privileged_exec: bool = False

    @model_validator(mode="after")
    def validate_boundary(self) -> SurfacePolicy:
        if self.principal in (Principal.AGENT, Principal.OBSERVER):
            if self.allow_lifecycle or self.allow_privileged_exec:
                raise ValueError(
                    "agent and observer cannot control lifecycle or privileged execution"
                )
        if self.allow_privileged_exec and self.principal != Principal.GRADER:
            raise ValueError("privileged execution is restricted to graders")
        if self.principal == Principal.OBSERVER and (self.tools or self.fs_read):
            raise ValueError("observer surfaces expose streams only")
        if self.fs_read and self.principal != Principal.GRADER:
            raise ValueError("filesystem read policies are supported only for graders")
        if set(self.stream) - _STREAMS:
            raise ValueError("unknown observation stream")
        if self.stream and self.principal != Principal.OBSERVER:
            raise ValueError("only observers may subscribe to streams")
        for pattern in self.tools:
            if not pattern or any(c.isspace() for c in pattern):
                raise ValueError("tool patterns must be nonempty without whitespace")
            if self.principal == Principal.AGENT:
                # A literal prefix with an optional trailing '*' keeps overlap
                # with the reserved namespaces decidable.
                prefix = pattern.removesuffix("*")
                if any(c in prefix for c in "*?[]"):
                    raise ValueError(
                        "agent tool patterns support only a trailing wildcard"
                    )
                if any(fnmatchcase(name, pattern) for name in RESERVED_TOOL_NAMES):
                    raise ValueError("agent policy includes lifecycle tools")
                if pattern.startswith("grader.") or (
                    pattern.endswith("*") and "grader.".startswith(prefix)
                ):
                    raise ValueError("agent policy includes privileged tools")
        for pattern in self.fs_read:
            if (
                not pattern.startswith("/")
                or ".." in PurePosixPath(pattern).parts
                or "\0" in pattern
            ):
                raise ValueError(
                    "filesystem policies require absolute paths without traversal"
                )
        return self

    def permits_tool(self, name: str) -> bool:
        """
        Whether this surface exposes the tool `name`.

        Args:
            name (`str`):
                The concrete tool name.

        Returns:
            `bool`: `True` when a declared pattern matches and the principal may hold it.
        """
        if self.principal == Principal.OBSERVER:
            return False
        if self.principal == Principal.AGENT and (
            name in RESERVED_TOOL_NAMES or name.startswith("grader.")
        ):
            return False
        return any(fnmatchcase(name, pattern) for pattern in self.tools)

    def permits_read(self, path: Path | PurePosixPath) -> bool:
        """
        Whether a grader may read the logical path `path`.

        Args:
            path (`Path` or `PurePosixPath`):
                Absolute logical path, e.g. `/workspace/out.txt`.

        Returns:
            `bool`: `True` when a declared `fs_read` glob matches.
        """
        return any(fnmatchcase(str(path), pattern) for pattern in self.fs_read)


class EnforcementSpec(BaseModel):
    """
    Which backend enforces the contract, and what it must guarantee.

    Attributes:
        backend (`str`, *optional*, defaults to `"openshell"`):
            Registered backend name: `"openshell"` or `"local"`.
        require (`tuple[Guarantee, ...]`):
            Guarantees the environment depends on. Startup refuses if the backend
            cannot provide all of them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    backend: str = Field(default="openshell", min_length=1)
    require: tuple[Guarantee, ...] = ()


class WorkloadPaths(BaseModel):
    """
    The filesystem the workload may touch inside its sandbox.

    The episode workspace is uploaded to `/sandbox/workspace`. Writable paths are
    confined to `/sandbox`, `/tmp`, and `/dev/null`.

    Attributes:
        read_only (`tuple[str, ...]`):
            Absolute paths the workload may read.
        read_write (`tuple[str, ...]`):
            Absolute paths the workload may read and write.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    read_only: tuple[str, ...] = DEFAULT_READ_ONLY
    read_write: tuple[str, ...] = DEFAULT_READ_WRITE

    @model_validator(mode="after")
    def validate_paths(self) -> WorkloadPaths:
        for value in self.read_only:
            _absolute_sandbox_path(value)
        _check_writable(self.read_write)
        return self


class EgressRule(BaseModel):
    """
    One permitted egress destination.

    Attributes:
        host (`str`):
            Destination hostname.
        port (`int`, *optional*, defaults to `443`):
            Destination port.
        binaries (`tuple[str, ...]`):
            Absolute paths of the programs allowed to use this rule, e.g. the
            harness interpreter. Empty allows every program in the workload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    host: str = Field(min_length=1, pattern=r"^[A-Za-z0-9*][A-Za-z0-9.*-]*$")
    port: int = Field(default=443, ge=1, le=65535)
    binaries: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_binaries(self) -> EgressRule:
        for value in self.binaries:
            _absolute_sandbox_path(value)
        return self


class EgressPolicy(BaseModel):
    """
    Network egress the workload is allowed.

    Attributes:
        mode (`str`, *optional*, defaults to `"none"`):
            `"none"` (no egress) or `"allowlist"` (only `allow`).
        allow (`tuple[EgressRule, ...]`):
            Permitted destinations. Required with `"allowlist"`, forbidden otherwise.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["none", "allowlist"] = "none"
    allow: tuple[EgressRule, ...] = ()

    @model_validator(mode="after")
    def validate_mode(self) -> EgressPolicy:
        if self.mode == "none" and self.allow:
            raise ValueError("egress.allow is only valid when egress.mode is allowlist")
        if self.mode == "allowlist" and not self.allow:
            raise ValueError("egress.mode allowlist requires at least one rule")
        return self


class ProcessSpec(BaseModel):
    """
    A process openenvd runs alongside the environment worker.

    Attributes:
        trust (`str`):
            `"workload"` runs in the episode sandbox with the worker, started after
            it on every reset. `"privileged"` runs on demand in a fresh sandbox
            holding a workspace snapshot and the privileged assets.
        argv (`tuple[str, ...]`):
            Workload command; `argv[0]` is an absolute path inside the image.
        asset (`str`, *optional*):
            Privileged executable, by privileged asset name.
        args (`tuple[str, ...]`):
            Extra arguments for a privileged executable.
        env (`dict[str, str]`):
            Additional non-secret environment variables.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    trust: Literal["workload", "privileged"]
    argv: tuple[str, ...] = ()
    asset: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shape(self) -> ProcessSpec:
        if self.trust == "workload":
            if not self.argv or self.asset is not None or self.args:
                raise ValueError("workload processes declare argv only")
            _absolute_sandbox_path(self.argv[0])
        elif self.asset is None or self.argv:
            raise ValueError("privileged processes declare an asset, not argv")
        for value in (*self.argv, *self.args, *self.env.values()):
            if "\0" in value:
                raise ValueError("process arguments cannot contain NUL")
        for key in self.env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key.startswith(
                "OPENENVD_"
            ):
                raise ValueError(f"invalid process environment variable {key!r}")
        return self


class OpenShellConfig(BaseModel):
    """
    Options for the `openshell` enforcement backend.

    By default the sandbox policy is rendered from the backend-neutral `workload`
    and `egress` declarations. `policy` replaces it with a native OpenShell
    policy, which must still meet openenvd's isolation invariants.

    Attributes:
        image (`str`):
            Container image holding the environment and OpenEnv.
        gateway (`str`):
            OpenShell gateway name.
        workspace (`str`, *optional*, defaults to `"default"`):
            OpenShell gateway workspace.
        python (`str`, *optional*, defaults to `"/usr/local/bin/python3"`):
            Interpreter used to start the worker; must not be writable.
        run_as_user (`str`, *optional*, defaults to `"1000"`):
            Positive numeric user id for the workload.
        run_as_group (`str`, *optional*, defaults to `"1000"`):
            Positive numeric group id for the workload.
        policy (`dict`, *optional*):
            A complete native OpenShell policy, replacing the rendered one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    image: str = Field(min_length=1)
    gateway: str = Field(min_length=1)
    workspace: str = "default"
    python: str = "/usr/local/bin/python3"
    run_as_user: str = "1000"
    run_as_group: str = "1000"
    policy: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_isolation(self) -> OpenShellConfig:
        if (
            self.image.startswith("-")
            or _has_control_chars(self.image)
            or any(char.isspace() for char in self.image)
        ):
            raise ValueError(
                "OpenShell image must be a single container image reference"
            )
        for name in ("gateway", "workspace"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", getattr(self, name)):
                raise ValueError(f"OpenShell {name} must be a nonempty identifier")
        _absolute_sandbox_path(self.python)
        for name in ("run_as_user", "run_as_group"):
            _check_identity(name, getattr(self, name))
        if self.policy is not None:
            _check_native_policy(self.policy)
        return self


def _check_identity(name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[0-9]+", value)
        or not int(value)
    ):
        raise ValueError(f"OpenShell {name} must be a positive numeric ID")


def _check_native_policy(policy: dict[str, Any]) -> None:
    landlock = policy.get("landlock")
    if not isinstance(landlock, dict) or landlock.get("compatibility") != (
        "hard_requirement"
    ):
        raise ValueError("OpenShell requires Landlock hard_requirement")
    process = policy.get("process")
    if not isinstance(process, dict):
        raise ValueError("OpenShell requires an explicit non-root process identity")
    for name in ("run_as_user", "run_as_group"):
        _check_identity(name, process.get(name))
    filesystem = policy.get("filesystem_policy")
    if (
        not isinstance(filesystem, dict)
        or filesystem.get("include_workdir") is not False
    ):
        raise ValueError("OpenShell filesystem_policy requires include_workdir=false")
    _check_writable(filesystem.get("read_write"))


class OpenEnvDConfig(BaseModel):
    """
    The `openenvd:` block of `openenv.yaml`.

    Attributes:
        enabled (`bool`, *optional*, defaults to `False`):
            Opt-in switch. When `False` the environment runs unchanged.
        enforcement ([`~openenv.core.openenvd.policy.EnforcementSpec`]):
            Backend selection and required guarantees.
        openshell ([`~openenv.core.openenvd.policy.OpenShellConfig`], *optional*):
            Options for the `openshell` backend; required when it is selected.
        surfaces (`dict[Principal, SurfacePolicy]`):
            Declared surface per principal.
        privileged_assets (`dict[str, str]`):
            Asset name to a path relative to the daemon's asset root. Assets stay on
            the daemon host and never enter the episode sandbox.
        workload ([`~openenv.core.openenvd.policy.WorkloadPaths`]):
            Filesystem the workload may touch.
        egress ([`~openenv.core.openenvd.policy.EgressPolicy`]):
            Network egress the workload is allowed.
        processes (`dict[str, ProcessSpec]`):
            Processes beyond the implicit environment worker, by name.

    Examples:

    ```python
    config = OpenEnvDConfig.model_validate(
        {
            "enabled": True,
            "enforcement": {"backend": "openshell", "require": ["asset_isolation"]},
            "openshell": {"image": "my-env:openshell", "gateway": "local"},
            "privileged_assets": {"oracle": "grade.sh"},
        }
    )
    ```
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    enforcement: EnforcementSpec = Field(default_factory=EnforcementSpec)
    openshell: OpenShellConfig | None = None
    surfaces: dict[Principal, SurfacePolicy] = Field(default_factory=dict)
    privileged_assets: dict[str, str] = Field(default_factory=dict)
    workload: WorkloadPaths = Field(default_factory=WorkloadPaths)
    egress: EgressPolicy = Field(default_factory=EgressPolicy)
    processes: dict[str, ProcessSpec] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def bind_principals(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        surfaces = value.get("surfaces", {})
        if isinstance(surfaces, dict):
            bound = {}
            for principal, policy in surfaces.items():
                if isinstance(policy, dict):
                    if "principal" in policy and policy["principal"] != principal:
                        raise ValueError("surface key and principal disagree")
                    policy = {**policy, "principal": principal}
                elif (
                    isinstance(policy, SurfacePolicy) and policy.principal != principal
                ):
                    raise ValueError("surface key and principal disagree")
                bound[principal] = policy
            value["surfaces"] = bound
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> OpenEnvDConfig:
        for name, path in self.privileged_assets.items():
            if not name or not name.replace("_", "").isalnum():
                raise ValueError("asset names must be alphanumeric")
            parsed = PurePosixPath(path)
            if not path or parsed.is_absolute() or ".." in parsed.parts or "\0" in path:
                raise ValueError(
                    "asset sources must be relative paths within the asset root"
                )
        for name, process in self.processes.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name == "worker":
                raise ValueError(f"invalid process name {name!r}")
            if (
                process.asset is not None
                and process.asset not in self.privileged_assets
            ):
                raise ValueError(f"process {name!r} names an undeclared asset")
        if self.openshell is not None and self.openshell.policy is not None:
            # Compare with defaults, not fields_set, so dumps round-trip.
            if self.workload != WorkloadPaths() or self.egress != EgressPolicy():
                raise ValueError(
                    "declare workload and egress, or a native openshell.policy, not both"
                )
        if self.enabled and self.enforcement.backend == "openshell":
            if self.openshell is None:
                raise ValueError("the openshell backend requires an openshell section")
            python = PurePosixPath(self.openshell.python)
            for path in _check_writable(self.writable_paths()):
                if python.is_relative_to(path):
                    raise ValueError(
                        "the worker Python interpreter must not be writable"
                    )
        return self

    def privileged_process(self, name: str) -> ProcessSpec | None:
        """
        The privileged process `name`, including the implicit `oracle`.

        Args:
            name (`str`):
                Process name.

        Returns:
            [`~openenv.core.openenvd.policy.ProcessSpec`] or `None`: the process
            when it is declared privileged, or `name == "oracle"` with an `oracle`
            asset and no explicit declaration.
        """
        process = self.processes.get(name)
        if process is not None:
            return process if process.trust == "privileged" else None
        if name == "oracle" and "oracle" in self.privileged_assets:
            return ProcessSpec(trust="privileged", asset="oracle")
        return None

    def writable_paths(self) -> tuple[str, ...]:
        """
        The workload's effective writable paths.

        Returns:
            `tuple[str, ...]`: from the native OpenShell policy when one is given,
            otherwise from `workload.read_write`.
        """
        if self.openshell is not None and self.openshell.policy is not None:
            return tuple(self.openshell.policy["filesystem_policy"]["read_write"])
        return self.workload.read_write


def load_config(manifest: Path) -> OpenEnvDConfig:
    """
    Read and validate the `openenvd:` block without importing environment code.

    Args:
        manifest (`Path`):
            Path to `openenv.yaml`.

    Returns:
        [`~openenv.core.openenvd.policy.OpenEnvDConfig`]: the validated block, or a
        disabled default when it is absent.
    """
    data = yaml.safe_load(Path(manifest).read_text())
    if not isinstance(data, dict):
        raise ValueError("openenv.yaml must contain a mapping")
    return OpenEnvDConfig.model_validate(data.get("openenvd") or {})
