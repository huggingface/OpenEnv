# SPDX-License-Identifier: BSD-3-Clause
"""Declarative principal policies shared by runtime and manifest validation."""

from __future__ import annotations

import re
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Principal(str, Enum):
    ORCHESTRATOR = "orchestrator"
    AGENT = "agent"
    GRADER = "grader"
    OBSERVER = "observer"


class ObservationEventType(str, Enum):
    HARNESS_EVENT = "harness_event"
    FS_CHANGE = "fs_change"
    PROCESS = "process"
    NETWORK = "network"
    RESOURCE = "resource"


_RESERVED = {"reset", "step", "state", "close"}
_STREAMS = {"harness_events", "fs_diff", "process", "network", "resource"}


class SurfacePolicy(BaseModel):
    """An explicit allowlist for one principal; unspecified permissions are denied."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    principal: Principal
    tools: tuple[str, ...] = ()
    fs_read: tuple[str, ...] = Field(
        default=(),
        description="Allowed file paths for grader tools; unsupported for other principals.",
    )
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
                # Restrict glob syntax to a literal prefix and optional trailing '*'.
                # This makes overlap with the entire reserved namespace decidable.
                prefix = pattern.removesuffix("*")
                if any(c in prefix for c in "*?[]"):
                    raise ValueError(
                        "agent tool patterns support only a trailing wildcard"
                    )
                if any(fnmatchcase(name, pattern) for name in _RESERVED):
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
        if self.principal == Principal.OBSERVER:
            return False
        if self.principal == Principal.AGENT and (
            name in _RESERVED or name.startswith("grader.")
        ):
            return False
        return any(fnmatchcase(name, pattern) for pattern in self.tools)

    def permits_read(self, path: Path) -> bool:
        return any(fnmatchcase(str(path), pattern) for pattern in self.fs_read)


def _default_openshell_policy() -> dict[str, Any]:
    return {
        "version": 1,
        "filesystem_policy": {
            "include_workdir": False,
            "read_only": [
                "/bin",
                "/usr",
                "/lib",
                "/lib64",
                "/etc",
                "/proc",
                "/opt",
                "/dev/urandom",
            ],
            "read_write": ["/sandbox", "/tmp", "/dev/null"],
        },
        "landlock": {"compatibility": "hard_requirement"},
        "process": {"run_as_user": "1000", "run_as_group": "1000"},
        "network_policies": {},
        "network_middlewares": {},
    }


def _absolute_sandbox_path(value: Any) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value.startswith("//")
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or ".." in PurePosixPath(value).parts
    ):
        raise ValueError("OpenShell paths must be absolute without traversal")
    return PurePosixPath(value)


class OpenShellConfig(BaseModel):
    """OpenShell gateway, image, and native sandbox policy for each episode.

    The image must contain the environment and OpenEnv installation. Network
    policy uses OpenShell's native schema and is validated by OpenShell itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    image: str = Field(min_length=1)
    gateway: str = Field(min_length=1)
    workspace: str = "default"
    python: str = "/usr/local/bin/python3"
    policy: dict[str, Any] = Field(default_factory=_default_openshell_policy)

    @model_validator(mode="after")
    def validate_isolation(self) -> OpenShellConfig:
        if self.image.startswith("-") or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in self.image
        ):
            raise ValueError(
                "OpenShell image must be a single container image reference"
            )
        for name in ("gateway", "workspace"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", getattr(self, name)):
                raise ValueError(f"OpenShell {name} must be a nonempty identifier")
        landlock = self.policy.get("landlock")
        if not isinstance(landlock, dict) or landlock.get("compatibility") != (
            "hard_requirement"
        ):
            raise ValueError("OpenShell requires Landlock hard_requirement")
        process = self.policy.get("process")
        if not isinstance(process, dict):
            raise ValueError("OpenShell requires an explicit non-root process identity")
        for name in ("run_as_user", "run_as_group"):
            value = process.get(name)
            if (
                not isinstance(value, str)
                or not re.fullmatch(r"[0-9]+", value)
                or int(value) == 0
            ):
                raise ValueError(f"OpenShell {name} must be a positive numeric ID")
        filesystem = self.policy.get("filesystem_policy")
        if (
            not isinstance(filesystem, dict)
            or filesystem.get("include_workdir") is not False
        ):
            raise ValueError(
                "OpenShell filesystem_policy requires include_workdir=false"
            )
        read_write = filesystem.get("read_write")
        if not isinstance(read_write, list):
            raise ValueError("OpenShell filesystem_policy.read_write must be a list")
        python = _absolute_sandbox_path(self.python)
        for value in read_write:
            path = _absolute_sandbox_path(value)
            if not (
                path.is_relative_to("/sandbox")
                or path.is_relative_to("/tmp")
                or path == PurePosixPath("/dev/null")
            ):
                raise ValueError(
                    "OpenShell writable paths must stay within /sandbox or /tmp, "
                    "or be /dev/null"
                )
            if python.is_relative_to(path):
                raise ValueError("OpenShell Python interpreter must not be writable")
        return self


class OpenEnvDConfig(BaseModel):
    """The optional ``openenvd`` section of an environment manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    openshell: OpenShellConfig | None = None
    surfaces: dict[Principal, SurfacePolicy] = Field(default_factory=dict)
    privileged_assets: dict[str, str] = Field(default_factory=dict)

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
    def protect_assets(self) -> OpenEnvDConfig:
        for name, path in self.privileged_assets.items():
            if not name or not name.replace("_", "").isalnum():
                raise ValueError("asset names must be alphanumeric")
            parsed = PurePosixPath(path)
            if not path or parsed.is_absolute() or ".." in parsed.parts or "\0" in path:
                raise ValueError(
                    "asset sources must be relative paths within the environment"
                )
        return self


def load_config(manifest: Path) -> OpenEnvDConfig:
    """Read and validate a manifest without importing environment code."""
    data = yaml.safe_load(manifest.read_text())
    if not isinstance(data, dict):
        raise ValueError("openenv.yaml must contain a mapping")
    return OpenEnvDConfig.model_validate(data.get("openenvd", {}))
