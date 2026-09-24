# SPDX-License-Identifier: BSD-3-Clause
"""Declarative principal policies shared by runtime and manifest validation."""

from __future__ import annotations

from enum import Enum
from fnmatch import fnmatchcase
from ipaddress import IPv4Network
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

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


class EgressRule(BaseModel):
    """Explicit IPv4 destination, protocol, and ports; special ranges stay denied."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    cidr: str
    protocol: Literal["tcp", "udp"]
    ports: tuple[Annotated[int, Field(strict=True, ge=1, le=65535)], ...] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_rule(self) -> EgressRule:
        if str(IPv4Network(self.cidr)) != self.cidr:
            raise ValueError("egress requires a canonical IPv4 CIDR")
        return self


class EgressPolicy(BaseModel):
    """Default-deny egress policy for the workload's kernel network."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    allow: tuple[EgressRule, ...] = ()


class OpenEnvDConfig(BaseModel):
    """The optional ``openenvd`` section of an environment manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    network: EgressPolicy = Field(default_factory=EgressPolicy)
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
