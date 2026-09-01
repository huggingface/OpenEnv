"""Validation provider protocol.

Validation needs more than start/stop: network-policy enforcement and in-sandbox exec
are grader dependencies. Rather than widen the core provider ABCs, validation defines
its own protocol and adapts the core providers (Docker-local, HF Sandbox) behind it.

GPU is a provider capability, not an unsupported package category: packages declaring
GPU resources validate on providers that offer GPUs; elsewhere the affected checks
SKIP with the capability named.

Universal invariants inherited from core: internal port 8000, readiness =
`GET /health` returning 200.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..manifest import NetworkPolicy
from ..types import ProviderCapability


@dataclass(frozen=True)
class ExecResult:
    """Result of executing a command inside the running sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float


@runtime_checkable
class RunningSubject(Protocol):
    """A started validation subject: reachable, execable, stoppable."""

    base_url: str

    def exec(self, argv: list[str], timeout_s: float) -> ExecResult: ...

    def stop(self) -> None: ...


@runtime_checkable
class ValidationProvider(Protocol):
    """
    Starts validation subjects in a sandbox with declared capabilities.

    A grader whose `requires_provider` names a capability the provider lacks is
    SKIPped with the capability named. The subject starts under the given network
    policy (the runner passes the manifest's declared policy; `None` means the
    default `public` mode — egress allowed). Enforcing `no-network`/`allowlist`
    modes requires the `NETWORK_POLICY` capability.
    """

    name: str
    capabilities: frozenset[ProviderCapability]

    def start(
        self,
        image_ref: str,
        *,
        network: NetworkPolicy | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> RunningSubject: ...
