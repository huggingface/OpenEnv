# SPDX-License-Identifier: BSD-3-Clause
"""Enforcement backends: the primitives openenvd builds episodes from."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from ..policy import Guarantee, OpenEnvDConfig


class EnforcementUnavailable(RuntimeError):
    """
    A backend cannot provide what the environment requires.

    Raised before any sandbox exists, instead of starting with weaker isolation
    than the manifest declares.

    Attributes:
        backend (`str`):
            The backend that refused.
        missing (`frozenset[Guarantee]`):
            Required guarantees the backend does not provide; empty when it does
            provide them but its prerequisites are absent on this host.
    """

    def __init__(self, backend: str, reason: str, missing: Iterable[Guarantee] = ()):
        self.backend = backend
        self.missing = frozenset(missing)
        detail = reason
        if self.missing:
            names = ", ".join(sorted(g.value for g in self.missing))
            detail = f"{reason} (missing guarantees: {names})"
        super().__init__(f"enforcement backend {backend!r} unavailable: {detail}")


@runtime_checkable
class Sandbox(Protocol):
    """
    One isolated workload instance, owned by exactly one episode or grading run.

    Attributes:
        id (`str`, *optional*):
            Backend identifier while the sandbox exists; `None` once deletion is
            confirmed.
        workspace (`str` or `Path`):
            Absolute workspace path as processes inside the sandbox see it; spawned
            processes start there.
    """

    id: str | None
    workspace: str | Path

    async def start(self, seed: Path, directory: Path) -> None:
        """
        Create the sandbox and upload a copy of `seed` as its workspace.

        Args:
            seed (`Path`):
                A local directory named `workspace` holding only regular files and
                directories. It is copied, never mounted.
            directory (`Path`):
                A private (mode `0700`) daemon directory for backend state.
        """
        ...

    async def spawn(
        self, argv: list[str], env: dict[str, str]
    ) -> asyncio.subprocess.Process:
        """
        Start a process inside the sandbox with only `env` in its environment.

        Args:
            argv (`list[str]`):
                The command to run.
            env (`dict[str, str]`):
                The complete environment for the process.

        Returns:
            `asyncio.subprocess.Process`: a local handle with piped stdio.
        """
        ...

    async def download(self, destination: Path) -> None:
        """
        Copy the sandbox workspace into the empty local directory `destination`.

        Args:
            destination (`Path`):
                An empty private directory.
        """
        ...

    async def close(self) -> None:
        """Stop every process and confirm the sandbox no longer exists."""
        ...


class EnforcementBackend(ABC):
    """
    Makes the openenvd contract structurally true, one sandbox at a time.

    A backend declares the [`~openenv.core.openenvd.policy.Guarantee`]s it
    provides. [`~openenv.core.openenvd.backends.EnforcementBackend.ensure`]
    refuses when a required guarantee is missing or the backend's host
    prerequisites are absent, so a manifest never claims isolation that the
    runtime silently dropped.

    Args:
        config ([`~openenv.core.openenvd.policy.OpenEnvDConfig`]):
            The environment's validated `openenvd:` block.
    """

    name: str = ""
    guarantees: frozenset[Guarantee] = frozenset()

    def __init__(self, config: OpenEnvDConfig):
        self.config = config

    async def ensure(self, required: Iterable[Guarantee]) -> None:
        """
        Refuse unless this backend can provide every required guarantee here.

        Args:
            required (`Iterable[Guarantee]`):
                Guarantees the environment depends on.

        Raises:
            [`~openenv.core.openenvd.backends.EnforcementUnavailable`]:
                If a guarantee is unsupported or a prerequisite is missing.
        """
        missing = frozenset(required) - self.guarantees
        if missing:
            raise EnforcementUnavailable(
                self.name, "backend does not provide required guarantees", missing
            )
        await self.probe()

    @abstractmethod
    async def probe(self) -> None:
        """
        Check host prerequisites (binaries, versions, gateways, kernel features).

        Raises:
            [`~openenv.core.openenvd.backends.EnforcementUnavailable`]:
                If the backend cannot enforce on this host.
        """

    @abstractmethod
    def sandbox(self, timeout_s: float) -> Sandbox:
        """
        Create an unstarted sandbox.

        Args:
            timeout_s (`float`):
                Bound for each backend operation and the sandbox's lifetime backstop.

        Returns:
            [`~openenv.core.openenvd.backends.Sandbox`]: the sandbox handle.
        """

    @property
    @abstractmethod
    def python(self) -> str:
        """The interpreter used to start Python workers inside a sandbox."""

    def workload_env(self) -> dict[str, str]:
        """
        The minimal environment for processes started inside a sandbox.

        Returns:
            `dict[str, str]`: `PATH`, `HOME`, and `TMPDIR`.
        """
        return {
            "PATH": str(Path(self.python).parent) + ":/usr/bin:/bin",
            "HOME": "/sandbox",
            "TMPDIR": "/tmp",
        }
