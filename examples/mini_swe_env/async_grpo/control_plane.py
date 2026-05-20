from __future__ import annotations

import contextlib
import queue as _queue_mod
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

from openenv.core.harness.agents.interception_server import InterceptionServer


_log = logging.getLogger(__name__)

_DEFAULT_INTERCEPTION_HOST = "0.0.0.0"
_DEFAULT_INTERCEPTION_PORT = 8765


def build_hf_space_base_url(space_id: str) -> str:
    """Return canonical public Space URL from ``owner/name``."""
    owner, sep, name = (space_id or "").strip().partition("/")
    if not sep or not owner or not name:
        raise ValueError(
            "SPACE_ID must be in 'owner/name' format to derive interception base URL"
        )
    return f"https://{owner}-{name}.hf.space"


@dataclass(frozen=True)
class SWEAsyncControlPlaneConfig:
    """Runtime config for the Space-hosted interception control plane."""

    host: str
    port: int
    auth_token: str
    interception_base_url: str

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        require_auth_token: bool = True,
    ) -> "SWEAsyncControlPlaneConfig":
        source = env or os.environ

        host = source.get("INTERCEPTION_HOST", _DEFAULT_INTERCEPTION_HOST).strip()
        port_raw = source.get("INTERCEPTION_PORT") or source.get("PORT")
        port = int(port_raw) if port_raw else _DEFAULT_INTERCEPTION_PORT

        token = (
            source.get("INTERCEPTION_AUTH_TOKEN")
            or source.get("INTERCEPTION_SECRET")
            or ""
        ).strip()
        if require_auth_token and not token:
            raise ValueError(
                "Missing interception auth token. Set INTERCEPTION_AUTH_TOKEN in Space secrets."
            )

        base_url = (source.get("INTERCEPTION_BASE_URL") or "").strip().rstrip("/")
        if not base_url:
            space_host = (source.get("SPACE_HOST") or "").strip().rstrip("/")
            if space_host:
                if "http" not in space_host:
                    space_host = f"https://{space_host}"
                base_url = space_host.rstrip("/")
            else:
                space_id = (source.get("SPACE_ID") or "").strip()
                if space_id:
                    base_url = build_hf_space_base_url(space_id)

        if not base_url:
            raise ValueError(
                "Could not resolve interception_base_url. Set INTERCEPTION_BASE_URL, "
                "SPACE_HOST, or SPACE_ID."
            )

        return cls(
            host=host,
            port=port,
            auth_token=token,
            interception_base_url=base_url,
        )


class SWEAsyncControlPlane:
    """Owns the interception server and rollout registration state.

    Trainer-side async rollout workers should use this object directly for
    ``register_rollout/get_intercept/unregister_rollout`` so rollout queues and
    intercept state remain in-process with the trainer.
    """

    def __init__(
        self,
        *,
        config: SWEAsyncControlPlaneConfig,
        server: InterceptionServer | None = None,
    ) -> None:
        self.config = config
        self.server = server or InterceptionServer(
            host=config.host,
            port=config.port,
            secret=config.auth_token,
            tool_name_allowlist={"answer"},
        )

    @property
    def interception_base_url(self) -> str:
        return self.config.interception_base_url

    @property
    def auth_token(self) -> str:
        return self.config.auth_token

    async def start(self) -> None:
        await self.server.start()
        # If port was 0 in config, InterceptionServer resolves it after bind.
        _log.info(
            "swe_async_control_plane_started base_url=%s port=%d",
            self.interception_base_url,
            self.server.port,
        )

    async def stop(self) -> None:
        await self.server.stop()
        _log.info("swe_async_control_plane_stopped")

    def register_rollout(
        self,
        rollout_id: str,
        *,
        state: dict[str, Any] | None = None,
    ) -> _queue_mod.Queue[str]:
        queue = self.server.register_rollout(rollout_id, state=state)
        stats = self.stats()
        _log.info(
            "swe_async_rollout_registered rollout_id=%s active_rollouts=%d",
            rollout_id,
            stats["active_rollouts"],
        )
        return queue

    def unregister_rollout(self, rollout_id: str) -> None:
        self.server.unregister_rollout(rollout_id)
        stats = self.stats()
        _log.info(
            "swe_async_rollout_unregistered rollout_id=%s active_rollouts=%d",
            rollout_id,
            stats["active_rollouts"],
        )

    @contextlib.contextmanager
    def rollout(
        self,
        rollout_id: str,
        *,
        state: dict[str, Any] | None = None,
    ):
        """Leak-safe context manager for one rollout registration."""
        queue = self.register_rollout(rollout_id, state=state)
        try:
            yield queue
        finally:
            self.unregister_rollout(rollout_id)

    def stats(self) -> dict[str, int]:
        return self.server.stats()
