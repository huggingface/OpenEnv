# SPDX-License-Identifier: BSD-3-Clause

"""Optional fastapi-guard security middleware for environment servers.

Opt-in via the `OPENENV_GUARD_*` environment variables. When disabled (the
default), `attach_guard` is a no-op and environment servers behave exactly as
before.

Install the extra with `pip install "openenv[security]"` and enable with
`OPENENV_GUARD_ENABLED=1` to get IP block/allow lists, rate limiting with
auto-ban, user-agent blocking, penetration-attempt detection, and optional
Redis-backed distributed state (fastapi-guard).

Health and schema endpoints are excluded from detection checks by default so
orchestrators can probe the server; global IP allow/block lists are still
enforced on those paths.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_EXCLUDED_PATHS = "/health,/schema,/metadata,/docs,/redoc,/openapi.json"
DEFAULT_TRUSTED_PROXIES = "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


def _env_list(name: str, default: str = "") -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_security_config() -> Any:
    from guard import SecurityConfig

    kwargs: dict[str, Any] = {
        "enable_rate_limiting": True,
        "rate_limit": int(os.environ.get("OPENENV_GUARD_RATE_LIMIT", "100")),
        "rate_limit_window": int(
            os.environ.get("OPENENV_GUARD_RATE_LIMIT_WINDOW", "60")
        ),
        "enable_ip_banning": True,
        "auto_ban_threshold": int(
            os.environ.get("OPENENV_GUARD_AUTO_BAN_THRESHOLD", "10")
        ),
        "auto_ban_duration": int(
            os.environ.get("OPENENV_GUARD_AUTO_BAN_DURATION", "300")
        ),
        "enable_penetration_detection": True,
        # In-memory state unless a Redis URL is configured: never implicitly
        # depend on a Redis server being reachable at localhost.
        "enable_redis": False,
        "blacklist": tuple(_env_list("OPENENV_GUARD_BLOCKED_IPS")),
        "blocked_user_agents": _env_list("OPENENV_GUARD_BLOCKED_USER_AGENTS"),
        "trusted_proxies": tuple(
            _env_list("OPENENV_GUARD_TRUSTED_PROXIES", DEFAULT_TRUSTED_PROXIES)
        ),
        "trusted_proxy_depth": int(
            os.environ.get("OPENENV_GUARD_TRUSTED_PROXY_DEPTH", "1")
        ),
        "exclude_paths": _env_list(
            "OPENENV_GUARD_EXCLUDED_PATHS", DEFAULT_EXCLUDED_PATHS
        ),
    }

    allowed_ips = _env_list("OPENENV_GUARD_ALLOWED_IPS")
    if allowed_ips:
        kwargs["whitelist"] = tuple(allowed_ips)

    redis_url = os.environ.get("OPENENV_GUARD_REDIS_URL")
    if redis_url:
        kwargs["enable_redis"] = True
        kwargs["redis_url"] = redis_url
        kwargs["redis_prefix"] = os.environ.get(
            "OPENENV_GUARD_REDIS_PREFIX", "openenv_guard:"
        )

    return SecurityConfig(**kwargs)


def attach_guard(app: Any) -> None:
    """Add the fastapi-guard middleware to `app` when enabled via env.

    No-op unless OPENENV_GUARD_ENABLED is set to 1/true/yes.
    """
    enabled = os.environ.get("OPENENV_GUARD_ENABLED", "").strip().lower()
    if enabled not in ("1", "true", "yes"):
        return
    try:
        from guard import SecurityMiddleware
    except ImportError as exc:
        raise ImportError(
            "OPENENV_GUARD_ENABLED requires fastapi-guard. "
            'Install it with: pip install "openenv[security]"'
        ) from exc

    app.add_middleware(SecurityMiddleware, config=_build_security_config())
