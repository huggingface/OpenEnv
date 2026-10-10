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


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else default


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("", "0", "false", "no", "off"):
        return False
    msg = f"{name} must be a boolean (1/true/yes/on or 0/false/no/off), got {raw!r}"
    raise ValueError(msg)


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
        "enable_rate_limit_auto_ban": _env_bool("OPENENV_GUARD_RATE_LIMIT_AUTO_BAN"),
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
        "passive_mode": _env_bool("OPENENV_GUARD_PASSIVE_MODE"),
        "custom_log_file": os.environ.get("OPENENV_GUARD_LOG_FILE") or None,
        "log_format": os.environ.get("OPENENV_GUARD_LOG_FORMAT", "text"),
        "security_headers": (
            {
                "enabled": True,
                "hsts": {"max_age": 31536000, "include_subdomains": True},
                "frame_options": "SAMEORIGIN",
                "content_type_options": "nosniff",
                "referrer_policy": "strict-origin-when-cross-origin",
            }
            if _env_bool("OPENENV_GUARD_SECURITY_HEADERS")
            else None
        ),
        "enforce_https": _env_bool("OPENENV_GUARD_ENFORCE_HTTPS"),
    }

    # Behind a TLS-terminating proxy, enforce_https must read the forwarded
    # scheme or every request looks like plain HTTP. Follow enforce_https
    # unless explicitly overridden.
    explicit_xfp = _env_str("OPENENV_GUARD_TRUST_X_FORWARDED_PROTO")
    if explicit_xfp is not None:
        kwargs["trust_x_forwarded_proto"] = _env_bool(
            "OPENENV_GUARD_TRUST_X_FORWARDED_PROTO"
        )
    else:
        kwargs["trust_x_forwarded_proto"] = _env_bool("OPENENV_GUARD_ENFORCE_HTTPS")

    if blocked_countries := _env_list("OPENENV_GUARD_BLOCKED_COUNTRIES"):
        kwargs["blocked_countries"] = frozenset(blocked_countries)
    if allowed_countries := _env_list("OPENENV_GUARD_ALLOWED_COUNTRIES"):
        kwargs["whitelist_countries"] = frozenset(allowed_countries)
    if cloud_providers := _env_list("OPENENV_GUARD_BLOCK_CLOUD_PROVIDERS"):
        kwargs["block_cloud_providers"] = frozenset(cloud_providers)

    if ipinfo_token := os.environ.get("IPINFO_TOKEN"):
        kwargs["ipinfo_token"] = ipinfo_token

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
    if getattr(app.state, "guard_attached", False):
        return
    try:
        from guard import SecurityMiddleware
    except ImportError as exc:
        raise ImportError(
            "OPENENV_GUARD_ENABLED requires fastapi-guard. "
            'Install it with: pip install "openenv[security]"'
        ) from exc

    app.add_middleware(SecurityMiddleware, config=_build_security_config())
    app.state.guard_attached = True
