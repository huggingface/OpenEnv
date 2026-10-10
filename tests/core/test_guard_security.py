# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the optional fastapi-guard security middleware wiring."""

import itertools
import os

import httpx
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.security import _build_security_config
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

# Rate limiting state in guard-core is process-wide, so every scenario gets
# its own TEST-NET client IP to stay hermetic.
_IP = itertools.count(1)


def _unique_ip() -> str:
    n = next(_IP)
    return f"198.51.{n // 250}.{(n % 250) + 1}"


class GuardAction(Action):
    value: str = ""


class GuardObservation(Observation):
    message: str = ""


class GuardEnvironment(Environment):
    def reset(self, **kwargs) -> GuardObservation:
        return GuardObservation(message="ready")

    def step(self, action: GuardAction, **kwargs) -> GuardObservation:
        return GuardObservation(message=action.value, reward=1.0)

    @property
    def state(self) -> State:
        return State()


def _make_app(monkeypatch, **env):
    for name in list(os.environ):
        if name.startswith("OPENENV_GUARD_"):
            monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return create_fastapi_app(GuardEnvironment, GuardAction, GuardObservation)


async def _request(app, path, client_ip=None):
    transport = httpx.ASGITransport(app=app, client=(client_ip or _unique_ip(), 50000))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        return await client.get(path)


async def test_disabled_by_default(monkeypatch):
    app = _make_app(monkeypatch)
    assert len(app.user_middleware) == 0
    response = await _request(app, "/health")
    assert response.status_code == 200


async def test_enabled_adds_middleware(monkeypatch):
    app = _make_app(monkeypatch, OPENENV_GUARD_ENABLED="1")
    assert len(app.user_middleware) == 1
    response = await _request(app, "/health")
    assert response.status_code == 200


async def test_blocked_ip_is_rejected(monkeypatch):
    blocked = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_BLOCKED_IPS=blocked,
    )
    response = await _request(app, "/health", client_ip=blocked)
    assert response.status_code == 403


async def test_ip_lists_enforced_on_excluded_paths(monkeypatch):
    blocked = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_BLOCKED_IPS=blocked,
    )
    # /health is in the default exclusion list, but global IP lists are
    # enforced there too.
    response = await _request(app, "/health", client_ip=blocked)
    assert response.status_code == 403


async def test_full_bundle_mapping(monkeypatch):
    for name in list(os.environ):
        if name.startswith("OPENENV_GUARD_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENENV_GUARD_ENABLED", "1")
    monkeypatch.setenv("OPENENV_GUARD_PASSIVE_MODE", "1")
    monkeypatch.setenv("OPENENV_GUARD_SECURITY_HEADERS", "1")
    monkeypatch.setenv("OPENENV_GUARD_ENFORCE_HTTPS", "1")
    monkeypatch.setenv("OPENENV_GUARD_BLOCKED_COUNTRIES", "RU")
    monkeypatch.setenv("IPINFO_TOKEN", "test-token")
    monkeypatch.setenv("OPENENV_GUARD_BLOCK_CLOUD_PROVIDERS", "AWS")
    monkeypatch.setenv("OPENENV_GUARD_LOG_FILE", "guard.log")
    config = _build_security_config()
    assert config.passive_mode is True
    assert config.enforce_https is True
    assert config.security_headers["enabled"] is True
    assert config.blocked_countries == frozenset({"RU"})
    assert config.block_cloud_providers == frozenset({"AWS"})
    assert config.custom_log_file == "guard.log"
    assert config.enable_redis is False


async def test_rate_limit_returns_429(monkeypatch):
    ip = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_RATE_LIMIT="2",
        OPENENV_GUARD_RATE_LIMIT_WINDOW="60",
    )
    assert (await _request(app, "/metadata", client_ip=ip)).status_code == 200
    assert (await _request(app, "/metadata", client_ip=ip)).status_code == 200
    assert (await _request(app, "/metadata", client_ip=ip)).status_code == 429
