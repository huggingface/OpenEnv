# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Endpoint shorthand catalog.

Lets the MCP tool ``run_rollout`` and the HTMX UI accept a short endpoint
label (``vllm`` / ``openai`` / ``hf_router``) and resolve the actual
``base_url`` / ``api_key`` / ``model`` from environment variables (with
sane defaults). Explicit overrides on the call always win.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


ENDPOINT_KINDS = ("vllm", "openai", "hf_router")


@dataclass(frozen=True)
class _EndpointSpec:
    base_url_env: str
    api_key_env: str
    model_env: str
    default_base_url: str | None
    default_api_key: str | None
    default_model: str | None
    disable_thinking_default: bool


_CATALOG: dict[str, _EndpointSpec] = {
    "vllm": _EndpointSpec(
        base_url_env="VLLM_URL",
        api_key_env="VLLM_API_KEY",
        model_env="VLLM_MODEL",
        default_base_url=None,  # cluster URL must be set
        default_api_key="intercepted",  # vLLM rarely enforces auth
        default_model="Qwen/Qwen3.5-4B",
        disable_thinking_default=True,  # Qwen3.5 thinks by default
    ),
    "openai": _EndpointSpec(
        base_url_env="OPENAI_BASE_URL",
        api_key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        default_base_url="https://api.openai.com/v1",
        default_api_key=None,
        default_model="gpt-4o-mini",
        disable_thinking_default=False,  # OpenAI rejects unknown kwargs
    ),
    "hf_router": _EndpointSpec(
        base_url_env="HF_ROUTER_BASE_URL",
        api_key_env="HF_ROUTER_API_KEY",
        model_env="HF_ROUTER_MODEL",
        default_base_url="https://router.huggingface.co/v1",
        default_api_key=None,
        default_model="Qwen/Qwen3-4B-Instruct-2507:nscale",
        disable_thinking_default=False,  # Instruct variant doesn't think
    ),
}


@dataclass(frozen=True)
class ResolvedEndpoint:
    kind: str
    base_url: str
    api_key: str
    model: str
    disable_thinking_default: bool


def resolve_endpoint(
    kind: str,
    *,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
) -> ResolvedEndpoint:
    """Resolve an endpoint shorthand into concrete (base_url, api_key, model).

    Precedence per field: **explicit arg > env var > catalog default**.
    Always normalizes to a ``/v1`` base URL.

    Raises ``ValueError`` for unknown kinds, missing creds, or missing model.
    """
    spec = _CATALOG.get(kind)
    if spec is None:
        raise ValueError(
            f"unknown endpoint kind: {kind!r}; expected one of {ENDPOINT_KINDS}"
        )
    base = (
        base_url or os.environ.get(spec.base_url_env) or spec.default_base_url or ""
    ).rstrip("/")
    if not base:
        raise ValueError(
            f"{kind}: no base_url (set {spec.base_url_env} env var or pass "
            "base_url=...)"
        )
    if not base.endswith("/v1"):
        base = f"{base}/v1"

    key = api_key or os.environ.get(spec.api_key_env) or spec.default_api_key or ""
    if not key:
        raise ValueError(
            f"{kind}: no api_key (set {spec.api_key_env} env var or pass api_key=...)"
        )

    mdl = model or os.environ.get(spec.model_env) or spec.default_model or ""
    if not mdl:
        raise ValueError(
            f"{kind}: no model (set {spec.model_env} env var or pass model=...)"
        )

    return ResolvedEndpoint(
        kind=kind,
        base_url=base,
        api_key=key,
        model=mdl,
        disable_thinking_default=spec.disable_thinking_default,
    )


def catalog_summary() -> list[dict[str, object]]:
    """Return a JSON-friendly view of the catalog (for the UI dropdown)."""
    out: list[dict[str, object]] = []
    for kind, spec in _CATALOG.items():
        out.append(
            {
                "kind": kind,
                "base_url_env": spec.base_url_env,
                "api_key_env": spec.api_key_env,
                "model_env": spec.model_env,
                "default_base_url": spec.default_base_url,
                "default_model": spec.default_model,
                "disable_thinking_default": spec.disable_thinking_default,
                "configured": _is_configured(spec),
            }
        )
    return out


def _is_configured(spec: _EndpointSpec) -> bool:
    base = os.environ.get(spec.base_url_env) or spec.default_base_url or ""
    key = os.environ.get(spec.api_key_env) or spec.default_api_key or ""
    model = os.environ.get(spec.model_env) or spec.default_model or ""
    return bool(base and key and model)
