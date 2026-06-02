# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration model for the OpenCode harness primitive."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Provider = Literal["openai_compatible", "openai", "anthropic"]


class OpenCodeConfig(BaseModel):
    """All configuration required to launch one OpenCode rollout in a sandbox.

    Field names are provider-agnostic. The primitive maps ``provider`` onto the
    correct ``opencode.json`` provider block (``@ai-sdk/openai-compatible``,
    ``@ai-sdk/openai``, or ``@ai-sdk/anthropic``) and injects ``base_url`` /
    ``api_key`` into it.
    """

    # --- LLM endpoint ---------------------------------------------------------
    provider: Provider = "openai_compatible"
    base_url: str
    api_key: str = "intercepted"
    model: str = "intercepted/model"
    request_timeout_ms: int = 600_000

    # --- OpenCode CLI ---------------------------------------------------------
    opencode_version: str = "latest"
    disabled_tools: list[str] = Field(
        default_factory=lambda: ["webfetch", "question"]
    )
    enabled_tools: list[str] | None = None
    system_prompt: str | None = None
    extra_opencode_json: dict[str, Any] = Field(default_factory=dict)

    # --- CLI invocation -------------------------------------------------------
    run_format: Literal["default", "json"] = "json"
    agent_timeout_s: float = 900.0
    extra_env: dict[str, str] = Field(default_factory=dict)
    extra_setup_shell: str | None = None

    # --- Sandbox paths --------------------------------------------------------
    # Root directory inside the sandbox where the primitive writes config,
    # task files, and logs. E2B's default user is ``user`` with home
    # ``/home/user``. Override when using a root-privileged backend (Docker).
    sandbox_home: str = "/home/user"

    # --- Transparent-proxy tuning --------------------------------------------
    # Cap ``max_tokens`` / ``max_completion_tokens`` on forwarded requests.
    # OpenCode defaults to a very large number (~32000) which exceeds some
    # provider limits (e.g. gpt-4o-mini = 16384). Only used in
    # ``mode="transparent_proxy"``. ``None`` disables the cap.
    proxy_max_tokens_cap: int | None = 16384
    # Per-turn top-k logprobs the proxy requests from the upstream.
    proxy_top_logprobs: int = 5
    # Disable reasoning/thinking mode for Qwen3 / Qwen3.5 models. Proxy sets
    # ``extra_body.chat_template_kwargs.enable_thinking=false`` on forwarded
    # requests. Ignored by providers that don't support the field.
    proxy_disable_thinking: bool = False


_PROVIDER_NPM = {
    "openai_compatible": "@ai-sdk/openai-compatible",
    "openai": "@ai-sdk/openai",
    "anthropic": "@ai-sdk/anthropic",
}


def provider_npm_package(provider: Provider) -> str:
    """Return the AI SDK npm package opencode should use for a provider."""
    return _PROVIDER_NPM[provider]
