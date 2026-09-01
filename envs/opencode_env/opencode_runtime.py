# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure builders for OpenCode sandbox bootstrap artifacts.

These functions produce the exact files and shell commands the sandbox needs to
run OpenCode against a configured LLM endpoint. No IO, no sandbox coupling —
the sandbox backend is responsible for writing files and running commands.
"""

from __future__ import annotations

import json
from typing import Any

from .config import OpenCodeConfig, provider_npm_package


def opencode_config_path(config: OpenCodeConfig) -> str:
    return f"{config.sandbox_home}/.config/opencode/opencode.json"


def instruction_path(config: OpenCodeConfig) -> str:
    return f"{config.sandbox_home}/task/instruction.md"


def agent_log_path(config: OpenCodeConfig) -> str:
    return f"{config.sandbox_home}/logs/agent/opencode.jsonl"


def system_prompt_path(config: OpenCodeConfig) -> str:
    return f"{config.sandbox_home}/task/system.md"


def verifier_reward_path(config: OpenCodeConfig) -> str:
    return f"{config.sandbox_home}/logs/verifier/reward.txt"


def workdir_path(config: OpenCodeConfig) -> str:
    return f"{config.sandbox_home}/workdir"


def opencode_bin_path(config: OpenCodeConfig) -> str:
    return f"{config.sandbox_home}/.opencode/bin/opencode"


def proxy_dir(config: OpenCodeConfig) -> str:
    return f"{config.sandbox_home}/proxy"


def proxy_source_path(config: OpenCodeConfig) -> str:
    return f"{proxy_dir(config)}/interception.py"


def proxy_trace_path(config: OpenCodeConfig) -> str:
    return f"{config.sandbox_home}/logs/agent/proxy_trace.jsonl"


def proxy_log_path(config: OpenCodeConfig) -> str:
    return f"{config.sandbox_home}/logs/agent/proxy.log"


def build_opencode_json(config: OpenCodeConfig) -> str:
    """Return the serialized ``opencode.json`` the sandbox should install.

    Provider block is keyed by a stable internal name (``intercepted``) so the
    same ``model`` string works across providers. Deep-merges
    ``config.extra_opencode_json`` last so callers can override anything.
    """

    provider_name = "intercepted"
    provider_block: dict[str, Any] = {
        "npm": provider_npm_package(config.provider),
        "name": "Intercepted",
        "options": {
            "baseURL": config.base_url,
            "apiKey": config.api_key,
            "timeout": config.request_timeout_ms,
        },
        "models": {
            config.model.split("/", 1)[-1]: {"name": "Intercepted Model"},
        },
    }

    doc: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"{provider_name}/{config.model.split('/', 1)[-1]}",
        "provider": {provider_name: provider_block},
    }

    tools = _build_tools_block(config)
    if tools:
        doc["tools"] = tools

    _deep_merge(doc, config.extra_opencode_json)
    return json.dumps(doc, indent=2)


def build_install_cmd(config: OpenCodeConfig) -> str:
    """Return the shell command that installs OpenCode + ensures PATH.

    The upstream installer takes ``--version x.y.z`` for pinning (which skips
    the api.github.com "latest" lookup); leaving it unset tracks ``latest``.
    An env-var prefix on ``curl`` would not reach the ``bash`` side of the
    pipe, so the version is passed as an installer argument instead.
    """

    version_args = ""
    if config.opencode_version and config.opencode_version != "latest":
        version_args = f" -s -- --version {config.opencode_version}"
    home = config.sandbox_home
    return (
        "set -e && "
        f"mkdir -p {home}/.config/opencode {home}/logs/agent {home}/logs/verifier {home}/task {home}/workdir && "
        f"curl -fsSL https://opencode.ai/install | bash{version_args} && "
        'export PATH="$HOME/.opencode/bin:$PATH" && '
        "opencode --version"
    )


def build_run_cmd(config: OpenCodeConfig) -> str:
    """Return the shell command that launches OpenCode against a task."""

    format_flag = "--format json" if config.run_format == "json" else ""
    return (
        "set -o pipefail && "
        'export PATH="$HOME/.opencode/bin:$PATH" && '
        f"cd {workdir_path(config)} && "
        f'opencode run {format_flag} "$(cat {instruction_path(config)})" '
        f"2>&1 | tee {agent_log_path(config)}"
    ).strip()


def build_env_vars(config: OpenCodeConfig, *, base_url_override: str | None = None) -> dict[str, str]:
    """Return env vars to set on the OpenCode process.

    When a proxy is wrapping ``config.base_url`` the factory passes the proxy's
    local URL via ``base_url_override`` so the sandbox process points at the
    proxy and the opencode.json on disk stays consistent with what the proxy
    forwards to.
    """

    env = dict(config.extra_env)
    env["OPENAI_BASE_URL"] = base_url_override or config.base_url
    env["OPENAI_API_KEY"] = config.api_key
    env["OPENCODE_CONFIG"] = opencode_config_path(config)
    return env


def _build_tools_block(config: OpenCodeConfig) -> dict[str, bool]:
    """Translate enabled/disabled lists into opencode's ``tools`` map."""

    if config.enabled_tools is not None:
        # Whitelist: everything not listed is disabled. OpenCode treats missing
        # keys as "default enabled", so we only need to explicitly disable the
        # ones we want off. Without a full known-tool list we can't do a true
        # whitelist; document this as a known limitation and require the caller
        # to rely on ``disabled_tools`` for full control.
        return {tool: True for tool in config.enabled_tools}
    return {tool: False for tool in config.disabled_tools}


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursively merge ``src`` into ``dst`` in place."""

    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
