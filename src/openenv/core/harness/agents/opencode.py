# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenCode agent adapter.

Expresses the OpenCode harness as a purely declarative :class:`CLIAgentSpec`.
All builders (command construction, config generation, env var resolution)
are self-contained with no imports from ``envs/opencode_env/``.

Registered on import::

    import openenv.core.harness.agents.opencode
    # OPENCODE_SPEC is now in the registry
"""

from __future__ import annotations

import json
from typing import Any

from . import register_agent
from .base import AgentEvent, ArtifactSpec, CLIAgentSpec, MCPConfigSpec


# Command / config / env builders


def _build_opencode_command(
    spec: CLIAgentSpec,
    config: Any,
    task: Any,
    mcp_config_path: str | None,
) -> str:
    """Build the ``opencode run`` shell command."""
    home = config.sandbox_home if hasattr(config, "sandbox_home") else "/home/user"
    run_format = config.run_format if hasattr(config, "run_format") else "json"
    format_flag = "--format json" if run_format == "json" else ""
    instruction_file = f"{home}/task/instruction.md"
    log_file = f"{home}/logs/agent/opencode.jsonl"
    workdir = f"{home}/workdir"

    return (
        f'export PATH="$HOME/.opencode/bin:$PATH" && '
        f"cd {workdir} && "
        f'opencode run {format_flag} "$(cat {instruction_file})" '
        f"2>&1 | tee {log_file}"
    ).strip()


def _build_opencode_mcp_config(
    spec: CLIAgentSpec,
    tools: list[Any],
    workdir: str,
) -> str:
    """Build the ``opencode.json`` content for the MCP config file."""
    return json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "model": "intercepted/model",
            "provider": {
                "intercepted": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Intercepted",
                    "options": {
                        "baseURL": "http://127.0.0.1:7000/v1",
                        "apiKey": "intercepted",
                        "timeout": 600000,
                    },
                    "models": {
                        "model": {"name": "Intercepted Model"},
                    },
                }
            },
        },
        indent=2,
    )


def _build_opencode_env_vars(
    spec: CLIAgentSpec,
    config: Any,
) -> dict[str, str]:
    """Build env vars for the OpenCode process."""
    home = config.sandbox_home if hasattr(config, "sandbox_home") else "/home/user"
    base_url = config.base_url if hasattr(config, "base_url") else ""
    api_key = config.api_key if hasattr(config, "api_key") else "intercepted"
    extra_env = config.extra_env if hasattr(config, "extra_env") else {}

    env = dict(extra_env)
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_KEY"] = api_key
    env["OPENCODE_CONFIG"] = f"{home}/.config/opencode/opencode.json"
    return env


def _parse_opencode_event(line: str) -> AgentEvent | None:
    """Parse one line of OpenCode's JSONL stdout."""
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    event_type = data.get("type", "")
    if event_type in ("assistant", "message"):
        return AgentEvent(type="assistant", data=data, raw=line)
    elif event_type in ("tool_call", "tool_use"):
        return AgentEvent(type="tool_call", data=data, raw=line)
    elif event_type in ("tool_result", "tool_response"):
        return AgentEvent(type="tool_result", data=data, raw=line)
    elif event_type == "error":
        return AgentEvent(type="error", data=data, raw=line)
    elif event_type in ("done", "complete", "end"):
        return AgentEvent(type="done", data=data, raw=line)
    return AgentEvent(type="assistant", data=data, raw=line)


# File resolvers


def _instruction_file_content(task: Any, config: Any) -> str:
    return task.instruction if hasattr(task, "instruction") else str(task)


def _system_prompt_content(task: Any, config: Any) -> str | None:
    if hasattr(config, "system_prompt") and config.system_prompt:
        return config.system_prompt
    return None


# Spec definition


OPENCODE_SPEC = CLIAgentSpec(
    name="opencode",
    install_check_cmd=["/home/user/.opencode/bin/opencode", "--version"],
    base_command=[
        "opencode",
        "run",
        "--format",
        "json",
        "--dangerously-skip-permissions",
    ],
    mcp_config=MCPConfigSpec(
        method="config_file",
        path_template="{home}/.config/opencode/opencode.json",
    ),
    supports_logprob_proxy=True,
    default_timeout_s=900.0,
    setup=(
        "set -e && "
        "mkdir -p /home/user/.config/opencode /home/user/logs/agent "
        "/home/user/logs/verifier /home/user/task /home/user/workdir && "
        "curl -fsSL https://opencode.ai/install | bash && "
        'export PATH="$HOME/.opencode/bin:$PATH" && '
        "opencode --version"
    ),
    files={
        "/home/user/task/instruction.md": _instruction_file_content,
        "/home/user/task/system.md": _system_prompt_content,
    },
    artifacts={
        "agent_log": ArtifactSpec(
            path="/home/user/logs/agent/opencode.jsonl",
            format="jsonl",
        ),
    },
    env={
        "PATH": "/home/user/.opencode/bin:$PATH",
        "OPENAI_BASE_URL": "{base_url}",
        "OPENAI_API_KEY": "{api_key}",
    },
    build_command=_build_opencode_command,
    build_mcp_config=_build_opencode_mcp_config,
    parse_events=_parse_opencode_event,
    build_env_vars=_build_opencode_env_vars,
)


# Auto-register on import
register_agent(OPENCODE_SPEC)


__all__ = [
    "OPENCODE_SPEC",
]
