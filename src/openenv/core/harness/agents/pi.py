# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pi coding agent adapter.

Pi runs in print mode for non-interactive harness usage::

    pi --no-session --no-context-files --provider <p> --model <m> --thinking off \\
       -p @/home/user/task/instruction.txt 2>&1 | tee /home/user/logs/agent/pi.txt

The provider and model are passed as CLI flags so the spec's ``env`` dict
only needs auth credentials (``HF_TOKEN``, ``OPENAI_API_KEY``, etc.).

Registered on import::

    import openenv.core.harness.agents.pi
    # PI_SPEC is now in the registry
"""

from __future__ import annotations

import json
import shlex
from typing import Any

from . import register_agent
from .base import AgentEvent, ArtifactSpec, CLIAgentSpec, MCPConfigSpec


def _instruction(task: Any, config: Any) -> str:
    return task.instruction if hasattr(task, "instruction") else str(task)


def _system_prompt(task: Any, config: Any) -> str | None:
    if hasattr(config, "system_prompt") and config.system_prompt:
        return config.system_prompt
    return None


def _build_command(
    spec: CLIAgentSpec,
    config: Any,
    task: Any,
    mcp_config_path: str | None,
) -> str:
    home = config.sandbox_home if hasattr(config, "sandbox_home") else "/home/user"
    instruction_file = f"{home}/task/instruction.txt"
    log_file = f"{home}/logs/agent/pi.txt"
    workdir = f"{home}/workdir"

    provider = ""
    if hasattr(config, "provider") and config.provider:
        provider = f" --provider {shlex.quote(config.provider)}"
    model = ""
    if hasattr(config, "model") and config.model:
        model = f" --model {shlex.quote(config.model)}"
    thinking = " --thinking off"
    if hasattr(config, "thinking") and config.thinking:
        thinking = f" --thinking {shlex.quote(config.thinking)}"

    workdir_q = shlex.quote(workdir)
    instruction_q = shlex.quote(instruction_file)
    log_q = shlex.quote(log_file)

    return (
        f"cd {workdir_q} && git init -q 2>/dev/null; "
        f"pi --no-session --no-context-files"
        f"{provider}{model}{thinking}"
        f" -p @{instruction_q}"
        f" 2>&1 | tee {log_q}"
    )


def _build_mcp_config(
    spec: CLIAgentSpec,
    tools: list[Any],
    workdir: str,
) -> str:
    return json.dumps({"mcpServers": {}}, indent=2)


def _parse_events(line: str) -> AgentEvent | None:
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return AgentEvent(type="assistant", data={"text": line}, raw=line)

    event_type = data.get("type", "")
    if event_type in ("assistant", "message", "response"):
        return AgentEvent(type="assistant", data=data, raw=line)
    if event_type in ("tool_call", "tool_use", "function_call"):
        return AgentEvent(type="tool_call", data=data, raw=line)
    if event_type in ("tool_result", "tool_response"):
        return AgentEvent(type="tool_result", data=data, raw=line)
    if event_type in ("thinking", "reasoning"):
        return AgentEvent(type="reasoning", data=data, raw=line)
    if event_type == "error":
        return AgentEvent(type="error", data=data, raw=line)
    if event_type in ("done", "complete", "end"):
        return AgentEvent(type="done", data=data, raw=line)
    return AgentEvent(type="assistant", data=data, raw=line)


PI_SPEC = CLIAgentSpec(
    name="pi",
    install_check_cmd=["pi", "--version"],
    base_command=["pi", "--no-session", "--no-context-files"],
    mcp_config=MCPConfigSpec(
        method="config_file",
        path_template="{workdir}/.mcp.json",
    ),
    default_timeout_s=600.0,
    setup=(
        "set -e && "
        "apt-get update -qq && apt-get install -y -qq curl ca-certificates gnupg && "
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && "
        "apt-get install -y -qq nodejs && "
        "curl -fsSL https://pi.dev/install.sh | sh && "
        "mkdir -p /home/user/logs/agent /home/user/task /home/user/workdir && "
        'export PATH="$HOME/.local/bin:$HOME/.pi/bin:$PATH" && '
        "pi --version"
    ),
    files={
        "/home/user/task/instruction.txt": _instruction,
        "/home/user/task/system.txt": _system_prompt,
    },
    artifacts={
        "agent_log": ArtifactSpec(path="/home/user/logs/agent/pi.txt"),
    },
    env={
        "HF_TOKEN": "{api_key}",
        "OPENAI_API_KEY": "{api_key}",
        "OPENAI_BASE_URL": "{base_url}",
        "PI_SKIP_VERSION_CHECK": "1",
        "PI_TELEMETRY": "0",
    },
    build_command=_build_command,
    build_mcp_config=_build_mcp_config,
    parse_events=_parse_events,
)

register_agent(PI_SPEC)

__all__ = ["PI_SPEC"]
