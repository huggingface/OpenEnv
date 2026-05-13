# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Agent spec and event protocols for CLI-based agentic harnesses.

Defines the declarative :class:`CLIAgentSpec` data model that captures
*everything* a CLI harness needs — install commands, file uploads, MCP
config format, environment variables, artifacts to collect, and three
small callables (command builder, MCP config builder, event parser).

The :class:`CLIAgentDriver` reads these fields mechanically without knowing
anything about the specific agent. Adding a new agent is filling in a
dataclass, not writing driver code.

Pattern borrowed from `verifiers <https://github.com/PrimeIntellect-ai/verifiers>`_
(Prime Intellect), where OpenCode, MiniSWEAgent, Pi, and RLM all express
their differences through constructor data passed to ``CLIHarness.__init__()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol


# MCP config injection


@dataclass(frozen=True)
class MCPConfigSpec:
    """How a harness discovers MCP tools.

    ``method`` controls how the driver injects MCP server configuration:

    - ``"config_file"`` — write a JSON file at ``path_template`` (e.g.
      ``"{workdir}/mcp.json"``).  The template receives ``{workdir}``
      and ``{home}`` substitutions at runtime.
    - ``"cli_flags"`` — the driver passes MCP configuration via CLI
      flags built by :attr:`CLIAgentSpec.build_command`.
    - ``"settings_file"`` — write into a global settings file (e.g.
      e.g. ``~/.config/agent/settings.json``).
    """

    method: Literal["config_file", "cli_flags", "settings_file"]
    path_template: str | None = None


# Artifacts


@dataclass(frozen=True)
class ArtifactSpec:
    """Declares a file to collect from the sandbox after the agent exits.

    The driver iterates :attr:`CLIAgentSpec.artifacts` and calls
    ``sandbox.read_text(spec.path)`` for each entry. No per-agent collection
    methods needed — the spec declares *what* to collect, the driver collects
    it.
    """

    path: str
    format: Literal["text", "json", "jsonl"] = "text"
    optional: bool = True


# Agent events (normalized across harnesses)


@dataclass
class AgentEvent:
    """Normalized event from any CLI harness's stdout.

    The :attr:`CLIAgentSpec.parse_events` callable converts raw JSONL lines
    into these events so the driver can log and observe the agent's progress
    without knowing which agent is running.
    """

    type: Literal[
        "assistant",
        "tool_call",
        "tool_result",
        "reasoning",
        "error",
        "done",
    ]
    data: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


# Task protocol


class AgentTask(Protocol):
    """Minimal interface a task must satisfy for the CLI agent driver."""

    @property
    def instruction(self) -> str: ...

    @property
    def setup_shell(self) -> str | None: ...

    @property
    def upload_files(self) -> dict[str, str]: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


# Agent config protocol


class AgentConfig(Protocol):
    """Minimal interface a config must satisfy for the CLI agent driver.

    This is intentionally thin — concrete configs like :class:`OpenCodeConfig`
    carry much more, but the generic driver only accesses these.
    """

    @property
    def base_url(self) -> str: ...

    @property
    def api_key(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def agent_timeout_s(self) -> float: ...


# CLIAgentSpec — the core declarative data model


@dataclass
class CLIAgentSpec:
    """Declarative specification for a CLI-based agentic harness.

    Following the pattern established by verifiers' ``CLIHarness`` (Prime
    Intellect), as much per-agent knowledge as possible is expressed as
    *data* rather than imperative code. The :class:`CLIAgentDriver`
    iterates these fields mechanically — it never needs to know what
    ``"pi"`` or ``"claude-code"`` means.

    Three callables cover the remaining agent-specific logic that can't
    be expressed as pure data:

    - :attr:`build_command` — constructs the CLI argv
    - :attr:`build_mcp_config` — serializes MCP server configuration
    - :attr:`parse_events` — converts raw stdout lines to :class:`AgentEvent`

    Everything else — file uploads, env vars, install scripts, artifact
    collection — is pure data.
    """

    name: str
    """Unique identifier: ``"opencode"``, ``"claude-code"``, ``"codex"``, etc."""

    install_check_cmd: list[str]
    """Command to probe whether the agent is already installed.

    Example: ``["claude", "--version"]``
    """

    base_command: list[str]
    """Base CLI invocation (before task-specific flags).

    Example: ``["claude", "--print", "--output-format", "stream-json"]``
    """

    mcp_config: MCPConfigSpec
    """How MCP tool configuration is injected."""

    supports_logprob_proxy: bool = True
    """Whether this agent can be routed through the interception proxy."""

    default_timeout_s: float = 600.0
    """Default per-rollout timeout in seconds."""

    setup: str | list[str] | None = None
    """Shell command(s) to install the agent CLI inside the sandbox.

    Run once after the sandbox is created, before any files are written.
    Skipped when ``install_check_cmd`` succeeds (pre-baked template).
    Can be a single string or a list of strings executed in order.
    """

    files: dict[str, str | Callable] | None = None
    """Files to upload into the sandbox before the agent starts.

    Keys are absolute sandbox paths. Values are either literal strings or
    callables ``(task, config) -> str`` resolved at rollout time.
    """

    artifacts: dict[str, ArtifactSpec] | None = None
    """Files to collect from the sandbox after the agent exits.

    The driver iterates this dict and calls ``sandbox.read_text(spec.path)``
    for each entry.
    """

    env: dict[str, str] | None = None
    """Environment variables for the agent process.

    Values can contain ``{model}``, ``{base_url}``, ``{api_key}`` placeholders
    resolved from the rollout config at runtime.
    """

    build_command: Callable[..., str] | None = None
    """``(spec, config, task, mcp_config_path) -> str``

    Build the full shell command line for launching the agent. Returns a
    string (not a list) because sandbox ``start_bg`` / ``exec`` take shell
    strings.
    """

    build_mcp_config: Callable[..., str] | None = None
    """``(spec, tools, workdir) -> str``

    Serialize MCP server configuration in the format the agent expects.
    Returns the file content (for ``config_file``/``settings_file`` methods)
    or empty string (for ``cli_flags``, where the command builder handles it).
    """

    parse_events: Callable[[str], AgentEvent | None] | None = None
    """``(line: str) -> AgentEvent | None``

    Parse one line of the agent's stdout into a normalized event.
    Return ``None`` for lines that are not parseable events.
    """

    build_env_vars: Callable[..., dict[str, str]] | None = None
    """``(spec, config) -> dict[str, str]``

    Optional override for env var construction. When provided, this is
    called *instead of* resolving placeholders in :attr:`env`. Prefer
    the declarative :attr:`env` dict for new agents.
    """


__all__ = [
    "AgentConfig",
    "AgentEvent",
    "AgentTask",
    "ArtifactSpec",
    "CLIAgentSpec",
    "MCPConfigSpec",
]
