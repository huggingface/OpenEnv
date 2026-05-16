# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE harness session and session factory.

Integrates ``mini_swe_env`` with the ``CLIAgentDriver`` / ``ResourceSession``
harness infrastructure so that ``build_harness_rollout_func`` works with SWE
tasks out of the box.

Session lifecycle::

    factory = SWESessionFactory(agent="pi", config=..., sandbox_backend=..., ...)
    session = factory.create(task=swe_task)

    # Training loop uses session.initial_messages(), session.next_request(), etc.
    session.wait_for_completion(timeout_s=600)
    vr = session.verify(transcript=[])
    print(vr.env_reward)
    session.close()

The factory handles:
  1. Sandbox creation
  2. Repo staging (git clone + checkout to ``base_commit``)
  3. In-sandbox MCP server deployment (``terminal`` tool via stdio)
  4. Setup command execution
  5. Agent bootstrap + launch
  6. Interception gate rollout registration (when mode="interception_gate")

The session handles:
  - ``verify()`` — runs task verify commands, computes reward
  - ``initial_messages()`` — instruction prompt
  - Interception gate: ``next_request()`` / ``deliver()``
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openenv.core.harness import Message, ResourceSessionFactory, VerifyResult
from openenv.core.harness.agents import get_agent_spec
from openenv.core.harness.agents.cli_driver import (
    CLIAgentDriver,
    CLIAgentSession,
)
from openenv.core.harness.agents.interception_server import InterceptionServer
from openenv.core.harness.sandbox import SandboxBackend, SandboxHandle

from .models import coerce_swe_task, SWETask, validate_swe_task


_log = logging.getLogger(__name__)

# Sandbox filesystem layout (must match sandbox_mcp_server.py).
HOME = "/home/user"
WORKDIR = "/testbed"
REWARD_FILE = f"{HOME}/logs/verifier/reward.txt"
MCP_CONFIG_PATH = f"{HOME}/.swe_mcp_config.json"
MCP_SERVER_PATH = f"{HOME}/.swe_mcp_server.py"
MCP_PORT = 8765
VERIFY_TIMEOUT_S = 300
SETUP_TIMEOUT_S = 600

# Source of the in-sandbox MCP server script.
_SANDBOX_MCP_SERVER_SOURCE = Path(__file__).parent / "server" / "sandbox_mcp_server.py"


@dataclass
class SWEAgentConfig:
    """Minimal config for the CLI agent driver."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    agent_timeout_s: float = 600.0
    sandbox_home: str = HOME
    provider: str = ""
    thinking: str = "off"


@dataclass
class _SWEAgentTask:
    """Internal task shape passed to CLIAgentDriver (not SWETask)."""

    instruction: str = ""
    setup_shell: str | None = None
    upload_files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── SWE Session ────────────────────────────────────────────────────────────


class SWESession(CLIAgentSession):
    """Per-rollout session with SWE-specific verify and reward logic.

    Extends :class:`CLIAgentSession` with:
    - ``verify()`` that runs the task's verify commands and computes reward
    - SWE task metadata
    """

    def __init__(
        self,
        *,
        swe_task: SWETask,
        verify_timeout_s: int = VERIFY_TIMEOUT_S,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._swe_task = swe_task
        self._verify_timeout_s = verify_timeout_s

    @property
    def swe_task(self) -> SWETask:
        return self._swe_task

    def initial_messages(self) -> list[Message]:
        """Return the SWE instruction as the initial prompt."""
        return [{"role": "user", "content": self._swe_task.instruction}]

    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        """Run verify commands in the sandbox and compute reward.

        Reward = passed_commands / total_commands unless
        ``/home/user/logs/verifier/reward.txt`` contains an explicit float.
        """
        passed = 0
        verify_details: list[dict[str, Any]] = []

        for cmd in self._swe_task.verify:
            t0 = time.time()
            try:
                r = self.sandbox.exec(cmd, cwd=WORKDIR, timeout=self._verify_timeout_s)
                detail = {
                    "cmd": cmd,
                    "exit_code": r.exit_code,
                    "stdout_tail": (r.stdout or "")[-2000:],
                    "stderr_tail": (r.stderr or "")[-2000:],
                    "duration_s": round(time.time() - t0, 3),
                }
                if r.exit_code == 0:
                    passed += 1
            except Exception as exc:
                detail = {
                    "cmd": cmd,
                    "exit_code": -1,
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_s": round(time.time() - t0, 3),
                }
            verify_details.append(detail)

        # Reward: explicit reward.txt overrides computed ratio.
        reward = self._read_reward()
        if reward is None and self._swe_task.verify:
            reward = passed / len(self._swe_task.verify)

        return VerifyResult(
            env_reward=reward,
            done=True,
            metrics={
                "verify_passed": passed,
                "verify_total": len(self._swe_task.verify),
                "instance_id": self._swe_task.instance_id,
            },
            artifacts={
                "verify_details": verify_details,
                "task_id": self._swe_task.task_id,
            },
        )

    def _read_reward(self) -> float | None:
        """Read explicit reward override from the sandbox."""
        try:
            raw = self.sandbox.read_text(REWARD_FILE)
            if raw and raw.strip():
                return float(raw.strip())
        except Exception:
            pass
        return None


# ── Tool-call parsing ──────────────────────────────────────────────────────


def parse_terminal_call(text: str) -> dict[str, Any] | None:
    """Parse a terminal tool-call from text.

    Handles multiple formats:
    - ``{"command": "..."}``
    - ``{"final_answer": "..."}``
    - ``terminal(command="...")`` (Python-style)

    Returns parsed arguments dict or None if not a terminal call.
    """
    text = text.strip()
    if not text:
        return None

    # Try direct JSON parse.
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and ("command" in data or "final_answer" in data):
                return data
        except json.JSONDecodeError:
            pass

    # Try extracting JSON from markdown code fences.
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                try:
                    data = json.loads(block)
                    if isinstance(data, dict) and (
                        "command" in data or "final_answer" in data
                    ):
                        return data
                except json.JSONDecodeError:
                    continue

    # Try Python-style: terminal(command="...") or terminal(final_answer="...")
    for key in ("command", "final_answer"):
        prefix = f"terminal({key}="
        if prefix in text:
            idx = text.index(prefix) + len(prefix)
            rest = text[idx:]
            # Try to extract quoted string
            if rest.startswith('"') or rest.startswith("'"):
                quote = rest[0]
                end = rest.find(quote, 1)
                if end > 0:
                    return {key: rest[1:end]}

    return None


# ── SWE Session Factory ───────────────────────────────────────────────────


class SWESessionFactory(ResourceSessionFactory):
    """Creates isolated SWE sessions from ``SWETask`` inputs.

    Compatible with :func:`build_harness_rollout_func`.
    """

    def __init__(
        self,
        *,
        agent: str = "pi",
        config: SWEAgentConfig,
        sandbox_backend: SandboxBackend,
        mode: Literal["black_box", "interception_gate"] = "black_box",
        install_timeout_s: int = 300,
        setup_timeout_s: int = SETUP_TIMEOUT_S,
        verify_timeout_s: int = VERIFY_TIMEOUT_S,
        interception_server: InterceptionServer | None = None,
        interception_base_url: str | None = None,
    ) -> None:
        if mode not in {"black_box", "interception_gate"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        if mode == "interception_gate":
            if interception_server is None:
                raise ValueError(
                    "interception_gate mode requires an InterceptionServer."
                )
            if interception_base_url is None:
                raise ValueError(
                    "interception_gate mode requires interception_base_url."
                )

        self._agent_name = agent
        self._config = config
        self._backend = sandbox_backend
        self._mode = mode
        self._verify_timeout_s = verify_timeout_s
        self._interception_server = interception_server
        self._interception_base_url = interception_base_url

        self._spec = get_agent_spec(agent)
        self._driver = CLIAgentDriver(
            spec=self._spec,
            sandbox_backend=sandbox_backend,
            mode=mode,
            install_timeout_s=install_timeout_s,
            setup_timeout_s=setup_timeout_s,
            interception_server=interception_server,
            interception_base_url=interception_base_url,
        )

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> SWESession:
        """Create one SWE session.

        ``task`` can be an ``SWETask``, a dict matching the SWETask schema,
        or a JSONL row from SWE-bench Lite.
        """
        swe_task = coerce_swe_task(task) if not isinstance(task, SWETask) else task
        validate_swe_task(swe_task)

        sandbox_timeout = int(self._config.agent_timeout_s) + 600
        sandbox = self._backend.create(
            timeout_s=sandbox_timeout,
            metadata={"episode_id": episode_id, "instance_id": swe_task.instance_id}
            if episode_id
            else {"instance_id": swe_task.instance_id},
        )

        try:
            # 1. Stage repo.
            self._stage_repo(sandbox, swe_task)

            # 2. Deploy terminal MCP server (script + config).
            self._deploy_terminal_tool(sandbox, swe_task)

            # 3. Run task setup commands.
            self._run_setup(sandbox, swe_task)

            # 4. Bootstrap agent (install CLI, write instruction, etc.).
            agent_task = self._build_agent_task(swe_task)
            self._driver._bootstrap_sandbox(sandbox, agent_task, self._config)

            # 5. Override MCP config to include terminal tool.
            self._write_terminal_mcp_config(sandbox)

        except Exception as exc:
            _log.error("SWESessionFactory.create: bootstrap failed: %r", exc)
            sandbox.kill()
            raise

        # 6. Handle interception gate rollout registration.
        base_url_override: str | None = None
        interception_rollout_id: str | None = None
        interception_queue: asyncio.Queue | None = None

        if self._mode == "interception_gate":
            assert self._interception_server is not None
            assert self._interception_base_url is not None
            rollout_id = episode_id or f"rollout_{uuid.uuid4().hex[:8]}"
            interception_rollout_id = rollout_id
            interception_queue = self._interception_server.register_rollout(rollout_id)
            base_url_override = (
                f"{self._interception_base_url.rstrip('/')}/rollout/{rollout_id}/v1"
            )

        # 7. Start agent.
        agent_task = self._build_agent_task(swe_task)
        agent_bg = self._driver._start_agent(
            sandbox, agent_task, self._config, base_url_override=base_url_override
        )

        return SWESession(
            swe_task=swe_task,
            verify_timeout_s=self._verify_timeout_s,
            spec=self._spec,
            sandbox=sandbox,
            task=agent_task,
            config=self._config,
            base_url_override=base_url_override,
            agent_bg_job=agent_bg,
            interception_server=self._interception_server,
            interception_rollout_id=interception_rollout_id,
            interception_queue=interception_queue,
        )

    # ── Bootstrap helpers ──────────────────────────────────────────────────

    def _stage_repo(self, sandbox: SandboxHandle, task: SWETask) -> None:
        """Clone the repo and reset to base_commit."""
        sandbox.exec(f"mkdir -p {WORKDIR}", timeout=10)
        clone_url = f"https://github.com/{task.repo}.git"
        r = sandbox.exec(
            f"git clone --quiet {clone_url} {WORKDIR}",
            timeout=SETUP_TIMEOUT_S,
        )
        if r.exit_code != 0:
            raise RuntimeError(
                f"git clone failed (exit {r.exit_code}): {r.stderr[:500]}"
            )
        r = sandbox.exec(
            f"git checkout --quiet {task.base_commit}",
            cwd=WORKDIR,
            timeout=60,
        )
        if r.exit_code != 0:
            raise RuntimeError(
                f"git checkout failed (exit {r.exit_code}): {r.stderr[:500]}"
            )

    def _deploy_terminal_tool(self, sandbox: SandboxHandle, task: SWETask) -> None:
        """Write the MCP server script + config into the sandbox."""
        mcp_source = _SANDBOX_MCP_SERVER_SOURCE.read_text()
        sandbox.write_text(MCP_SERVER_PATH, mcp_source)

        mcp_config = json.dumps(
            {
                "workspace": WORKDIR,
                "verify_commands": list(task.verify),
                "timeout_per_command_s": self._verify_timeout_s,
                "output_limit": 16_000,
                "port": MCP_PORT,
            },
            indent=2,
        )
        sandbox.write_text(MCP_CONFIG_PATH, mcp_config)

        # Ensure log dirs exist.
        sandbox.exec(
            f"mkdir -p {HOME}/logs/verifier {HOME}/logs/agent",
            timeout=10,
        )

    def _run_setup(self, sandbox: SandboxHandle, task: SWETask) -> None:
        """Run task setup commands in the workspace."""
        for cmd in task.setup:
            r = sandbox.exec(cmd, cwd=WORKDIR, timeout=SETUP_TIMEOUT_S)
            if r.exit_code != 0:
                raise RuntimeError(
                    f"Setup command failed (exit {r.exit_code}): "
                    f"{cmd[:120]}\nstderr: {(r.stderr or '')[:500]}"
                )

    def _write_terminal_mcp_config(self, sandbox: SandboxHandle) -> None:
        """Override the agent's MCP config to include the terminal tool.

        Uses stdio transport: the agent launches sandbox_mcp_server.py as a
        subprocess.  This is the most compatible transport across agents.
        """
        mcp_json = json.dumps(
            {
                "mcpServers": {
                    "swe-terminal": {
                        "command": "python3",
                        "args": [MCP_SERVER_PATH, "--stdio"],
                        "env": {"SWE_MCP_CONFIG": MCP_CONFIG_PATH},
                    },
                },
            },
            indent=2,
        )

        # Write to the path the agent spec declares.
        home = (
            self._config.sandbox_home if hasattr(self._config, "sandbox_home") else HOME
        )
        workdir = f"{home}/workdir"

        if (
            self._spec.mcp_config.method == "config_file"
            and self._spec.mcp_config.path_template
        ):
            mcp_path = self._spec.mcp_config.path_template.format(
                workdir=workdir, home=home
            )
            sandbox.write_text(mcp_path, mcp_json)

        # Also write to well-known global paths for fallback discovery.
        for global_path in [
            f"{home}/.mcp.json",
            f"{workdir}/.mcp.json",
        ]:
            try:
                sandbox.write_text(global_path, mcp_json)
            except Exception:
                pass

    def _build_agent_task(self, swe_task: SWETask) -> _SWEAgentTask:
        """Convert SWETask into the shape CLIAgentDriver expects."""
        return _SWEAgentTask(
            instruction=swe_task.instruction,
            setup_shell=None,  # We run setup ourselves before bootstrap.
            metadata={
                "task_id": swe_task.task_id,
                "instance_id": swe_task.instance_id,
                "repo": swe_task.repo,
            },
        )


__all__ = [
    "SWEAgentConfig",
    "SWESession",
    "SWESessionFactory",
    "parse_terminal_call",
]
