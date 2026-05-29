# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF Sandbox-backed single-tool coding environment inspired by Terminus."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation

try:
    from .hf_sandbox import HFSandbox
    from .local_sandbox import LocalSandbox
    from ..models import CommandResult, TerminusState
except ImportError:  # pragma: no cover
    from models import CommandResult, TerminusState
    from server.hf_sandbox import HFSandbox
    from server.local_sandbox import LocalSandbox


REWARD_FILE = "/home/user/logs/verifier/reward.txt"


class TerminusEnvironment(MCPEnvironment):
    """Single-tool terminal environment with one sandbox per episode."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._sandbox: Optional[Any] = None
        self._state = TerminusState(episode_id=str(uuid4()), step_count=0)

        mcp = FastMCP("terminus_env")

        @mcp.tool
        def terminal(command: str = "", final_answer: str = "") -> str:
            """Run a shell command or submit a final answer inside the sandbox.

            Args:
                command: Shell command to execute in the episode sandbox.
                final_answer: Optional answer string. When provided, stored
                    as the final answer and any reset-time verify commands run.

            Returns:
                Command output, or final-answer verification summary.
            """
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            if final_answer:
                self._state.submitted_answer = final_answer
                if not self._state.verify_commands:
                    return f"Answer submitted: {final_answer}"
                summary = self._run_verify_commands()
                return (
                    f"Answer submitted: {final_answer}\n"
                    f"Verification: {summary['passed']}/{summary['total']} passed; "
                    f"reward={summary['reward']}"
                )
            if not command.strip():
                return "Error: command or final_answer is required."
            result = self._run_shell_command(command)
            self._state.commands.append(result)
            return result.output

        super().__init__(mcp)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Create a fresh sandbox and run optional setup commands."""
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

        self._state = TerminusState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        backend = str(
            kwargs.get("sandbox_backend")
            or os.getenv("TERMINUS_SANDBOX_BACKEND", "hf")
        ).lower()
        sandbox_label = (
            "HF sandbox"
            if backend in {"hf", "hf-sandbox", "huggingface"}
            else f"{backend} sandbox"
        )
        try:
            self._sandbox = _create_sandbox(kwargs)
        except Exception as exc:  # noqa: BLE001
            return Observation(
                done=True,
                reward=None,
                metadata={
                    "status": "error",
                    "error": (
                        f"failed to create {sandbox_label}: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                },
            )

        self._state.sandbox_id = self._sandbox.sandbox_id
        setup_commands = _coerce_commands(
            kwargs.get("setup", kwargs.get("setup_scripts", []))
        )
        verify_commands = _coerce_commands(
            kwargs.get("verify", kwargs.get("verify_scripts", []))
        )
        self._state.verify_commands = verify_commands

        self._sandbox.run_shell("mkdir -p /home/user/logs/verifier")
        if setup_commands:
            setup_results = self._run_shell_commands(setup_commands)
            self._state.setup_results = setup_results
            failed = [result for result in setup_results if not result.success]
            if failed:
                return Observation(
                    done=True,
                    reward=None,
                    metadata={
                        "status": "error",
                        "sandbox_id": self._state.sandbox_id,
                        "message": "Setup command failed.",
                        "setup_results": [
                            result.model_dump() for result in setup_results
                        ],
                    },
                )

        msg = "Terminus environment ready. Use terminal(command=...) to work."
        if setup_commands:
            msg += f" Setup commands run: {len(setup_commands)}."
        if verify_commands:
            msg += f" Verify commands registered: {len(verify_commands)}."
        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "sandbox_id": self._state.sandbox_id,
                "message": msg,
                "setup_results": [
                    result.model_dump() for result in self._state.setup_results
                ],
                "verify_commands": verify_commands,
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=None,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use ListToolsAction or CallToolAction for MCP interactions."
                )
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        if self._state.submitted_answer is not None and self._state.last_reward is not None:
            obs.done = True
            obs.reward = self._state.last_reward
        elif obs.reward is None:
            obs.reward = 0.0
        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        if self._state.submitted_answer is not None and self._state.last_reward is not None:
            obs.done = True
            obs.reward = self._state.last_reward
        elif obs.reward is None:
            obs.reward = 0.0
        return obs

    @property
    def state(self) -> TerminusState:
        return self._state

    def close(self) -> None:
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

    def _run_shell_commands(self, commands: Iterable[str]) -> list[CommandResult]:
        return [self._run_shell_command(command) for command in commands]

    def _run_shell_command(self, command: str) -> CommandResult:
        result = self._sandbox.run_shell(command)
        output = _format_for_llm(result)
        return CommandResult(
            command=command,
            output=output,
            error=result.error,
            success=result.success,
        )

    def _run_verify_commands(self) -> dict[str, Any]:
        if not self._sandbox:
            return {"passed": 0, "total": 0, "reward": None}

        self._sandbox.run_shell("mkdir -p /home/user/logs/verifier")
        verify_results = self._run_shell_commands(self._state.verify_commands)
        self._state.verify_results = verify_results
        passed = sum(1 for result in verify_results if result.success)
        total = len(verify_results)
        reward = _read_reward_override(self._sandbox)
        if reward is None and total:
            reward = passed / total
        self._state.last_reward = reward
        return {"passed": passed, "total": total, "reward": reward}


def _coerce_commands(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def _create_sandbox(kwargs: dict[str, Any]) -> Any:
    backend = str(
        kwargs.get("sandbox_backend")
        or os.getenv("TERMINUS_SANDBOX_BACKEND", "hf")
    ).lower()
    if backend in {"local", "bwrap", "process"}:
        return LocalSandbox(root=kwargs.get("sandbox_root"))
    if backend not in {"hf", "hf-sandbox", "huggingface"}:
        raise ValueError(f"unknown sandbox backend: {backend}")
    return HFSandbox(
        image=kwargs.get("sandbox_image") or kwargs.get("hf_sandbox_image"),
        flavor=kwargs.get("sandbox_flavor") or kwargs.get("hf_sandbox_flavor"),
        timeout=kwargs.get("sandbox_timeout") or kwargs.get("hf_sandbox_timeout"),
        forward_hf_token=kwargs.get("forward_hf_token"),
    )


def _format_for_llm(result) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout.strip())
    if result.stderr:
        parts.append(result.stderr.strip())
    if result.error:
        parts.append(f"ERROR:\n{result.error}")
    return "\n".join(parts) if parts else "(no output)"


def _read_reward_override(sandbox: Any) -> Optional[float]:
    result = sandbox.run_shell(f"cat {REWARD_FILE} 2>/dev/null || true")
    raw = (result.stdout or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
