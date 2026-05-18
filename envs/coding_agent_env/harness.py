# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coding-agent session factory + session — backed by CLIAgentDriver."""

from __future__ import annotations

import queue as _queue_mod
import uuid
from typing import Any, Literal

from openenv.core.harness import ResourceSessionFactory
from openenv.core.harness.agents.cli_driver import (
    CLIAgentDriver,
    CLIAgentSession,
    Verifier,
)
from openenv.core.harness.agents.interception_server import InterceptionServer
from openenv.core.harness.agents.opencode import OPENCODE_SPEC
from openenv.core.harness.sandbox import SandboxBackend, SandboxHandle

from .config import CodingAgentConfig
from .opencode_runtime import agent_log_path, build_env_vars, build_run_cmd
from .task import CodingAgentTask


class CodingAgentSession(CLIAgentSession):
    def __init__(
        self,
        *,
        sandbox: SandboxHandle,
        config: CodingAgentConfig,
        task: CodingAgentTask,
        verifier: Verifier | None = None,
        base_url_override: str | None = None,
    ) -> None:
        super().__init__(
            spec=OPENCODE_SPEC,
            sandbox=sandbox,
            task=task,
            config=config,
            verifier=verifier,
            base_url_override=base_url_override,
        )

    def fetch_trace(self) -> str:
        return self.sandbox.read_text(agent_log_path(self.config))

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        budget = timeout_s if timeout_s is not None else self.config.agent_timeout_s
        if self._agent_bg_job is None:
            raise RuntimeError("Agent not started.")
        return self._agent_bg_job.wait(timeout=budget)

    def start_agent(self) -> None:
        if self._agent_bg_job is not None:
            return
        cmd = build_run_cmd(self.config)
        envs = build_env_vars(self.config, base_url_override=self._base_url_override)
        self._agent_bg_job = self.sandbox.start_bg(cmd, envs=envs)


class CodingAgentSessionFactory(ResourceSessionFactory):
    def __init__(
        self,
        *,
        config: CodingAgentConfig,
        sandbox_backend: SandboxBackend,
        mode: Literal["black_box", "interception_gate"] = "black_box",
        verifier: Verifier | None = None,
        install_timeout_s: int = 240,
        setup_timeout_s: int = 300,
        interception_server: InterceptionServer | None = None,
        interception_base_url: str | None = None,
    ) -> None:
        if mode not in {"black_box", "interception_gate"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self._config = config
        self._backend = sandbox_backend
        self._verifier = verifier
        self._driver = CLIAgentDriver(
            spec=OPENCODE_SPEC,
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
    ) -> CodingAgentSession:
        import logging

        _log = logging.getLogger(__name__)
        oc_task = CodingAgentTask.coerce(task)
        setup_parts: list[str] = []
        if self._config.extra_setup_shell:
            setup_parts.append(self._config.extra_setup_shell)
        if oc_task.setup_shell:
            setup_parts.append(oc_task.setup_shell)
        if setup_parts:
            oc_task = oc_task.model_copy(
                update={"setup_shell": "set -e\n" + "\n".join(setup_parts)}
            )

        sandbox_timeout = int(self._config.agent_timeout_s) + 300
        sandbox = self._backend.create(
            timeout_s=sandbox_timeout,
            metadata={"episode_id": episode_id} if episode_id else None,
        )
        try:
            self._bootstrap_sandbox(sandbox, oc_task)
        except Exception as exc:
            _log.error("factory.create: bootstrap failed: %r", exc)
            sandbox.kill()
            raise

        # Wire up interception_gate if the driver is configured for it
        base_url_override: str | None = None
        interception_rollout_id: str | None = None
        interception_queue: _queue_mod.Queue[str] | None = None

        if self._driver.mode == "interception_gate":
            assert self._driver._interception_server is not None
            assert self._driver._interception_base_url is not None
            rollout_id = episode_id or f"rollout_{uuid.uuid4().hex[:8]}"
            interception_rollout_id = rollout_id
            interception_queue = self._driver._interception_server.register_rollout(
                rollout_id
            )
            base_url_override = (
                f"{self._driver._interception_base_url.rstrip('/')}"
                f"/rollout/{rollout_id}/v1"
            )

        session = CodingAgentSession(
            sandbox=sandbox,
            config=self._config,
            task=oc_task,
            verifier=self._verifier,
            base_url_override=base_url_override,
        )
        # Pass interception fields to the parent CLIAgentSession
        session._interception_server = self._driver._interception_server
        session._interception_rollout_id = interception_rollout_id
        session._interception_queue = interception_queue

        session.start_agent()
        return session

    def _bootstrap_sandbox(self, sandbox: SandboxHandle, task: CodingAgentTask) -> None:
        self._driver.bootstrap_sandbox(sandbox, task, self._config)


__all__ = [
    "CodingAgentSession",
    "CodingAgentSessionFactory",
    "CodingAgentTask",
    "Verifier",
]
