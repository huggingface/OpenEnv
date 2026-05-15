# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coding-agent session factory + session — backed by CLIAgentDriver."""

from __future__ import annotations

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
from .opencode_runtime import (
    agent_log_path,
    build_env_vars,
    build_install_cmd,
    build_opencode_json,
    build_run_cmd,
    instruction_path,
    opencode_config_path,
    system_prompt_path,
)
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
        self._mode = mode
        self._verifier = verifier
        self._install_timeout_s = install_timeout_s
        self._setup_timeout_s = setup_timeout_s
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
        session = CodingAgentSession(
            sandbox=sandbox,
            config=self._config,
            task=oc_task,
            verifier=self._verifier,
        )
        session.start_agent()
        return session

    def _bootstrap_sandbox(self, sandbox: SandboxHandle, task: CodingAgentTask) -> None:
        self._driver._wait_for_sandbox_ready(sandbox)
        if not self._driver._agent_already_installed(sandbox):
            self._driver._exec_with_retry(
                sandbox,
                build_install_cmd(self._config),
                timeout=self._install_timeout_s,
                attempts=3,
                backoff_s=3.0,
                label="opencode install",
            )
        sandbox.write_text(
            opencode_config_path(self._config), build_opencode_json(self._config)
        )
        sandbox.write_text(instruction_path(self._config), task.instruction)
        if self._config.system_prompt:
            sandbox.write_text(
                system_prompt_path(self._config), self._config.system_prompt
            )
        for remote_path, content in task.upload_files.items():
            sandbox.write_text(remote_path, content)
        if self._config.extra_setup_shell:
            self._driver._exec_with_retry(
                sandbox,
                self._config.extra_setup_shell,
                timeout=self._setup_timeout_s,
                attempts=2,
                backoff_s=2.0,
                label="extra_setup_shell",
            )
        if task.setup_shell:
            r = sandbox.exec(task.setup_shell, timeout=self._setup_timeout_s)
            if r.exit_code != 0:
                raise RuntimeError(
                    f"task.setup_shell failed ({r.exit_code}): {r.stderr}"
                )


__all__ = [
    "CodingAgentSession",
    "CodingAgentSessionFactory",
    "CodingAgentTask",
    "Verifier",
]
