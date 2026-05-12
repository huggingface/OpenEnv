# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenCode session factory + session — backed by CLIAgentDriver.

This module exposes :class:`OpenCodeSession` and
:class:`OpenCodeSessionFactory` built on top of the generic
:class:`CLIAgentDriver` / :class:`CLIAgentSession` /
:class:`CLIAgentSessionFactory` from ``openenv.core.harness.agents``.

OpenCode-specific configuration (``opencode.json`` generation, provider
mapping, tool enable/disable) is handled by
:mod:`opencode_env.opencode_runtime` builders wired into the
:data:`OPENCODE_SPEC` via callable hooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from openenv.core.harness import ResourceSessionFactory
from openenv.core.harness.agents.cli_driver import (
    CLIAgentDriver,
    CLIAgentSession,
    Verifier,
)
from openenv.core.harness.agents.opencode import OPENCODE_SPEC
from openenv.core.harness.sandbox import BgJob, SandboxBackend, SandboxHandle

from .config import OpenCodeConfig
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
from .task import OpenCodeTask


# Inside-sandbox proxy paths (Mode B).
_PROXY_PORT = 7000
_PROXY_TRACE_PATH = "/home/user/logs/agent/proxy_trace.jsonl"
_PROXY_LOG_PATH = "/home/user/logs/agent/proxy.log"

_PROXY_SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "openenv"
    / "core"
    / "harness"
    / "sandbox"
    / "interception.py"
)


class OpenCodeSession(CLIAgentSession):
    """One live OpenCode rollout inside a sandbox.

    Extends :class:`CLIAgentSession` with OpenCode-specific convenience
    methods (``fetch_trace``, ``wait_for_completion`` with config-aware
    timeout). Fully backward-compatible with code that used the old
    ``OpenCodeSession`` API.
    """

    def __init__(
        self,
        *,
        sandbox: SandboxHandle,
        config: OpenCodeConfig,
        task: OpenCodeTask,
        verifier: Verifier | None = None,
        base_url_override: str | None = None,
        proxy_trace_path: str | None = None,
        proxy_bg_job: BgJob | None = None,
        agent_bg_job: BgJob | None = None,
    ) -> None:
        super().__init__(
            spec=OPENCODE_SPEC,
            sandbox=sandbox,
            task=task,
            config=config,
            verifier=verifier,
            base_url_override=base_url_override,
            proxy_trace_path=proxy_trace_path,
            proxy_bg_job=proxy_bg_job,
            agent_bg_job=agent_bg_job,
        )

    def fetch_trace(self) -> str:
        """Return the raw ``opencode run`` log (JSONL when ``run_format=json``)."""
        return self.sandbox.read_text(agent_log_path(self.config))

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        """Block until the agent exits, returning its exit code."""
        budget = timeout_s if timeout_s is not None else self.config.agent_timeout_s
        if self._agent_bg_job is None:
            raise RuntimeError("Agent not started; call start_agent() first.")
        return self._agent_bg_job.wait(timeout=budget)

    def start_agent(self) -> None:
        """Launch ``opencode run`` as a background subprocess in the sandbox.

        Provided for backward compatibility — the factory now starts the
        agent during ``create()``, so calling this manually is a no-op
        if the agent is already running.
        """
        if self._agent_bg_job is not None:
            return
        cmd = build_run_cmd(self.config)
        envs = build_env_vars(self.config, base_url_override=self._base_url_override)
        self._agent_bg_job = self.sandbox.start_bg(cmd, envs=envs)


class OpenCodeSessionFactory(ResourceSessionFactory):
    """Produce isolated per-rollout :class:`OpenCodeSession` instances.

    The factory owns sandbox provisioning, opencode install, config injection,
    and (Mode B) proxy startup. Each :meth:`create` call returns a fresh
    sandbox with a running agent.

    Internally delegates to :class:`CLIAgentDriver` for the generic
    sandbox lifecycle (readiness probing, install retry, proxy startup).
    OpenCode-specific config generation uses ``opencode_runtime`` builders.
    """

    def __init__(
        self,
        *,
        config: OpenCodeConfig,
        sandbox_backend: SandboxBackend,
        mode: Literal["black_box", "transparent_proxy"] = "black_box",
        verifier: Verifier | None = None,
        install_timeout_s: int = 240,
        setup_timeout_s: int = 300,
    ) -> None:
        if mode not in {"black_box", "transparent_proxy"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self._config = config
        self._backend = sandbox_backend
        self._mode = mode
        self._verifier = verifier
        self._install_timeout_s = install_timeout_s
        self._setup_timeout_s = setup_timeout_s

        # Build a CLIAgentDriver for the shared lifecycle.
        self._driver = CLIAgentDriver(
            spec=OPENCODE_SPEC,
            sandbox_backend=sandbox_backend,
            mode=mode,
            install_timeout_s=install_timeout_s,
            setup_timeout_s=setup_timeout_s,
            proxy_top_logprobs=config.proxy_top_logprobs,
            proxy_max_tokens_cap=config.proxy_max_tokens_cap,
            proxy_disable_thinking=config.proxy_disable_thinking,
        )

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> OpenCodeSession:
        import logging

        _log = logging.getLogger(__name__)

        oc_task = OpenCodeTask.coerce(task)
        sandbox_timeout = int(self._config.agent_timeout_s) + 300

        _log.info(
            "factory.create: creating sandbox timeout=%ds mode=%s",
            sandbox_timeout,
            self._mode,
        )
        sandbox = self._backend.create(
            timeout_s=sandbox_timeout,
            metadata={"episode_id": episode_id} if episode_id else None,
        )
        sid = getattr(sandbox, "sandbox_id", "?")
        _log.info("factory.create: sandbox=%s — bootstrapping…", sid)

        try:
            self._bootstrap_sandbox(sandbox, oc_task)
        except Exception as exc:
            _log.error("factory.create: bootstrap failed: %r", exc)
            sandbox.kill()
            raise

        base_url_override: str | None = None
        proxy_trace_path: str | None = None
        proxy_bg_job: BgJob | None = None
        if self._mode == "transparent_proxy":
            _log.info(
                "factory.create: starting interception proxy on :%d → %s",
                _PROXY_PORT,
                self._config.base_url,
            )
            proxy_bg_job, base_url_override, proxy_trace_path = (
                self._driver._start_proxy(
                    sandbox,
                    base_url=self._config.base_url,
                    api_key=self._config.api_key,
                    model=self._config.model,
                )
            )
            _log.info("factory.create: proxy up at %s", base_url_override)
            # Rewrite opencode.json so opencode points at the proxy.
            proxy_cfg = OpenCodeConfig(
                **{
                    **self._config.model_dump(),
                    "provider": "openai_compatible",
                    "base_url": base_url_override,
                }
            )
            sandbox.write_text(
                opencode_config_path(self._config),
                build_opencode_json(proxy_cfg),
            )

        session = OpenCodeSession(
            sandbox=sandbox,
            config=self._config,
            task=oc_task,
            verifier=self._verifier,
            base_url_override=base_url_override,
            proxy_trace_path=proxy_trace_path,
            proxy_bg_job=proxy_bg_job,
        )
        session.start_agent()
        return session

    # ------------------------------------------------------------------
    # Bootstrap — delegates to CLIAgentDriver utilities
    # ------------------------------------------------------------------

    def _bootstrap_sandbox(
        self,
        sandbox: SandboxHandle,
        task: OpenCodeTask,
    ) -> None:
        """Install opencode, write config + task files, run optional setup."""

        # Stage 1: wait for the sandbox to be responsive.
        self._driver._wait_for_sandbox_ready(sandbox)

        # Stage 2: install opencode (skipped if pre-baked).
        if not self._driver._agent_already_installed(sandbox):
            self._driver._exec_with_retry(
                sandbox,
                build_install_cmd(self._config),
                timeout=self._install_timeout_s,
                attempts=3,
                backoff_s=3.0,
                label="opencode install",
            )

        # Stage 3: write opencode.json + task files.
        sandbox.write_text(
            opencode_config_path(self._config),
            build_opencode_json(self._config),
        )
        sandbox.write_text(instruction_path(self._config), task.instruction)

        if self._config.system_prompt:
            sandbox.write_text(
                system_prompt_path(self._config),
                self._config.system_prompt,
            )

        for remote_path, content in task.upload_files.items():
            sandbox.write_text(remote_path, content)

        # Stage 4: extra setup
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

    def _start_proxy(
        self,
        sandbox: SandboxHandle,
    ) -> tuple[BgJob, str, str]:
        """Start proxy — delegates to driver."""
        return self._driver._start_proxy(
            sandbox,
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            model=self._config.model,
        )


__all__ = [
    "OpenCodeSession",
    "OpenCodeSessionFactory",
    "OpenCodeTask",
    "Verifier",
]
