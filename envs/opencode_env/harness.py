# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenCode session factory + session backed by CLIAgentDriver."""

from __future__ import annotations

import json
import queue as _queue_mod
import shlex
import uuid
from pathlib import Path
from typing import Any, Literal

from openenv.core.harness import ResourceSessionFactory
from openenv.core.harness.agents.cli_driver import (
    CLIAgentDriver,
    CLIAgentSession,
    Verifier,
    build_interception_rollout_url,
)
from openenv.core.harness.agents.interception_server import InterceptionServer
from openenv.core.harness.agents.opencode import OPENCODE_SPEC
from openenv.core.harness.sandbox import BgJob, SandboxBackend, SandboxHandle

from .config import OpenCodeConfig
from .opencode_runtime import (
    agent_log_path,
    build_env_vars,
    build_opencode_json,
    build_run_cmd,
    opencode_config_path,
)
from .task import OpenCodeTask


# Inside-sandbox transparent proxy paths.
_PROXY_PORT = 7000
_PROXY_TRACE_PATH = "/home/user/logs/agent/proxy_trace.jsonl"
_PROXY_LOG_PATH = "/home/user/logs/agent/proxy.log"
_PROXY_SOURCE_PATH = Path(__file__).parent / "sandbox" / "interception.py"


class OpenCodeSession(CLIAgentSession):
    def __init__(
        self,
        *,
        sandbox: SandboxHandle,
        config: OpenCodeConfig,
        task: OpenCodeTask,
        verifier: Verifier | None = None,
        base_url_override: str | None = None,
        agent_bg_job: BgJob | None = None,
        proxy_trace_path: str | None = None,
        proxy_bg_job: BgJob | None = None,
        interception_server: InterceptionServer | None = None,
        interception_rollout_id: str | None = None,
        interception_queue: _queue_mod.Queue[str | None] | None = None,
    ) -> None:
        super().__init__(
            spec=OPENCODE_SPEC,
            sandbox=sandbox,
            task=task,
            config=config,
            verifier=verifier,
            base_url_override=base_url_override,
            agent_bg_job=agent_bg_job,
            interception_server=interception_server,
            interception_rollout_id=interception_rollout_id,
            interception_queue=interception_queue,
        )
        self._proxy_trace_path = proxy_trace_path
        self._proxy_bg_job = proxy_bg_job

    def fetch_trace(self) -> str:
        return self.sandbox.read_text(agent_log_path(self.config))

    def fetch_proxy_trace(self) -> list[dict[str, Any]]:
        """Return per-turn proxy-captured records (transparent_proxy only)."""
        if self._proxy_trace_path is None:
            return []
        try:
            content = self.sandbox.read_text(self._proxy_trace_path)
        except Exception:
            return []
        records: list[dict[str, Any]] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    def close(self) -> None:
        if self._proxy_bg_job is not None:
            try:
                self._proxy_bg_job.kill()
            except Exception:
                pass
            self._proxy_bg_job = None
        super().close()

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


class OpenCodeSessionFactory(ResourceSessionFactory):
    def __init__(
        self,
        *,
        config: OpenCodeConfig,
        sandbox_backend: SandboxBackend,
        mode: Literal[
            "black_box", "transparent_proxy", "interception_gate"
        ] = "transparent_proxy",
        verifier: Verifier | None = None,
        install_timeout_s: int = 240,
        setup_timeout_s: int = 300,
        interception_server: InterceptionServer | None = None,
        interception_base_url: str | None = None,
    ) -> None:
        if mode not in {"black_box", "transparent_proxy", "interception_gate"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self._config = config
        self._backend = sandbox_backend
        self._mode = mode
        self._verifier = verifier
        driver_mode: Literal["black_box", "interception_gate"] = (
            "black_box" if mode == "transparent_proxy" else mode
        )
        self._driver = CLIAgentDriver(
            spec=OPENCODE_SPEC,
            sandbox_backend=sandbox_backend,
            mode=driver_mode,
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
    ) -> OpenCodeSession:
        import logging

        _log = logging.getLogger(__name__)
        oc_task = OpenCodeTask.coerce(task)
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

        base_url_override: str | None = None
        interception_rollout_id: str | None = None
        interception_queue: _queue_mod.Queue[str | None] | None = None
        proxy_trace_path: str | None = None
        proxy_bg_job: BgJob | None = None

        if self._mode == "interception_gate":
            interception_server = self._driver._interception_server
            if interception_server is None:
                raise RuntimeError(
                    "interception_gate mode requires an InterceptionServer"
                )
            interception_base_url = self._driver._interception_base_url
            if interception_base_url is None:
                raise RuntimeError(
                    "interception_gate mode requires interception_base_url"
                )
            rollout_id = episode_id or f"rollout_{uuid.uuid4().hex[:8]}"
            interception_rollout_id = rollout_id
            interception_queue = interception_server.register_rollout(rollout_id)
            base_url_override = build_interception_rollout_url(
                interception_base_url,
                rollout_id,
            )
        elif self._mode == "transparent_proxy":
            proxy_bg_job, base_url_override, proxy_trace_path = self._start_proxy(
                sandbox
            )

        run_config = self._config
        if base_url_override is not None:
            api_key = self._config.api_key
            if self._mode == "interception_gate":
                assert self._driver._interception_server is not None
                api_key = self._driver._interception_server.secret
            run_config = self._config.model_copy(
                update={
                    "provider": "openai_compatible",
                    "base_url": base_url_override,
                    "api_key": api_key,
                }
            )
        sandbox.write_text(
            opencode_config_path(self._config),
            build_opencode_json(run_config),
        )
        agent_bg_job = self._driver._start_agent(
            sandbox,
            oc_task,
            run_config,
            base_url_override=base_url_override,
        )

        return OpenCodeSession(
            sandbox=sandbox,
            config=run_config,
            task=oc_task,
            verifier=self._verifier,
            base_url_override=base_url_override,
            agent_bg_job=agent_bg_job,
            proxy_trace_path=proxy_trace_path,
            proxy_bg_job=proxy_bg_job,
            interception_server=self._driver._interception_server,
            interception_rollout_id=interception_rollout_id,
            interception_queue=interception_queue,
        )

    def _start_proxy(
        self,
        sandbox: SandboxHandle,
    ) -> tuple[BgJob, str, str]:
        """Start the in-sandbox logprob-capturing proxy."""
        proxy_already_present = sandbox.exists("/home/user/proxy/interception.py")

        if not proxy_already_present:
            self._driver._exec_with_retry(
                sandbox,
                "pip install --quiet 'fastapi>=0.104' 'uvicorn[standard]>=0.24' "
                "'httpx>=0.27' 2>&1 | tail -20",
                timeout=180,
                attempts=3,
                backoff_s=2.0,
                label="proxy deps install",
            )
            sandbox.write_text(
                "/home/user/proxy/interception.py",
                _PROXY_SOURCE_PATH.read_text(),
            )
            sandbox.write_text("/home/user/proxy/__init__.py", "")

        proxy_args = [
            "python",
            "interception.py",
            "--upstream-url",
            self._config.base_url,
            "--trace",
            _PROXY_TRACE_PATH,
            "--port",
            str(_PROXY_PORT),
            "--top-logprobs",
            str(self._config.proxy_top_logprobs),
        ]
        if self._config.proxy_max_tokens_cap is not None:
            proxy_args.extend(
                ["--max-tokens-cap", str(self._config.proxy_max_tokens_cap)]
            )
        if self._config.proxy_disable_thinking:
            proxy_args.append("--disable-thinking")
        if self._config.model:
            proxy_args.extend(["--model-override", self._config.model])

        quoted_proxy_args = " ".join(shlex.quote(arg) for arg in proxy_args)
        proxy_cmd = (
            "cd /home/user/proxy && "
            f"{quoted_proxy_args} "
            f"> {shlex.quote(_PROXY_LOG_PATH)} 2>&1"
        )
        proxy_env = {"OPENCODE_UPSTREAM_API_KEY": self._config.api_key}
        proxy_job = sandbox.start_bg(proxy_cmd, envs=proxy_env)

        import time

        attempts = 120
        interval_s = 0.5
        for _ in range(attempts):
            r = sandbox.exec(
                f"curl -sf http://127.0.0.1:{_PROXY_PORT}/healthz",
                timeout=5,
            )
            if r.exit_code == 0:
                break
            time.sleep(interval_s)
        else:
            log = ""
            try:
                log = sandbox.read_text(_PROXY_LOG_PATH)
            except Exception:
                pass
            proxy_job.kill()
            raise RuntimeError(
                f"proxy did not start within {attempts * interval_s:.0f}s. "
                f"log:\n{log[-2000:]}"
            )

        base_url_override = f"http://127.0.0.1:{_PROXY_PORT}/v1"
        return proxy_job, base_url_override, _PROXY_TRACE_PATH

    def _bootstrap_sandbox(self, sandbox: SandboxHandle, task: OpenCodeTask) -> None:
        self._driver.bootstrap_sandbox(sandbox, task, self._config)


__all__ = [
    "OpenCodeSession",
    "OpenCodeSessionFactory",
    "OpenCodeTask",
    "Verifier",
]
