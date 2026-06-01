# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared CLI agent driver, session, and session factory.

Two modes are supported:

- ``black_box`` — the agent talks directly to the upstream LLM. No logprob
  capture. For eval and demos.
- ``interception_gate`` — the agent's LLM calls are routed to an
  :class:`InterceptionServer` running on the trainer host. The training
  loop owns the forward pass and delivers responses back. For RL training.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue as _queue_mod
import shlex
import time
import uuid
from typing import Any, Callable, Literal

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    Message,
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)
from openenv.core.harness.sandbox import BgJob, SandboxBackend, SandboxHandle

from .base import CLIAgentSpec
from .interception_server import deliver_response, InterceptionServer, ToolHandler


_log = logging.getLogger(__name__)

Verifier = Callable[..., VerifyResult]


def build_interception_rollout_url(base_url: str, rollout_id: str) -> str:
    """Build OpenAI-compatible interception endpoint for one rollout."""
    return f"{base_url.rstrip('/')}/rollout/{rollout_id}/v1"


class _ConfigOverrideView:
    """Read-only attribute view with optional overrides."""

    def __init__(self, base: Any, **overrides: Any) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


class CLIAgentSession(ResourceSession):
    """Per-rollout session wrapping one sandbox with one running agent CLI."""

    def __init__(
        self,
        *,
        spec: CLIAgentSpec,
        sandbox: SandboxHandle,
        task: Any,
        config: Any,
        verifier: Verifier | None = None,
        base_url_override: str | None = None,
        agent_bg_job: BgJob | None = None,
        interception_server: InterceptionServer | None = None,
        interception_rollout_id: str | None = None,
        interception_queue: _queue_mod.Queue[str] | None = None,
    ) -> None:
        self.spec = spec
        self.sandbox = sandbox
        self.task = task
        self.config = config
        self._verifier = verifier
        self._base_url_override = base_url_override
        self._agent_bg_job = agent_bg_job
        self._interception_server = interception_server
        self._interception_rollout_id = interception_rollout_id
        self._interception_queue = interception_queue

    def initial_messages(self) -> list[Message]:
        instruction = (
            self.task.instruction
            if hasattr(self.task, "instruction")
            else str(self.task)
        )
        return [{"role": "user", "content": instruction}]

    def list_tools(self) -> list[Tool]:
        return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(
            error=(
                f"{self.spec.name} session does not expose external tool calls; "
                "the CLI agent owns its own tool loop."
            )
        )

    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        if self._verifier is None:
            return VerifyResult(env_reward=None, done=True)
        return self._verifier(self.sandbox, self.task)

    def close(self) -> None:
        if self._agent_bg_job is not None:
            try:
                self._agent_bg_job.kill()
            except Exception:
                pass
            self._agent_bg_job = None
        if (
            self._interception_server is not None
            and self._interception_rollout_id is not None
        ):
            self._interception_server.unregister_rollout(self._interception_rollout_id)
        self.sandbox.kill()

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        """Block until the agent exits, returning its exit code."""
        if self._agent_bg_job is None:
            raise RuntimeError("Agent not started.")
        default_timeout = (
            self.config.agent_timeout_s
            if hasattr(self.config, "agent_timeout_s")
            else self.spec.default_timeout_s
        )
        budget = timeout_s if timeout_s is not None else default_timeout
        return self._agent_bg_job.wait(timeout=budget)

    def collect_artifacts(self) -> dict[str, Any]:
        """Collect all artifacts declared in ``spec.artifacts`` from the sandbox."""
        result: dict[str, Any] = {}
        if not self.spec.artifacts:
            return result
        for name, artifact_spec in self.spec.artifacts.items():
            try:
                content = self.sandbox.read_text(artifact_spec.path)
                if artifact_spec.format == "json":
                    result[name] = json.loads(content)
                elif artifact_spec.format == "jsonl":
                    records = []
                    for line in content.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            _log.debug(
                                "Skipping non-JSON line in %s: %s",
                                artifact_spec.path,
                                line[:120],
                            )
                    result[name] = records
                else:
                    result[name] = content
            except Exception:
                if not artifact_spec.optional:
                    raise
                _log.debug(
                    "Optional artifact %r (%s) not found, skipping",
                    name,
                    artifact_spec.path,
                )
        return result

    # interception_gate API

    async def next_request(
        self, timeout_s: float | None = None
    ) -> dict[str, Any] | None:
        """Await the next LLM request from the agent (interception_gate only).

        Returns the intercept dict, or ``None`` when the agent has exited.
        """
        if self._interception_queue is None:
            raise RuntimeError(
                "next_request() is only available in interception_gate mode."
            )
        server = self._interception_server
        assert server is not None

        deadline = time.time() + (timeout_s or self.spec.default_timeout_s)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"{self.spec.name} interception_gate: no request within timeout"
                )
            try:
                request_id = await asyncio.to_thread(
                    self._interception_queue.get,
                    timeout=min(remaining, 1.0),
                )
                # None sentinel = agent process exited (sent by /exit endpoint)
                if request_id is None:
                    return None
                intercept = server.get_intercept(request_id)
                if intercept is not None:
                    return intercept
            except _queue_mod.Empty:
                pass

            if self._agent_bg_job is not None:
                try:
                    self._agent_bg_job.wait(timeout=0)
                    return None
                except TimeoutError:
                    pass
            continue

    async def deliver(
        self, intercept: dict[str, Any], response_dict: dict[str, Any]
    ) -> None:
        """Return a trainer-generated response to the waiting agent."""
        await deliver_response(intercept, response_dict)

    def register_tool_handler(
        self,
        tool_name: str,
        handler: ToolHandler,
        *,
        tool_definition: dict[str, Any] | None = None,
    ) -> None:
        """Register a host-side interception tool for this rollout."""
        if self._interception_server is None or self._interception_rollout_id is None:
            raise RuntimeError(
                "register_tool_handler() is only available in interception_gate mode."
            )
        self._interception_server.register_tool_handler(
            self._interception_rollout_id,
            tool_name,
            handler,
            tool_definition=tool_definition,
        )


class CLIAgentDriver:
    """Shared driver for all CLI-based agentic harnesses."""

    def __init__(
        self,
        spec: CLIAgentSpec,
        sandbox_backend: SandboxBackend,
        mode: Literal["black_box", "interception_gate"] = "black_box",
        *,
        install_timeout_s: int = 240,
        setup_timeout_s: int = 300,
        interception_server: InterceptionServer | None = None,
        interception_base_url: str | None = None,
    ) -> None:
        if mode not in {"black_box", "interception_gate"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        if mode == "interception_gate":
            if interception_server is None:
                raise ValueError(
                    "interception_gate mode requires an InterceptionServer instance."
                )
            if interception_base_url is None:
                raise ValueError(
                    "interception_gate mode requires interception_base_url."
                )
        self.spec = spec
        self.sandbox_backend = sandbox_backend
        self.mode = mode
        self._install_timeout_s = install_timeout_s
        self._setup_timeout_s = setup_timeout_s
        self._interception_server = interception_server
        self._interception_base_url = interception_base_url

    def bootstrap_sandbox(self, sandbox: SandboxHandle, task: Any, config: Any) -> None:
        """Public bootstrap hook used by external wrappers.

        Runs readiness checks, optional install, file upload, MCP config write,
        and task setup shell execution.
        """
        self._bootstrap_sandbox(sandbox, task, config)

    def create_session(
        self,
        task: Any,
        config: Any,
        *,
        verifier: Verifier | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> CLIAgentSession:
        timeout_s = (
            config.agent_timeout_s
            if hasattr(config, "agent_timeout_s")
            else self.spec.default_timeout_s
        )
        sandbox_timeout = int(timeout_s) + 300
        sandbox = self.sandbox_backend.create(
            timeout_s=sandbox_timeout,
            metadata={"episode_id": episode_id} if episode_id else None,
        )
        try:
            self._bootstrap_sandbox(sandbox, task, config)
        except Exception as exc:
            _log.error("%s driver: bootstrap failed: %r", self.spec.name, exc)
            sandbox.kill()
            raise

        base_url_override: str | None = None
        interception_rollout_id: str | None = None
        interception_queue: _queue_mod.Queue[str] | None = None

        if self.mode == "interception_gate":
            assert self._interception_server is not None
            assert self._interception_base_url is not None
            rollout_id = episode_id or f"rollout_{uuid.uuid4().hex[:8]}"
            interception_rollout_id = rollout_id
            interception_queue = self._interception_server.register_rollout(rollout_id)
            base_url_override = build_interception_rollout_url(
                self._interception_base_url,
                rollout_id,
            )

        agent_bg_job = self._start_agent(
            sandbox, task, config, base_url_override=base_url_override
        )

        return CLIAgentSession(
            spec=self.spec,
            sandbox=sandbox,
            task=task,
            config=config,
            verifier=verifier,
            base_url_override=base_url_override,
            agent_bg_job=agent_bg_job,
            interception_server=self._interception_server,
            interception_rollout_id=interception_rollout_id,
            interception_queue=interception_queue,
        )

    def _bootstrap_sandbox(
        self, sandbox: SandboxHandle, task: Any, config: Any
    ) -> None:
        self._wait_for_sandbox_ready(sandbox)
        if not self._agent_already_installed(sandbox):
            self._install_agent(sandbox)
        self._ensure_extension_dir(sandbox, config)
        self._upload_files(sandbox, task, config)
        self._write_mcp_config(sandbox, config)
        setup_shell = task.setup_shell if hasattr(task, "setup_shell") else None
        if setup_shell:
            r = sandbox.exec(setup_shell, timeout=self._setup_timeout_s)
            if r.exit_code != 0:
                raise RuntimeError(
                    f"task.setup_shell failed ({r.exit_code}): {r.stderr}"
                )

    def _wait_for_sandbox_ready(
        self, sandbox: SandboxHandle, *, attempts: int = 15, delay_s: float = 1.0
    ) -> None:
        last_err = ""
        for _ in range(attempts):
            try:
                r = sandbox.exec("echo ok", timeout=5)
                if r.exit_code == 0 and "ok" in (r.stdout or ""):
                    return
                last_err = (r.stderr or r.stdout or "").strip() or f"exit={r.exit_code}"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(delay_s)
        raise RuntimeError(
            f"sandbox did not become ready within {attempts * delay_s:.0f}s "
            f"(last error: {last_err})"
        )

    def _agent_already_installed(self, sandbox: SandboxHandle) -> bool:
        cmd = " ".join(shlex.quote(c) for c in self.spec.install_check_cmd)
        try:
            r = sandbox.exec(cmd, timeout=10)
            return r.exit_code == 0
        except Exception:
            return False

    def _install_agent(self, sandbox: SandboxHandle) -> None:
        if self.spec.setup is None:
            raise RuntimeError(
                f"Agent {self.spec.name!r} is not installed and no setup commands provided."
            )
        commands = (
            [self.spec.setup] if isinstance(self.spec.setup, str) else self.spec.setup
        )
        for cmd in commands:
            self._exec_with_retry(
                sandbox,
                cmd,
                timeout=self._install_timeout_s,
                attempts=3,
                backoff_s=3.0,
                label=f"{self.spec.name} install",
            )

    def _resolve_sandbox_home(self, sandbox: SandboxHandle, config: Any) -> str:
        configured = getattr(config, "sandbox_home", None)
        if isinstance(configured, str) and configured.strip():
            return configured
        try:
            result = sandbox.exec('printf %s "$HOME"', timeout=5)
            candidate = (result.stdout or "").strip()
            if result.exit_code == 0 and candidate:
                return candidate
        except Exception:
            pass
        return "/home/user"

    def _ensure_extension_dir(self, sandbox: SandboxHandle, config: Any) -> None:
        template = self.spec.extension_dir_template
        if not template:
            return
        home = self._resolve_sandbox_home(sandbox, config)
        extension_dir = template.format(home=home)
        result = sandbox.exec(f"mkdir -p {shlex.quote(extension_dir)}", timeout=10)
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to create extension dir {extension_dir!r}: {result.stderr}"
            )

    def _upload_files(self, sandbox: SandboxHandle, task: Any, config: Any) -> None:
        if not self.spec.files:
            return
        for path, content_or_fn in self.spec.files.items():
            content = (
                content_or_fn(task, config)
                if callable(content_or_fn)
                else content_or_fn
            )
            if content is not None:
                sandbox.write_text(path, content)
        upload_files = task.upload_files if hasattr(task, "upload_files") else {}
        for path, content in upload_files.items():
            sandbox.write_text(path, content)

    def _write_mcp_config(self, sandbox: SandboxHandle, config: Any) -> None:
        if self.spec.build_mcp_config is None:
            return
        if (
            self.spec.mcp_config.method == "config_file"
            and self.spec.mcp_config.path_template
        ):
            home = (
                config.sandbox_home if hasattr(config, "sandbox_home") else "/home/user"
            )
            workdir = (
                config.workdir
                if hasattr(config, "workdir") and getattr(config, "workdir")
                else f"{home}/workdir"
            )
            mcp_path = self.spec.mcp_config.path_template.format(
                workdir=workdir, home=home
            )
            mcp_content = self.spec.build_mcp_config(self.spec, [], workdir)
            if mcp_content:
                sandbox.write_text(mcp_path, mcp_content)

    def _start_agent(
        self,
        sandbox: SandboxHandle,
        task: Any,
        config: Any,
        *,
        base_url_override: str | None = None,
    ) -> BgJob:
        command_config = config
        if (
            self.mode == "interception_gate"
            and self._interception_server is not None
            and self.spec.name == "pi"
            and base_url_override
        ):
            self._write_pi_models_config(
                sandbox,
                config,
                rollout_url=base_url_override,
                api_key=self._interception_server.secret,
            )
            command_config = _ConfigOverrideView(config, provider="openenv")

        if self.spec.build_command is not None:
            cmd = self.spec.build_command(self.spec, command_config, task, None)
        else:
            cmd = " ".join(shlex.quote(c) for c in self.spec.base_command)
        envs = self._resolve_env_vars(config, base_url_override=base_url_override)
        if self.spec.name == "pi":
            home = self._resolve_sandbox_home(sandbox, config)
            # Make pi config discovery independent of the runtime user's $HOME.
            envs["PI_CODING_AGENT_DIR"] = f"{home}/.pi/agent"
        if self.mode == "interception_gate" and self._interception_server is not None:
            envs["OPENAI_API_KEY"] = self._interception_server.secret
            envs["ANTHROPIC_API_KEY"] = self._interception_server.secret

            # Append an exit notification so the InterceptionServer detects
            # agent exit immediately instead of waiting for the full timeout.
            # The /exit endpoint enqueues a None sentinel on the request queue,
            # causing next_request() to return None.
            if base_url_override:
                exit_url = f"{base_url_override.rstrip('/')}/exit"
                auth_header = (
                    "Authorization: Bearer "
                    f"{self._interception_server.secret}"
                )
                cmd = (
                    f"{{ {cmd} ; }} ; "
                    f"curl -sf -X POST -H {shlex.quote(auth_header)} "
                    f"{shlex.quote(exit_url)} || true"
                )

        return sandbox.start_bg(cmd, envs=envs)

    def _write_pi_models_config(
        self,
        sandbox: SandboxHandle,
        config: Any,
        *,
        rollout_url: str,
        api_key: str,
    ) -> None:
        home = self._resolve_sandbox_home(sandbox, config)
        model = config.model if hasattr(config, "model") else "model"
        content = json.dumps(
            {
                "providers": {
                    "openenv": {
                        "baseUrl": rollout_url,
                        "api": "openai-completions",
                        "apiKey": api_key,
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": False,
                        },
                        "models": [{"id": model, "reasoning": False}],
                    }
                }
            },
            indent=2,
        )
        sandbox.write_text(f"{home}/.pi/agent/models.json", content)

    def _resolve_env_vars(
        self,
        config: Any,
        *,
        base_url_override: str | None = None,
    ) -> dict[str, str]:
        if self.spec.build_env_vars is not None:
            return self.spec.build_env_vars(self.spec, config)
        if not self.spec.env:
            return {}
        base_url = base_url_override or (
            config.base_url if hasattr(config, "base_url") else ""
        )
        api_key = config.api_key if hasattr(config, "api_key") else "intercepted"
        model = config.model if hasattr(config, "model") else ""
        substitutions = {"base_url": base_url, "api_key": api_key, "model": model}
        resolved: dict[str, str] = {}
        for key, value in self.spec.env.items():
            try:
                resolved[key] = value.format(**substitutions)
            except KeyError:
                resolved[key] = value
        return resolved

    def _exec_with_retry(
        self,
        sandbox: SandboxHandle,
        cmd: str,
        *,
        timeout: float,
        attempts: int = 3,
        backoff_s: float = 3.0,
        label: str = "cmd",
    ) -> Any:
        last_stdout = ""
        last_stderr = ""
        last_exit = 0
        for i in range(attempts):
            try:
                r = sandbox.exec(cmd, timeout=timeout)
                if r.exit_code == 0:
                    return r
                last_stdout = r.stdout or ""
                last_stderr = r.stderr or ""
                last_exit = r.exit_code
            except Exception as exc:
                last_stderr = f"{type(exc).__name__}: {exc}"
                last_exit = -1
            if i + 1 < attempts:
                time.sleep(backoff_s * (2**i))
        raise RuntimeError(
            f"{label} failed after {attempts} attempts "
            f"(exit={last_exit}, stderr={last_stderr!r}, "
            f"stdout_tail={last_stdout[-400:]!r})"
        )


class CLIAgentSessionFactory(ResourceSessionFactory):
    def __init__(
        self,
        *,
        spec: CLIAgentSpec,
        config: Any,
        sandbox_backend: SandboxBackend,
        mode: Literal["black_box", "interception_gate"] = "black_box",
        verifier: Verifier | None = None,
        install_timeout_s: int = 240,
        setup_timeout_s: int = 300,
        interception_server: InterceptionServer | None = None,
        interception_base_url: str | None = None,
    ) -> None:
        self._spec = spec
        self._config = config
        self._verifier = verifier
        self._driver = CLIAgentDriver(
            spec=spec,
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
    ) -> CLIAgentSession:
        return self._driver.create_session(
            task=task,
            config=self._config,
            verifier=self._verifier,
            seed=seed,
            episode_id=episode_id,
        )


__all__ = [
    "CLIAgentDriver",
    "CLIAgentSession",
    "CLIAgentSessionFactory",
    "Verifier",
    "build_interception_rollout_url",
]
