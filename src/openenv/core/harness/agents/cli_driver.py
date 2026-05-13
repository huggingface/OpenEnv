# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared CLI agent driver, session, and session factory.

The :class:`CLIAgentDriver` factors out the common 70% of CLI harness
lifecycle — sandbox creation, MCP config injection, interception proxy
setup, subprocess management, and result collection.

It is **fully generic**: it reads the :class:`CLIAgentSpec`'s declarative
data fields and executes them mechanically. No per-agent code lives here.

The :class:`CLIAgentSession` implements :class:`ResourceSession` and
the :class:`CLIAgentSessionFactory` implements :class:`ResourceSessionFactory`,
so the CLI agent driver integrates seamlessly with the existing harness
runtime from PR #603.
"""

from __future__ import annotations

import json
import logging
import shlex
import time
from pathlib import Path
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


_log = logging.getLogger(__name__)

# Interception proxy defaults
_PROXY_PORT = 7000
_PROXY_TRACE_PATH = "/home/user/logs/agent/proxy_trace.jsonl"
_PROXY_LOG_PATH = "/home/user/logs/agent/proxy.log"

# Where the proxy source lives on disk. Uploaded into sandboxes that don't
# already have it baked in.
_PROXY_SOURCE_PATH = Path(__file__).resolve().parents[1] / "sandbox" / "interception.py"

# Verifier type — same as opencode_env's Verifier alias
Verifier = Callable[..., VerifyResult]


# CLIAgentSession


class CLIAgentSession(ResourceSession):
    """Per-rollout session wrapping one sandbox with one running agent CLI.

    The session is created already-running: :meth:`CLIAgentSessionFactory.create`
    launches the agent before returning. Typical usage::

        session = factory.create(task)
        session.wait_for_completion()
        result = session.verify([])
        session.close()
    """

    def __init__(
        self,
        *,
        spec: CLIAgentSpec,
        sandbox: SandboxHandle,
        task: Any,
        config: Any,
        verifier: Verifier | None = None,
        base_url_override: str | None = None,
        proxy_trace_path: str | None = None,
        proxy_bg_job: BgJob | None = None,
        agent_bg_job: BgJob | None = None,
    ) -> None:
        self.spec = spec
        self.sandbox = sandbox
        self.task = task
        self.config = config
        self._verifier = verifier
        self._base_url_override = base_url_override
        self._proxy_trace_path = proxy_trace_path
        self._proxy_bg_job = proxy_bg_job
        self._agent_bg_job = agent_bg_job

    # ResourceSession contract

    def initial_messages(self) -> list[Message]:
        instruction = (
            self.task.instruction
            if hasattr(self.task, "instruction")
            else str(self.task)
        )
        return [{"role": "user", "content": instruction}]

    def list_tools(self) -> list[Tool]:
        # CLI agents own their own tool loop — none are exposed to the harness.
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
        if self._proxy_bg_job is not None:
            try:
                self._proxy_bg_job.kill()
            except Exception:
                pass
            self._proxy_bg_job = None
        self.sandbox.kill()

    # CLI-agent-specific API

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        """Block until the agent exits, returning its exit code."""
        budget = timeout_s if timeout_s is not None else self.spec.default_timeout_s
        if hasattr(self.config, "agent_timeout_s"):
            budget = timeout_s if timeout_s is not None else self.config.agent_timeout_s
        if self._agent_bg_job is None:
            raise RuntimeError("Agent not started.")
        return self._agent_bg_job.wait(timeout=budget)

    def collect_artifacts(self) -> dict[str, Any]:
        """Collect all artifacts declared in ``spec.artifacts`` from the sandbox.

        Returns a dict keyed by artifact name. Missing optional artifacts are
        silently skipped.
        """
        result: dict[str, Any] = {}
        if not self.spec.artifacts:
            return result
        for name, artifact_spec in self.spec.artifacts.items():
            try:
                content = self.sandbox.read_text(artifact_spec.path)
                if artifact_spec.format == "json":
                    result[name] = json.loads(content)
                elif artifact_spec.format == "jsonl":
                    # Parse valid JSON lines, skip non-JSON preamble
                    # (e.g. opencode emits database migration messages
                    # before the first JSON event).
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

    def fetch_proxy_trace(self) -> list[dict[str, Any]]:
        """Return per-turn proxy-captured records (transparent_proxy mode only).

        Each entry has ``request``, ``response``, ``completion_tokens``,
        ``completion_token_ids``, ``per_token_logps``, ``finish_reason``,
        and ``latency_s``. Returns ``[]`` in black_box mode.
        """
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


# CLIAgentDriver — shared lifecycle


class CLIAgentDriver:
    """Shared driver for all CLI-based agentic harnesses.

    Implements the common lifecycle:

    1. Create sandbox (via :class:`SandboxBackend`)
    2. Wait for sandbox ready (``echo ok`` probe)
    3. Install agent CLI — run ``spec.setup`` commands (skipped if
       ``spec.install_check_cmd`` succeeds, i.e. pre-baked template)
    4. Upload ``spec.files`` into the sandbox
    5. Write MCP config (via ``spec.build_mcp_config``)
    6. Set environment variables from ``spec.env`` (with placeholder
       resolution)
    7. Optionally start interception proxy (transparent_proxy mode)
    8. Build CLI command (via ``spec.build_command``)
    9. Launch agent as bg process
    10. Return a :class:`CLIAgentSession`
    """

    def __init__(
        self,
        spec: CLIAgentSpec,
        sandbox_backend: SandboxBackend,
        mode: Literal["black_box", "transparent_proxy"] = "black_box",
        *,
        install_timeout_s: int = 240,
        setup_timeout_s: int = 300,
        proxy_top_logprobs: int = 5,
        proxy_max_tokens_cap: int | None = 16384,
        proxy_disable_thinking: bool = False,
    ) -> None:
        if mode not in {"black_box", "transparent_proxy"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self.spec = spec
        self.sandbox_backend = sandbox_backend
        self.mode = mode
        self._install_timeout_s = install_timeout_s
        self._setup_timeout_s = setup_timeout_s
        self._proxy_top_logprobs = proxy_top_logprobs
        self._proxy_max_tokens_cap = proxy_max_tokens_cap
        self._proxy_disable_thinking = proxy_disable_thinking

    def create_session(
        self,
        task: Any,
        config: Any,
        *,
        verifier: Verifier | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> CLIAgentSession:
        """Create a fully bootstrapped session with a running agent.

        This is the main entry point. It:
        1. Creates a sandbox
        2. Bootstraps it (install agent, upload files, write MCP config)
        3. Optionally starts the interception proxy
        4. Launches the agent subprocess
        5. Returns a ready-to-use :class:`CLIAgentSession`
        """
        timeout_s = (
            config.agent_timeout_s
            if hasattr(config, "agent_timeout_s")
            else self.spec.default_timeout_s
        )
        sandbox_timeout = int(timeout_s) + 300

        _log.info(
            "%s driver: creating sandbox timeout=%ds mode=%s",
            self.spec.name,
            sandbox_timeout,
            self.mode,
        )
        sandbox = self.sandbox_backend.create(
            timeout_s=sandbox_timeout,
            metadata={"episode_id": episode_id} if episode_id else None,
        )
        sid = getattr(sandbox, "sandbox_id", "?")
        _log.info("%s driver: sandbox=%s — bootstrapping…", self.spec.name, sid)

        try:
            self._bootstrap_sandbox(sandbox, task, config)
        except Exception as exc:
            _log.error("%s driver: bootstrap failed: %r", self.spec.name, exc)
            sandbox.kill()
            raise

        base_url_override: str | None = None
        proxy_trace_path: str | None = None
        proxy_bg_job: BgJob | None = None

        if self.mode == "transparent_proxy":
            base_url = config.base_url if hasattr(config, "base_url") else ""
            api_key = config.api_key if hasattr(config, "api_key") else "intercepted"
            model = config.model if hasattr(config, "model") else ""

            _log.info(
                "%s driver: starting interception proxy on :%d → %s",
                self.spec.name,
                _PROXY_PORT,
                base_url,
            )
            proxy_bg_job, base_url_override, proxy_trace_path = self._start_proxy(
                sandbox,
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
            _log.info("%s driver: proxy up at %s", self.spec.name, base_url_override)

        agent_bg_job = self._start_agent(
            sandbox,
            task,
            config,
            base_url_override=base_url_override,
        )

        return CLIAgentSession(
            spec=self.spec,
            sandbox=sandbox,
            task=task,
            config=config,
            verifier=verifier,
            base_url_override=base_url_override,
            proxy_trace_path=proxy_trace_path,
            proxy_bg_job=proxy_bg_job,
            agent_bg_job=agent_bg_job,
        )

    # Bootstrap stages

    def _bootstrap_sandbox(
        self,
        sandbox: SandboxHandle,
        task: Any,
        config: Any,
    ) -> None:
        """Install agent, upload files, write MCP config."""

        # Stage 1: wait for sandbox readiness
        self._wait_for_sandbox_ready(sandbox)

        # Stage 2: install agent CLI (skip if pre-baked)
        if not self._agent_already_installed(sandbox):
            self._install_agent(sandbox)

        # Stage 3: upload spec.files
        self._upload_files(sandbox, task, config)

        # Stage 4: write MCP config (if the spec provides a builder)
        self._write_mcp_config(sandbox, config)

        # Stage 5: run task.setup_shell if present
        setup_shell = task.setup_shell if hasattr(task, "setup_shell") else None
        if setup_shell:
            r = sandbox.exec(setup_shell, timeout=self._setup_timeout_s)
            if r.exit_code != 0:
                raise RuntimeError(
                    f"task.setup_shell failed ({r.exit_code}): {r.stderr}"
                )

    def _wait_for_sandbox_ready(
        self,
        sandbox: SandboxHandle,
        *,
        attempts: int = 15,
        delay_s: float = 1.0,
    ) -> None:
        """Probe sandbox until ``echo ok`` succeeds."""
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
        """Check if the agent CLI is already available in the sandbox."""
        cmd = " ".join(shlex.quote(c) for c in self.spec.install_check_cmd)
        try:
            r = sandbox.exec(cmd, timeout=10)
            return r.exit_code == 0
        except Exception:
            return False

    def _install_agent(self, sandbox: SandboxHandle) -> None:
        """Run ``spec.setup`` commands to install the agent CLI."""
        if self.spec.setup is None:
            raise RuntimeError(
                f"Agent {self.spec.name!r} is not installed in the sandbox "
                "and no setup commands are provided in the spec."
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

    def _upload_files(
        self,
        sandbox: SandboxHandle,
        task: Any,
        config: Any,
    ) -> None:
        """Upload ``spec.files`` into the sandbox, resolving callables."""
        if not self.spec.files:
            return
        for path, content_or_fn in self.spec.files.items():
            if callable(content_or_fn):
                content = content_or_fn(task, config)
            else:
                content = content_or_fn
            if content is not None:
                sandbox.write_text(path, content)

        # Also upload task.upload_files if the task has them.
        upload_files = task.upload_files if hasattr(task, "upload_files") else {}
        for path, content in upload_files.items():
            sandbox.write_text(path, content)

    def _write_mcp_config(
        self,
        sandbox: SandboxHandle,
        config: Any,
    ) -> None:
        """Write MCP configuration using the spec's builder."""
        if self.spec.build_mcp_config is None:
            return
        if (
            self.spec.mcp_config.method == "config_file"
            and self.spec.mcp_config.path_template
        ):
            workdir = (
                config.sandbox_home + "/workdir"
                if hasattr(config, "sandbox_home")
                else "/home/user/workdir"
            )
            home = (
                config.sandbox_home if hasattr(config, "sandbox_home") else "/home/user"
            )
            mcp_path = self.spec.mcp_config.path_template.format(
                workdir=workdir,
                home=home,
            )
            mcp_content = self.spec.build_mcp_config(self.spec, [], workdir)
            if mcp_content:
                sandbox.write_text(mcp_path, mcp_content)

    # Agent launch

    def _start_agent(
        self,
        sandbox: SandboxHandle,
        task: Any,
        config: Any,
        *,
        base_url_override: str | None = None,
    ) -> BgJob:
        """Build CLI command, resolve env vars, and launch as bg process."""
        # Build command via spec hook
        if self.spec.build_command is not None:
            cmd = self.spec.build_command(self.spec, config, task, None)
        else:
            cmd = " ".join(shlex.quote(c) for c in self.spec.base_command)

        # Resolve environment variables
        envs = self._resolve_env_vars(config, base_url_override=base_url_override)

        _log.info("%s driver: launching agent", self.spec.name)
        return sandbox.start_bg(cmd, envs=envs)

    def _resolve_env_vars(
        self,
        config: Any,
        *,
        base_url_override: str | None = None,
    ) -> dict[str, str]:
        """Build the env var dict for the agent process.

        If ``spec.build_env_vars`` is provided, delegate to it.
        Otherwise resolve ``{placeholder}`` substitutions in ``spec.env``.
        """
        if self.spec.build_env_vars is not None:
            return self.spec.build_env_vars(self.spec, config)

        if not self.spec.env:
            return {}

        base_url = base_url_override or (
            config.base_url if hasattr(config, "base_url") else ""
        )
        api_key = config.api_key if hasattr(config, "api_key") else "intercepted"
        model = config.model if hasattr(config, "model") else ""

        substitutions = {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }

        resolved: dict[str, str] = {}
        for key, value in self.spec.env.items():
            try:
                resolved[key] = value.format(**substitutions)
            except KeyError:
                # If a placeholder isn't in our substitutions, keep it as-is.
                resolved[key] = value
        return resolved

    # Interception proxy

    def _start_proxy(
        self,
        sandbox: SandboxHandle,
        *,
        base_url: str,
        api_key: str,
        model: str,
    ) -> tuple[BgJob, str, str]:
        """Install deps, start proxy as bg job, wait for healthz.

        Returns ``(proxy_bg_job, base_url_override, proxy_trace_path)``.
        """
        proxy_already_present = sandbox.exists("/home/user/proxy/interception.py")

        if not proxy_already_present:
            self._exec_with_retry(
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
            base_url,
            "--trace",
            _PROXY_TRACE_PATH,
            "--port",
            str(_PROXY_PORT),
            "--top-logprobs",
            str(self._proxy_top_logprobs),
        ]
        if self._proxy_max_tokens_cap is not None:
            proxy_args.extend(["--max-tokens-cap", str(self._proxy_max_tokens_cap)])
        if self._proxy_disable_thinking:
            proxy_args.append("--disable-thinking")
        if model:
            proxy_args.extend(["--model-override", model])

        quoted = " ".join(shlex.quote(a) for a in proxy_args)
        proxy_cmd = (
            f"cd /home/user/proxy && {quoted} > {shlex.quote(_PROXY_LOG_PATH)} 2>&1"
        )
        proxy_env = {"OPENCODE_UPSTREAM_API_KEY": api_key}
        proxy_job = sandbox.start_bg(proxy_cmd, envs=proxy_env)

        # Wait for proxy healthz
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
            log_content = ""
            try:
                log_content = sandbox.read_text(_PROXY_LOG_PATH)
            except Exception:
                pass
            proxy_job.kill()
            raise RuntimeError(
                f"proxy did not start within {attempts * interval_s:.0f}s. "
                f"log:\n{log_content[-2000:]}"
            )

        override_url = f"http://127.0.0.1:{_PROXY_PORT}/v1"
        return proxy_job, override_url, _PROXY_TRACE_PATH

    # Utilities

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
        """Run ``sandbox.exec`` with exponential backoff on transient failure."""
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
                if last_stderr.strip():
                    break
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


# CLIAgentSessionFactory


class CLIAgentSessionFactory(ResourceSessionFactory):
    """Factory that produces :class:`CLIAgentSession` instances for any
    registered agent.

    Wraps :class:`CLIAgentDriver` to satisfy the
    :class:`ResourceSessionFactory` contract from PR #603.
    """

    def __init__(
        self,
        *,
        spec: CLIAgentSpec,
        config: Any,
        sandbox_backend: SandboxBackend,
        mode: Literal["black_box", "transparent_proxy"] = "black_box",
        verifier: Verifier | None = None,
        install_timeout_s: int = 240,
        setup_timeout_s: int = 300,
        proxy_top_logprobs: int = 5,
        proxy_max_tokens_cap: int | None = 16384,
        proxy_disable_thinking: bool = False,
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
            proxy_top_logprobs=proxy_top_logprobs,
            proxy_max_tokens_cap=proxy_max_tokens_cap,
            proxy_disable_thinking=proxy_disable_thinking,
        )

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> CLIAgentSession:
        """Create one isolated session for a rollout."""
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
]
