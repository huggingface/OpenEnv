# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenCode session factory + session implementation.

Implements the :class:`ResourceSessionFactory` / :class:`ResourceSession`
contracts from ``openenv.core.harness`` (PR #471). The session wraps one
sandbox running the ``opencode`` CLI agent.

Two operating modes:

  - ``mode="black_box"`` — opencode talks directly to ``config.base_url``.
    No proxy, no logprob capture. Use for smoke tests / SFT / eval.
  - ``mode="transparent_proxy"`` — an in-sandbox FastAPI proxy
    sits between opencode and the upstream LLM. It injects ``logprobs=true``
    on every request and writes per-turn ``(messages, completion_tokens,
    per_token_logps)`` to ``proxy_trace.jsonl`` for GRPO consumption.

Single driver path: opencode is started as a background subprocess via
``opencode run --format json --dangerously-skip-permissions ...`` and we
poll its exit code. The previous ``opencode serve`` driver was removed —
opencode CLI is the only path now.
"""

from __future__ import annotations

import json
import shlex
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

from .config import OpenCodeConfig
from .opencode_runtime import (
    agent_log_path,
    build_env_vars,
    build_install_cmd,
    build_opencode_json,
    build_run_cmd,
    instruction_path,
    opencode_bin_path,
    opencode_config_path,
    proxy_dir,
    proxy_log_path,
    proxy_source_path,
    proxy_trace_path,
    system_prompt_path,
)
from .sandbox.base import BgJob, SandboxBackend, SandboxHandle
from .task import OpenCodeTask

# The upstream installer resolves "latest" via api.github.com and prints this
# to stdout (with an empty stderr) when the lookup fails, e.g. on rate limit.
_INSTALL_VERSION_FETCH_ERROR = "Failed to fetch version information"


# Mode B proxy port. In-sandbox paths derive from config.sandbox_home (opencode_runtime).
_PROXY_PORT = 7000

# Local proxy source, uploaded to proxy_source_path(config) unless already baked in.
_PROXY_SOURCE_PATH = Path(__file__).parent / "sandbox" / "interception.py"


Verifier = Callable[[SandboxHandle, OpenCodeTask], VerifyResult]


class OpenCodeSession(ResourceSession):
    """One live OpenCode rollout inside a sandbox.

    The session is created already-running: :meth:`OpenCodeSessionFactory.create`
    calls :meth:`start_agent` before returning. Typical usage::

        session = factory.create(task)
        session.wait_for_completion()
        result = session.verify([])
        session.close()
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
    ) -> None:
        self.sandbox = sandbox
        self.config = config
        self.task = task
        self._verifier = verifier
        self._base_url_override = base_url_override
        self._bg_job: BgJob | None = None
        self._proxy_trace_path = proxy_trace_path
        self._proxy_bg_job = proxy_bg_job

    # ------------------------------------------------------------------
    # ResourceSession contract (PR #471)
    # ------------------------------------------------------------------
    def initial_messages(self) -> list[Message]:
        return [{"role": "user", "content": self.task.instruction}]

    def list_tools(self) -> list[Tool]:
        # OpenCode owns its own tool loop — none are exposed to the harness.
        return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(
            error=(
                "OpenCodeSession does not expose external tool calls; the "
                "CLI agent owns its own tool loop."
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
        if self._bg_job is not None:
            try:
                self._bg_job.kill()
            except Exception:
                pass
            self._bg_job = None
        if self._proxy_bg_job is not None:
            try:
                self._proxy_bg_job.kill()
            except Exception:
                pass
            self._proxy_bg_job = None
        self.sandbox.kill()

    # ------------------------------------------------------------------
    # OpenCode-specific session API
    # ------------------------------------------------------------------
    def start_agent(self) -> None:
        """Launch ``opencode run`` as a background subprocess in the sandbox."""
        if self._bg_job is not None:
            return
        cmd = build_run_cmd(self.config)
        envs = build_env_vars(self.config, base_url_override=self._base_url_override)
        self._bg_job = self.sandbox.start_bg(cmd, envs=envs)

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        """Block until the agent exits, returning its exit code."""
        budget = timeout_s if timeout_s is not None else self.config.agent_timeout_s
        if self._bg_job is None:
            raise RuntimeError("Agent not started; call start_agent() first.")
        return self._bg_job.wait(timeout=budget)

    def fetch_trace(self) -> str:
        """Return the raw ``opencode run`` log (JSON-lines when ``run_format=json``)."""
        return self.sandbox.read_text(agent_log_path(self.config))

    def fetch_proxy_trace(self) -> list[dict[str, Any]]:
        """Return per-turn proxy-captured records (Mode B only).

        Each entry has ``request``, ``response``, ``completion_tokens``,
        ``completion_token_ids``, ``per_token_logps``, ``finish_reason``,
        and ``latency_s``. Returns ``[]`` in Mode A.
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


class OpenCodeSessionFactory(ResourceSessionFactory[OpenCodeSession]):
    """Produce isolated per-rollout :class:`OpenCodeSession` instances.

    The factory owns sandbox provisioning, opencode install, config injection,
    and (Mode B) proxy startup. Each :meth:`create` call returns a fresh
    sandbox with a running agent.
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
        create_attempts: int = 3,
        create_backoff_s: float = 2.0,
    ) -> None:
        if mode not in {"black_box", "transparent_proxy"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self._config = config
        self._backend = sandbox_backend
        self._mode = mode
        self._verifier = verifier
        self._install_timeout_s = install_timeout_s
        self._setup_timeout_s = setup_timeout_s
        self._create_attempts = create_attempts
        self._create_backoff_s = create_backoff_s

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
        start_agent: bool = True,
    ) -> OpenCodeSession:
        """Create one session, retrying with exponential backoff.

        Session creation spins up a sandbox, installs opencode, and starts the
        proxy + agent, the flakiest step in a rollout. Each failed attempt tears
        its own sandbox down (see :meth:`_create_once`), so a retry never leaks.

        Pass ``start_agent=False`` to return before launching the agent (for
        example to run setup first), then call ``session.start_agent()``.
        """
        import logging
        import time

        _log = logging.getLogger(__name__)
        last_exc: Exception = RuntimeError("create_attempts must be >= 1")
        for i in range(self._create_attempts):
            try:
                return self._create_once(
                    task, seed=seed, episode_id=episode_id, start_agent=start_agent
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if i + 1 < self._create_attempts:
                    backoff = self._create_backoff_s * (2**i)
                    _log.warning(
                        "factory.create attempt %d/%d failed (%r); retrying in %.1fs",
                        i + 1, self._create_attempts, exc, backoff,
                    )
                    time.sleep(backoff)
        raise last_exc

    def _create_once(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
        start_agent: bool = True,
    ) -> OpenCodeSession:
        import logging
        _log = logging.getLogger(__name__)

        oc_task = OpenCodeTask.coerce(task)
        sandbox_timeout = int(self._config.agent_timeout_s) + 300

        _log.info(
            "factory.create: creating sandbox timeout=%ds mode=%s",
            sandbox_timeout, self._mode,
        )
        sandbox = self._backend.create(
            timeout_s=sandbox_timeout,
            metadata={"episode_id": episode_id} if episode_id else None,
        )
        sid = (
            getattr(sandbox, "sandbox_id", None)
            or getattr(getattr(sandbox, "raw", None), "sandbox_id", "?")
        )
        _log.info("factory.create: sandbox=%s — bootstrapping…", sid)
        # Any failure past here (bootstrap/proxy/agent) must tear the sandbox down.
        try:
            self._bootstrap_sandbox(sandbox, oc_task)

            base_url_override: str | None = None
            proxy_trace_path: str | None = None
            proxy_bg_job: BgJob | None = None
            if self._mode == "transparent_proxy":
                _log.info(
                    "factory.create: starting interception proxy on :%d → %s",
                    _PROXY_PORT, self._config.base_url,
                )
                proxy_bg_job, base_url_override, proxy_trace_path = self._start_proxy(
                    sandbox
                )
                _log.info("factory.create: proxy up at %s", base_url_override)
                # Rewrite opencode.json so opencode points at the proxy. Force
                # ``openai_compatible`` so opencode hits ``/v1/chat/completions``
                # (which the proxy serves) rather than provider-specific paths.
                from .config import OpenCodeConfig as _OCC

                proxy_cfg = _OCC(
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
            if start_agent:
                session.start_agent()
            return session
        except Exception as exc:
            _log.error("factory.create: setup failed, killing sandbox: %r", exc)
            try:
                sandbox.kill()  # best-effort: don't let a cleanup failure mask the root cause
            except Exception:
                _log.exception("factory.create: sandbox.kill() during cleanup also failed")
            raise

    # ------------------------------------------------------------------
    def _wait_for_sandbox_ready(
        self,
        sandbox: SandboxHandle,
        *,
        attempts: int = 15,
        delay_s: float = 1.0,
    ) -> None:
        """Probe the sandbox until ``echo ok`` succeeds.

        E2B (and other backends) sometimes return the handle before the
        guest is fully ready. Issue ``echo ok`` with short timeouts until
        it succeeds. Returns silently on success; raises ``RuntimeError``
        on prolonged failure.
        """
        import time

        last_err = ""
        for _ in range(attempts):
            try:
                r = sandbox.exec("echo ok", timeout=5)
                if r.exit_code == 0 and "ok" in (r.stdout or ""):
                    return
                last_err = (r.stderr or r.stdout or "").strip() or f"exit={r.exit_code}"
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(delay_s)
        raise RuntimeError(
            f"sandbox did not become ready within {attempts * delay_s:.0f}s "
            f"(last error: {last_err})"
        )

    def _exec_with_retry(
        self,
        sandbox: SandboxHandle,
        cmd: str,
        *,
        timeout: float,
        attempts: int = 3,
        backoff_s: float = 3.0,
        label: str = "cmd",
        fatal_markers: tuple[str, ...] = (),
    ):
        """Run ``sandbox.exec`` with exponential backoff on transient failure.

        Transient = ``exit_code != 0`` AND empty stderr (SIGKILL / network
        blip signature) OR an exception during exec. A failure whose stdout
        contains one of ``fatal_markers`` is deterministic and not retried,
        even if stderr is empty. Final failure is raised as ``RuntimeError``
        carrying the last exit code + stderr.
        """
        import time

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
                if any(marker in last_stdout for marker in fatal_markers):
                    break
            except Exception as exc:  # noqa: BLE001
                last_stderr = f"{type(exc).__name__}: {exc}"
                last_exit = -1
            if i + 1 < attempts:
                time.sleep(backoff_s * (2**i))
        raise RuntimeError(
            f"{label} failed after {attempts} attempts "
            f"(exit={last_exit}, stderr={last_stderr!r}, stdout_tail={last_stdout[-400:]!r})"
        )

    def _opencode_already_installed(self, sandbox: SandboxHandle) -> bool:
        """Cheap probe — returns True if opencode is on disk in the sandbox.

        Used to skip the slow ``curl install`` step when running against a
        prebaked template that already ships opencode.
        """
        try:
            r = sandbox.exec(
                f"{opencode_bin_path(self._config)} --version",
                timeout=10,
            )
            return r.exit_code == 0
        except Exception:
            return False

    def _bootstrap_sandbox(
        self,
        sandbox: SandboxHandle,
        task: OpenCodeTask,
    ) -> None:
        """Install opencode, write config + task files, run optional setup."""

        # Stage 1: wait for the sandbox to be responsive.
        self._wait_for_sandbox_ready(sandbox)

        # Stage 2: install opencode (skipped if a prebaked template already
        # has it). curl|bash is flaky — retry with backoff.
        if not self._opencode_already_installed(sandbox):
            try:
                self._exec_with_retry(
                    sandbox,
                    build_install_cmd(self._config),
                    timeout=self._install_timeout_s,
                    attempts=3,
                    backoff_s=3.0,
                    label="opencode install",
                    fatal_markers=(_INSTALL_VERSION_FETCH_ERROR,),
                )
            except RuntimeError as exc:
                if _INSTALL_VERSION_FETCH_ERROR in str(exc):
                    raise RuntimeError(
                        "opencode install could not resolve 'latest' from the "
                        "GitHub API (rate limited?). Pin opencode_version to a "
                        "release tag to install without touching the API."
                    ) from exc
                raise

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

        if self._config.extra_setup_shell:
            self._exec_with_retry(
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
        """Install proxy deps + start the proxy as a bg job inside the sandbox.

        Returns ``(proxy_bg_job, base_url_override, proxy_trace_path)``.
        Skips the pip install + source-upload steps when the prebaked
        template already has them in place.
        """
        proxy_already_present = sandbox.exists(proxy_source_path(self._config))

        if not proxy_already_present:
            # Install proxy deps (idempotent on retries).
            self._exec_with_retry(
                sandbox,
                "set -o pipefail && pip install --quiet 'fastapi>=0.104' "
                "'uvicorn[standard]>=0.24' 'httpx>=0.27' 2>&1 | tail -20",
                timeout=180,
                attempts=3,
                backoff_s=2.0,
                label="proxy deps install",
            )
            # Upload the proxy module into the sandbox.
            sandbox.write_text(
                proxy_source_path(self._config),
                _PROXY_SOURCE_PATH.read_text(),
            )
            sandbox.write_text(f"{proxy_dir(self._config)}/__init__.py", "")

        proxy_args = [
            "python",
            "interception.py",
            "--upstream-url",
            self._config.base_url,
            "--trace",
            proxy_trace_path(self._config),
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
        # Force the upstream model id on every forwarded request — opencode's
        # internal title-gen call sometimes strips the provider prefix.
        if self._config.model:
            proxy_args.extend(["--model-override", self._config.model])

        quoted_proxy_args = " ".join(shlex.quote(arg) for arg in proxy_args)
        proxy_cmd = (
            f"cd {shlex.quote(proxy_dir(self._config))} && "
            f"{quoted_proxy_args} "
            f"> {shlex.quote(proxy_log_path(self._config))} 2>&1"
        )
        proxy_env = {"OPENCODE_UPSTREAM_API_KEY": self._config.api_key}
        proxy_job = sandbox.start_bg(proxy_cmd, envs=proxy_env)

        # Wait for the proxy to start listening. Cold uvicorn boot inside
        # E2B can take anywhere from <1s to ~30s depending on cache state.
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
                log = sandbox.read_text(proxy_log_path(self._config))
            except Exception:
                pass
            proxy_job.kill()
            raise RuntimeError(
                f"proxy did not start within {attempts * interval_s:.0f}s. "
                f"log:\n{log[-2000:]}"
            )

        base_url_override = f"http://127.0.0.1:{_PROXY_PORT}/v1"
        return proxy_job, base_url_override, proxy_trace_path(self._config)


__all__ = [
    "OpenCodeSession",
    "OpenCodeSessionFactory",
    "OpenCodeTask",
    "Verifier",
]
