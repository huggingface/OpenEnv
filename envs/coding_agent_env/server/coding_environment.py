# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coding-agent MCP environment.

Single MCP tool ``run_rollout`` with a uniform task shape:

  - ``instruction``  — prompt for the selected agent
  - ``setup``        — bash commands run BEFORE the agent (in the sandbox)
  - ``verify``       — bash commands run AFTER the agent

Reward = ``passed_verify_commands / total`` unless a verify command writes
a float to ``/home/user/logs/verifier/reward.txt`` (override).

Returns a JSON-serialized :class:`RolloutResult` with reward,
setup/verify command results, and file outputs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation

    from .catalog import ENDPOINT_KINDS, resolve_endpoint
except ImportError:  # pragma: no cover
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation
    from server.catalog import ENDPOINT_KINDS, resolve_endpoint  # type: ignore


# One rollout (sandbox cold start + harness install + agent run +
# verifier) typically takes 30-180s; can spike to ~600s under load. Override
# OpenEnv's 30s MCP-tool default so the server doesn't cut us off.
_RUN_ROLLOUT_TIMEOUT_S = 900.0

# Inside-sandbox paths the server writes/reads.
HOME = "/home/user"
WORKDIR = f"{HOME}/workdir"
INSTRUCTION_PATH = f"{HOME}/task/instruction.md"
_log = logging.getLogger(__name__)

REWARD_FILE = f"{HOME}/logs/verifier/reward.txt"
PROXY_LOG = f"{HOME}/logs/agent/proxy.log"
AGENT_LOG = f"{HOME}/logs/agent/opencode.jsonl"
VERIFY_TIMEOUT_S = 120
_SUPPORTED_AGENTS = ("opencode", "pi")
_AGENT_LOG_BY_AGENT: dict[str, str] = {
    "opencode": f"{HOME}/logs/agent/opencode.jsonl",
    "pi": f"{HOME}/logs/agent/pi.txt",
}


class _GenericAgentConfig(BaseModel):
    """Minimal config shape for CLIAgentSessionFactory-backed agents."""

    base_url: str
    api_key: str
    model: str
    agent_timeout_s: float = 600.0
    sandbox_home: str = HOME
    provider: str | None = None
    thinking: str | None = "off"
    extra_env: dict[str, str] = Field(default_factory=dict)


class CodingAgentEnvironment(MCPEnvironment):
    """Per-session environment exposing a single ``run_rollout`` MCP tool."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        # Lazy imports so module import stays cheap and so tests can patch.
        try:
            from ..models import (
                CodingAgentState,
                CommandResult,
                RolloutResult,
                RolloutTurn,
            )
        except ImportError:  # pragma: no cover
            from models import (  # type: ignore
                CodingAgentState,
                CommandResult,
                RolloutResult,
                RolloutTurn,
            )

        from openenv.core.harness.agents import get_agent_spec
        from openenv.core.harness.agents.cli_driver import CLIAgentSessionFactory

        from coding_agent_env.config import CodingAgentConfig
        from coding_agent_env.harness import CodingAgentSessionFactory
        from coding_agent_env.task import CodingAgentTask

        try:
            from openenv.core.harness.sandbox import E2BSandboxBackend
        except ImportError:
            E2BSandboxBackend = None  # type: ignore[assignment,misc]

        self._CommandResult = CommandResult
        self._RolloutResult = RolloutResult
        self._RolloutTurn = RolloutTurn
        self._CodingAgentState = CodingAgentState
        self._CodingAgentConfig = CodingAgentConfig
        self._CodingAgentSessionFactory = CodingAgentSessionFactory
        self._CodingAgentTask = CodingAgentTask
        self._E2BSandboxBackend = E2BSandboxBackend
        self._CLIAgentSessionFactory = CLIAgentSessionFactory
        self._get_agent_spec = get_agent_spec

        # Don't raise on missing E2B_API_KEY here — OpenEnv's web-interface
        # layer instantiates the env at import time for schema introspection,
        # and we want the docs / Gradio UI to load even when the operator is
        # just exploring. The real check happens lazily in
        # ``_run_rollout_impl`` (any rollout without creds fails fast there
        # with a clear error in the result payload).
        self._state = self._CodingAgentState(episode_id=str(uuid4()))

        mcp = FastMCP("coding_agent_env")

        @mcp.tool
        def run_rollout(
            # Agent + endpoint.
            agent: str = "opencode",
            # Endpoint — either a shorthand (resolved from env vars + catalog
            # defaults) OR explicit base_url+api_key+model. Explicit fields
            # always win over the catalog.
            endpoint: str = "",
            base_url: str = "",
            api_key: str = "",
            model: str = "",
            # Task
            instruction: str = "",
            setup: Optional[list[str]] = None,
            verify: Optional[list[str]] = None,
            # Bookkeeping / tunables
            task_id: str = "",
            mode: str = "black_box",
            disable_thinking: Optional[bool] = None,
            max_tokens_cap: int = 4096,
            top_logprobs: int = 5,
            agent_timeout_s: float = 600.0,
            template: str = "",
        ) -> str:
            """Run one coding-agent rollout end-to-end.

            ``agent`` selects the harness CLI to run inside the sandbox.
            Currently supported: ``"opencode"``, ``"pi"``.

            ``endpoint`` is the shorthand selector (one of
            ``"vllm"`` / ``"openai"`` / ``"hf_router"``) — the server
            resolves base_url / api_key / model from env vars + catalog
            defaults. Pass any of those explicitly to override.

            See ``coding_agent_env.client.CodingAgentEnv.run_rollout`` for full
            arg docs. Returns a JSON-serialized ``RolloutResult``.
            """
            # Resolve via catalog when shorthand is provided.
            disable_thinking_resolved = disable_thinking
            if endpoint:
                resolved = resolve_endpoint(
                    endpoint, base_url=base_url, api_key=api_key, model=model
                )
                base_url = resolved.base_url
                api_key = resolved.api_key
                model = resolved.model
                if disable_thinking_resolved is None:
                    disable_thinking_resolved = resolved.disable_thinking_default
            if disable_thinking_resolved is None:
                disable_thinking_resolved = False

            agent = (agent or "opencode").strip()
            if agent not in _SUPPORTED_AGENTS:
                raise ValueError(
                    f"unsupported agent {agent!r}; supported agents: {_SUPPORTED_AGENTS}"
                )
            if not (base_url and api_key and model):
                raise ValueError(
                    "must provide either ``endpoint`` (one of "
                    f"{ENDPOINT_KINDS}) or all of base_url + api_key + model"
                )
            if not instruction:
                raise ValueError("instruction is required")

            return self._run_rollout_impl(
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                instruction=instruction,
                setup=list(setup or []),
                verify=list(verify or []),
                task_id=task_id,
                mode=mode,
                disable_thinking=disable_thinking_resolved,
                max_tokens_cap=max_tokens_cap,
                top_logprobs=top_logprobs,
                agent_timeout_s=agent_timeout_s,
                template=template,
            )

        super().__init__(mcp)

    # ── OpenEnv lifecycle ──────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **_: Any,
    ) -> Observation:
        self._state = self._CodingAgentState(episode_id=episode_id or str(uuid4()))
        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "message": (
                    "coding_agent_env ready. Call run_rollout(agent=..., ...) with a task."
                ),
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
                    "Use CallToolAction(name='run_rollout', ...)."
                ),
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        if timeout_s is None:
            timeout_s = _RUN_ROLLOUT_TIMEOUT_S
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        if timeout_s is None:
            timeout_s = _RUN_ROLLOUT_TIMEOUT_S
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> Any:
        return self._state

    # ── Rollout orchestration ──────────────────────────────────────────────

    def _run_rollout_impl(
        self,
        *,
        agent: str,
        base_url: str,
        api_key: str,
        model: str,
        instruction: str,
        setup: list[str],
        verify: list[str],
        task_id: str,
        mode: str,
        disable_thinking: bool,
        max_tokens_cap: int,
        top_logprobs: int,
        agent_timeout_s: float,
        template: str,
        progress_cb=None,
    ) -> str:
        # Optional progress callback: receives short status strings at each
        # phase boundary so the Gradio UI can stream live updates. Safe to
        # be None (silently no-op).
        def _emit(msg: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(msg)
                except Exception:
                    pass

        result = self._RolloutResult(task_id=task_id, mode=mode)
        t0 = time.time()

        # Late credential check — keeps the server importable in dev /
        # docs-only contexts.
        if not os.environ.get("E2B_API_KEY"):
            result.error = (
                "E2B_API_KEY is not set on the server. Configure it in the "
                "Space's secrets / your .env / your shell before calling "
                "run_rollout."
            )
            result.wall_s = round(time.time() - t0, 3)
            _emit("error: E2B_API_KEY missing on server")
            return result.model_dump_json()

        _emit(f"resolving config (agent={agent}, model={model}, mode={mode})")

        config = self._build_agent_config(
            agent=agent,
            mode=mode,
            base_url=base_url,
            api_key=api_key,
            model=model,
            agent_timeout_s=agent_timeout_s,
            disable_thinking=disable_thinking,
            top_logprobs=top_logprobs,
            max_tokens_cap=max_tokens_cap,
        )

        # Concatenate setup commands into a single ``set -e`` script so the
        # primitive runs them inside _bootstrap_sandbox BEFORE the agent
        # starts. This avoids the race where the agent's first tool call
        # depends on files or packages that setup is still installing.
        setup_shell: str | None = None
        if setup:
            # ``set -e`` makes the script abort on the first failing command.
            setup_shell = "set -e\n" + "\n".join(setup)

        rollout_task = self._CodingAgentTask(
            instruction=instruction,
            setup_shell=setup_shell,
            metadata={"task_id": task_id, "agent": agent},
        )

        session = None
        try:
            factory = self._build_session_factory(
                agent=agent,
                config=config,
                mode=mode,
                template=template,
                disable_thinking=disable_thinking,
                top_logprobs=top_logprobs,
                max_tokens_cap=max_tokens_cap,
            )
            _emit(
                f"creating E2B sandbox (template={template or 'default'}) — "
                "this is the slow phase (~5–60s cold, ~5s with template)"
            )
            session = factory.create(task=rollout_task)
            result.sandbox_id = session.sandbox.sandbox_id
            _emit(f"sandbox ready: {result.sandbox_id} — agent started (mode={mode})")

            # setup commands already ran atomically during sandbox bootstrap.
            # Avoid re-running them here because many setup scripts are not
            # idempotent (e.g., migrations, one-shot installs, destructive prep).
            # We still surface per-command bookkeeping for callers.
            for cmd in setup:
                result.setup_results.append(
                    self._CommandResult(
                        cmd=cmd,
                        exit_code=None,
                        stdout="executed during bootstrap (individual exit code not captured)",
                        stderr="",
                        duration_s=0.0,
                    )
                )

            # Block until the agent is done.
            if result.error is None:
                _emit(
                    f"agent running — {agent} CLI in sandbox "
                    f"(timeout {int(agent_timeout_s)}s)"
                )
                try:
                    result.agent_exit_code = session.wait_for_completion(
                        timeout_s=agent_timeout_s
                    )
                    _emit(f"agent finished: exit_code={result.agent_exit_code}")
                except TimeoutError as exc:
                    result.error = f"agent timeout: {exc}"
                    _emit(f"agent TIMEOUT: {exc}")

            # Run verify commands one at a time, capture each.
            verify_passed = 0
            for i, cmd in enumerate(verify, 1):
                _emit(f"verify [{i}/{len(verify)}]: {cmd[:80]}")
                cr = self._exec_command(session.sandbox, cmd)
                result.verify_results.append(cr)
                if cr.exit_code == 0:
                    verify_passed += 1

            # Reward: explicit reward.txt wins; else passed/total of verify.
            override = self._read_reward(session.sandbox)
            if override is not None:
                result.reward = override
            elif verify:
                result.reward = verify_passed / len(verify)
            else:
                result.reward = None

            # Collect filesystem + proxy trace.
            _emit("collecting workdir files + proxy trace + logs")
            result.files, result.files_extra = self._collect_files(session.sandbox)
            result.proxy_turns = self._collect_proxy_turns(session)
            result.proxy_log_tail = self._safe_read(session.sandbox, PROXY_LOG)[-2000:]
            result.agent_log_tail = self._collect_agent_log_tail(session, agent)
            _emit(
                f"collected: {len(result.files)} file(s), "
                f"{len(result.proxy_turns)} proxy turn(s), "
                f"reward={'%.2f' % result.reward if result.reward is not None else 'n/a'}"
            )
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            _emit(f"ERROR: {result.error}")
            if session is not None:
                result.proxy_log_tail = self._safe_read(session.sandbox, PROXY_LOG)[
                    -2000:
                ]
                result.agent_log_tail = self._collect_agent_log_tail(session, agent)
        finally:
            if session is not None:
                try:
                    _emit("tearing down sandbox")
                    session.close()
                except Exception:
                    pass

        result.wall_s = round(time.time() - t0, 3)
        _emit(f"done in {result.wall_s:.1f}s")

        # Bookkeeping on the per-session state.
        self._state.rollouts_completed += 1
        self._state.last_reward = result.reward
        self._state.last_task_id = task_id or None
        self._state.last_sandbox_id = result.sandbox_id or None

        return result.model_dump_json()

    def _build_agent_config(
        self,
        *,
        agent: str,
        mode: str,
        base_url: str,
        api_key: str,
        model: str,
        agent_timeout_s: float,
        disable_thinking: bool,
        top_logprobs: int,
        max_tokens_cap: int,
    ) -> Any:
        if agent == "opencode":
            if top_logprobs:
                _log.warning(
                    "top_logprobs=%d is not supported for agent='opencode' "
                    "and will have no effect. Use interception_gate mode for "
                    "logprob capture.",
                    top_logprobs,
                )
            return self._CodingAgentConfig(
                provider="openai_compatible",
                base_url=base_url.rstrip("/"),
                api_key=api_key,
                model=model,
                agent_timeout_s=agent_timeout_s,
                disable_thinking=disable_thinking,
                max_tokens_cap=max_tokens_cap if max_tokens_cap > 0 else None,
            )

        provider = self._infer_pi_provider(base_url)
        return _GenericAgentConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            agent_timeout_s=agent_timeout_s,
            provider=provider,
            thinking="off" if disable_thinking else None,
        )

    def _build_session_factory(
        self,
        *,
        agent: str,
        config: Any,
        mode: str,
        template: str,
        disable_thinking: bool,
        top_logprobs: int,
        max_tokens_cap: int,
    ) -> Any:
        if self._E2BSandboxBackend is None:
            raise RuntimeError(
                "E2BSandboxBackend unavailable: install optional dependency 'e2b'."
            )

        backend_kwargs: dict[str, Any] = {}
        if template:
            backend_kwargs["template"] = template
        backend = self._E2BSandboxBackend(**backend_kwargs)

        if agent == "opencode":
            return self._CodingAgentSessionFactory(
                config=config,
                sandbox_backend=backend,
                mode=mode,
                verifier=None,
            )

        spec = self._get_agent_spec(agent)
        return self._CLIAgentSessionFactory(
            spec=spec,
            config=config,
            sandbox_backend=backend,
            mode=mode,
            verifier=None,
        )

    @staticmethod
    def _infer_pi_provider(base_url: str) -> str:
        url = (base_url or "").lower()
        if "router.huggingface.co" in url:
            return "huggingface"
        if "anthropic" in url:
            return "anthropic"
        if "googleapis.com" in url or "generativelanguage" in url:
            return "gemini"
        return "openai"

    def _collect_agent_log_tail(self, session: Any, agent: str) -> str:
        if hasattr(session, "collect_artifacts"):
            try:
                artifacts = session.collect_artifacts()
                if isinstance(artifacts, dict) and "agent_log" in artifacts:
                    val = artifacts["agent_log"]
                    if isinstance(val, str):
                        return val[-2000:]
                    return json.dumps(val, default=str)[-2000:]
            except Exception:
                pass
        path = _AGENT_LOG_BY_AGENT.get(agent, AGENT_LOG)
        return self._safe_read(session.sandbox, path)[-2000:]

    # ── Helpers ────────────────────────────────────────────────────────────

    def _exec_command(self, sandbox: Any, cmd: str) -> Any:
        t = time.time()
        try:
            r = sandbox.exec(cmd, timeout=VERIFY_TIMEOUT_S)
            return self._CommandResult(
                cmd=cmd,
                exit_code=int(r.exit_code),
                stdout=(r.stdout or "")[-2000:],
                stderr=(r.stderr or "")[-2000:],
                duration_s=round(time.time() - t, 3),
            )
        except Exception as exc:  # noqa: BLE001
            return self._CommandResult(
                cmd=cmd,
                exit_code=-1,
                stderr=f"{type(exc).__name__}: {exc}",
                duration_s=round(time.time() - t, 3),
            )

    def _read_reward(self, sandbox: Any) -> float | None:
        raw = self._safe_read(sandbox, REWARD_FILE).strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _collect_files(self, sandbox: Any) -> tuple[dict[str, str], list[str]]:
        listing = sandbox.exec(
            f"find {WORKDIR} -maxdepth 2 -type f -size -64k 2>/dev/null | head -32",
            timeout=10,
        )
        files: dict[str, str] = {}
        extras: list[str] = []
        for line in (listing.stdout or "").splitlines():
            path = line.strip()
            if not path:
                continue
            try:
                files[path] = sandbox.read_text(path)[:8000]
            except Exception:
                extras.append(path)
        return files, extras

    def _collect_proxy_turns(self, session: Any) -> list[Any]:
        """Logprob capture is now owned by the training loop via interception_gate."""
        return []

    @staticmethod
    def _safe_read(sandbox: Any, path: str) -> str:
        try:
            return sandbox.read_text(path) or ""
        except Exception:
            return ""
