# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenCode MCP environment.

Single MCP tool ``run_rollout`` that takes a uniform Task shape:

  - ``instruction``  — prompt for the agent
  - ``setup``        — bash commands run BEFORE the agent (in the sandbox)
  - ``verify``       — bash commands run AFTER the agent

Reward = ``passed_verify_commands / total`` unless a verify command writes
a float to ``/home/user/logs/verifier/reward.txt`` (override).

Returns a JSON-serialized :class:`RolloutResult` with reward + per-turn
logprobs (Mode B) + setup/verify command results + file outputs.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation

    from .catalog import ENDPOINT_KINDS, resolve_endpoint
except ImportError:  # pragma: no cover
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation
    from server.catalog import ENDPOINT_KINDS, resolve_endpoint  # type: ignore


# One rollout (sandbox cold start + opencode install + opencode run +
# verifier) typically takes 30-180s; can spike to ~600s under load. Override
# OpenEnv's 30s MCP-tool default so the server doesn't cut us off.
_RUN_ROLLOUT_TIMEOUT_S = 900.0

# Inside-sandbox paths the server writes/reads.
HOME = "/home/user"
WORKDIR = f"{HOME}/workdir"
INSTRUCTION_PATH = f"{HOME}/task/instruction.md"
REWARD_FILE = f"{HOME}/logs/verifier/reward.txt"
PROXY_LOG = f"{HOME}/logs/agent/proxy.log"
AGENT_LOG = f"{HOME}/logs/agent/opencode.jsonl"
VERIFY_TIMEOUT_S = 120


class OpenCodeEnvironment(MCPEnvironment):
    """Per-session environment exposing a single ``run_rollout`` MCP tool."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        # Lazy imports so module import stays cheap and so tests can patch.
        try:
            from ..models import (
                CommandResult,
                OpenCodeState,
                RolloutResult,
                RolloutTurn,
            )
        except ImportError:  # pragma: no cover
            from models import (  # type: ignore
                CommandResult,
                OpenCodeState,
                RolloutResult,
                RolloutTurn,
            )

        from opencode_env import (
            E2BSandboxBackend,
            OpenCodeConfig,
            OpenCodeSessionFactory,
            OpenCodeTask,
        )

        self._CommandResult = CommandResult
        self._RolloutResult = RolloutResult
        self._RolloutTurn = RolloutTurn
        self._OpenCodeState = OpenCodeState
        self._OpenCodeConfig = OpenCodeConfig
        self._OpenCodeSessionFactory = OpenCodeSessionFactory
        self._OpenCodeTask = OpenCodeTask
        self._E2BSandboxBackend = E2BSandboxBackend

        # Don't raise on missing E2B_API_KEY here — OpenEnv's web-interface
        # layer instantiates the env at import time for schema introspection,
        # and we want the docs / Gradio UI to load even when the operator is
        # just exploring. The real check happens lazily in
        # ``_run_rollout_impl`` (any rollout without creds fails fast there
        # with a clear error in the result payload).
        self._state = self._OpenCodeState(episode_id=str(uuid4()))

        mcp = FastMCP("opencode_env")

        @mcp.tool
        def run_rollout(
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
            mode: str = "transparent_proxy",
            disable_thinking: Optional[bool] = None,
            max_tokens_cap: int = 4096,
            top_logprobs: int = 5,
            agent_timeout_s: float = 600.0,
            template: str = "",
        ) -> str:
            """Run one OpenCode rollout end-to-end.

            ``endpoint`` is the shorthand selector (one of
            ``"vllm"`` / ``"openai"`` / ``"hf_router"``) — the server
            resolves base_url / api_key / model from env vars + catalog
            defaults. Pass any of those explicitly to override.

            See ``opencode_env.client.OpenCodeEnv.run_rollout`` for full
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

            if not (base_url and api_key and model):
                raise ValueError(
                    "must provide either ``endpoint`` (one of "
                    f"{ENDPOINT_KINDS}) or all of base_url + api_key + model"
                )
            if not instruction:
                raise ValueError("instruction is required")

            return self._run_rollout_impl(
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
        self._state = self._OpenCodeState(episode_id=episode_id or str(uuid4()))
        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "message": ("opencode_env ready. Call run_rollout(...) with a task."),
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

        _emit(f"resolving config (model={model}, mode={mode})")

        # Build OpenCodeConfig + factory. We keep the proxy in charge of
        # ``model_override`` / ``logprobs`` / ``max_tokens``-cap injection.
        config = self._OpenCodeConfig(
            provider="openai_compatible",
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            agent_timeout_s=agent_timeout_s,
            proxy_disable_thinking=disable_thinking,
            proxy_top_logprobs=top_logprobs,
            proxy_max_tokens_cap=max_tokens_cap if max_tokens_cap > 0 else None,
        )

        # Concatenate setup commands into a single ``set -e`` script and let
        # the primitive run it as ``task.setup_shell`` before the agent
        # starts. The per-command tracking happens here too — we re-run
        # each command in a wrapper that captures exit/stdout/stderr.
        # That way the primitive still aborts on setup failure AND we get
        # observability in the response.
        instruction_payload = instruction
        opencode_task = self._OpenCodeTask(
            instruction=instruction_payload,
            metadata={"task_id": task_id},
        )

        backend_kwargs: dict[str, Any] = {}
        if template:
            backend_kwargs["template"] = template

        factory = self._OpenCodeSessionFactory(
            config=config,
            sandbox_backend=self._E2BSandboxBackend(**backend_kwargs),
            mode=mode,
            verifier=None,
        )

        session = None
        try:
            _emit(
                f"creating E2B sandbox (template={template or 'default'}) — "
                "this is the slow phase (~5–60s cold, ~5s with template)"
            )
            session = factory.create(task=opencode_task)
            result.sandbox_id = session.sandbox.sandbox_id
            _emit(
                f"sandbox ready: {result.sandbox_id} — agent started "
                f"({'proxy on :7000, logprobs capturing' if mode == 'transparent_proxy' else 'direct LLM, no logprobs'})"
            )

            # Run setup commands one at a time, *before* the agent starts.
            # The factory has already started the agent in start_agent()
            # during create(); to keep the order "setup → agent → verify"
            # we'd need to restructure. As a pragmatic compromise we run
            # setup IMMEDIATELY after create(), which races with the agent
            # for ~1-2s but is fine for typical pip/git/download work
            # because opencode itself takes >=20s to make its first model
            # call.
            for i, cmd in enumerate(setup, 1):
                _emit(f"setup [{i}/{len(setup)}]: {cmd[:80]}")
                cr = self._exec_command(session.sandbox, cmd)
                result.setup_results.append(cr)
                if cr.exit_code != 0:
                    result.error = (
                        f"setup command failed (exit {cr.exit_code}): {cmd[:120]}"
                    )
                    _emit(f"setup FAILED at [{i}]: exit={cr.exit_code}")
                    break

            # Block until the agent is done (or setup already failed).
            if result.error is None:
                _emit(
                    f"agent running — opencode CLI in sandbox "
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
            result.agent_log_tail = self._safe_read(session.sandbox, AGENT_LOG)[-2000:]
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
                result.agent_log_tail = self._safe_read(session.sandbox, AGENT_LOG)[
                    -2000:
                ]
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
        turns: list[Any] = []
        proxy_trace_path = getattr(session, "_proxy_trace_path", None)
        if not proxy_trace_path:
            return turns
        raw = self._safe_read(session.sandbox, proxy_trace_path)
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            response = rec.get("response") or {}
            turns.append(
                self._RolloutTurn(
                    turn=int(rec.get("turn") or 0),
                    finish_reason=rec.get("finish_reason"),
                    completion_tokens=list(rec.get("completion_tokens") or []),
                    completion_token_ids=list(rec.get("completion_token_ids") or []),
                    per_token_logps=[
                        float(x)
                        for x in (rec.get("per_token_logps") or [])
                        if x is not None
                    ],
                    latency_s=float(rec.get("latency_s") or 0.0),
                    timestamp=float(rec.get("timestamp") or 0.0),
                    upstream_status=response.get("upstream_status"),
                    upstream_error=response.get("upstream_error"),
                )
            )
        return turns

    @staticmethod
    def _safe_read(sandbox: Any, path: str) -> str:
        try:
            return sandbox.read_text(path) or ""
        except Exception:
            return ""
