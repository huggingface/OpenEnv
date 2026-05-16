# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE environment implementation (v2 — SWE-Gym-native grading).

Single MCP tool ``run_swe_rollout`` with the ``SWEGymTask`` shape:

  - ``instance_id``  — SWE-Gym instance identifier
  - ``repo``         — GitHub ``org/repo``
  - ``base_commit``  — commit hash in per-task Docker image
  - ``problem_statement`` — agent instruction (problem text)

**Reward** is binary via SWE-Gym test-case outcomes:
  - ``1.0`` if resolved (all FAIL_TO_PASS pass, all PASS_TO_PASS still pass)
  - ``0.0`` otherwise

**Per-task Docker images** from SWE-Gym (``xingyaoww/...``): the repo is
pre-cloned at ``/testbed`` with all dependencies installed.  No git clone
or dependency install at runtime.

The ``terminal`` tool is delivered via an in-sandbox MCP server
(:mod:`sandbox_mcp_server`) started before the agent launches.
"""

from __future__ import annotations

import json
import logging
import shlex
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation

    from ..grading import GradeResult, grade_from_case_results
    from ..models import (
        SWECommandResult,
        SWEGymTask,
        SWERolloutResult,
        SWEState,
        SWETask,
        validate_swe_task,
    )
    from ..task_loader_swegym import (
        get_instance_image,
        validate_swegym_task,
    )
except ImportError:  # pragma: no cover
    from grading import GradeResult, grade_from_case_results  # type: ignore
    from models import (  # type: ignore
        SWECommandResult,
        SWEGymTask,
        SWERolloutResult,
        SWEState,
        SWETask,
        validate_swe_task,
    )
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation
    from task_loader_swegym import get_instance_image, validate_swegym_task  # type: ignore

_log = logging.getLogger(__name__)

# Long timeout for the single MCP tool (sandbox cold-start + agent run +
# verify can take 10-30 min for real SWE tasks).
_RUN_ROLLOUT_TIMEOUT_S = 2400.0

# Sandbox filesystem layout — SWE-Gym convention.
# Per-task images have the repo pre-cloned at /testbed.
TESTBED = "/testbed"
HOME = "/home/user"
MCP_CONFIG_PATH = f"{HOME}/.swe_mcp_config.json"
MCP_SERVER_PATH = f"{HOME}/.swe_mcp_server.py"
MCP_PORT = 8765
VERIFY_TIMEOUT_S = 300
SETUP_TIMEOUT_S = 600

# Path to the sandbox_mcp_server.py source alongside this module.
_SANDBOX_MCP_SERVER_SOURCE = Path(__file__).parent / "sandbox_mcp_server.py"

_SUPPORTED_AGENTS = ("pi", "opencode")
_AGENT_LOG_PATHS: dict[str, str] = {
    "pi": f"{HOME}/logs/agent/pi.txt",
    "opencode": f"{HOME}/logs/agent/opencode.jsonl",
}


class SWEEnvironment(MCPEnvironment):
    """Per-session SWE environment exposing ``run_swe_rollout`` MCP tool.

    Updated for SWE-Gym:
      - Per-task Docker images (no git clone needed).
      - Binary reward via SWE-Gym case outcomes.
      - Working directory is ``/testbed``.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        from openenv.core.harness.agents import get_agent_spec
        from openenv.core.harness.agents.cli_driver import CLIAgentSessionFactory

        self._get_agent_spec = get_agent_spec
        self._CLIAgentSessionFactory = CLIAgentSessionFactory

        self._state = SWEState(episode_id=str(uuid4()))

        mcp = FastMCP("mini_swe_env")

        @mcp.tool
        def run_swe_rollout(
            # Task fields (match SWEGymTask / SWETask shape).
            instance_id: str = "",
            repo: str = "",
            base_commit: str = "",
            instruction: str = "",
            setup: Optional[list[str]] = None,
            verify: Optional[list[str]] = None,
            timeout_s: int = 1800,
            # Agent config.
            agent: str = "pi",
            base_url: str = "",
            api_key: str = "",
            model: str = "",
            agent_timeout_s: float = 600.0,
            # Infrastructure.
            sandbox_backend: str = "docker",
            sandbox_image: str = "",
            task_id: str = "",
            task_json: str = "",
        ) -> str:
            """Run one SWE rollout end-to-end.

            Pass either individual fields (instance_id, repo, ...) or a
            complete SWETask/SWEGymTask as ``task_json``.  Returns a
            JSON-serialized ``SWERolloutResult``.
            """
            return self._run_swe_rollout_impl(
                instance_id=instance_id,
                repo=repo,
                base_commit=base_commit,
                instruction=instruction,
                setup=list(setup or []),
                verify=list(verify or []),
                timeout_s=timeout_s,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                agent_timeout_s=agent_timeout_s,
                sandbox_backend=sandbox_backend,
                sandbox_image=sandbox_image,
                task_id=task_id,
                task_json=task_json,
            )

        super().__init__(mcp)

    # ── OpenEnv lifecycle ──────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **_: Any,
    ) -> Observation:
        self._state = SWEState(episode_id=episode_id or str(uuid4()))
        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "message": (
                    "mini_swe_env ready. Call run_swe_rollout(...) with an SWE task."
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
                    "Use CallToolAction(name='run_swe_rollout', ...)."
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

    def _run_swe_rollout_impl(
        self,
        *,
        instance_id: str,
        repo: str,
        base_commit: str,
        instruction: str,
        setup: list[str],
        verify: list[str],
        timeout_s: int,
        agent: str,
        base_url: str,
        api_key: str,
        model: str,
        agent_timeout_s: float,
        sandbox_backend: str,
        sandbox_image: str,
        task_id: str,
        task_json: str,
    ) -> str:
        result = SWERolloutResult(task_id=task_id)
        t0 = time.time()

        # ── Resolve task ──────────────────────────────────────────────
        task = self._resolve_task(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            instruction=instruction,
            setup=setup,
            verify=verify,
            timeout_s=timeout_s,
            task_json=task_json,
            task_id=task_id,
        )
        if isinstance(task, str):
            # Error string
            result.error = task
            result.wall_s = round(time.time() - t0, 3)
            return result.model_dump_json()

        result.task_id = task.task_id
        result.instance_id = task.instance_id

        # ── Validate agent + LLM config ───────────────────────────────
        agent = (agent or "pi").strip()
        if agent not in _SUPPORTED_AGENTS:
            result.error = (
                f"Unsupported agent {agent!r}; supported: {_SUPPORTED_AGENTS}"
            )
            result.wall_s = round(time.time() - t0, 3)
            return result.model_dump_json()

        if not (base_url and api_key and model):
            result.error = "Must provide base_url, api_key, and model."
            result.wall_s = round(time.time() - t0, 3)
            return result.model_dump_json()

        # ── Create sandbox ────────────────────────────────────────────
        sandbox = None
        session = None
        try:
            # Use per-task image if available (SWE-Gym convention).
            image = sandbox_image or task.sandbox_image
            backend = self._create_backend(sandbox_backend, image)
            sandbox = backend.create(
                timeout_s=int(agent_timeout_s) + 600,
            )
            result.sandbox_id = sandbox.sandbox_id

            # ── Stage repo (only if not using a per-task image) ───────
            if not image:
                self._stage_repo(sandbox, task)

            # ── Run setup commands ────────────────────────────────────
            for cmd in task.setup:
                cr = self._exec_command(sandbox, cmd, cwd=TESTBED)
                result.setup_results.append(cr)
                if cr.exit_code != 0:
                    result.error = f"Setup failed (exit {cr.exit_code}): {cmd[:120]}"
                    break

            if result.error is not None:
                result.wall_s = round(time.time() - t0, 3)
                return result.model_dump_json()

            # ── Start in-sandbox MCP server ───────────────────────────
            self._deploy_mcp_server(sandbox, task)

            # ── Launch agent ──────────────────────────────────────────
            spec = self._get_agent_spec(agent)
            config = self._build_agent_config(
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                agent_timeout_s=agent_timeout_s,
            )
            rollout_task = self._build_agent_task(task)

            from openenv.core.harness.agents.cli_driver import CLIAgentDriver

            driver = CLIAgentDriver(
                spec=spec,
                sandbox_backend=backend,
                mode="black_box",
            )
            driver._bootstrap_sandbox(sandbox, rollout_task, config)
            agent_bg = driver._start_agent(sandbox, rollout_task, config)

            from openenv.core.harness.agents.cli_driver import CLIAgentSession

            session = CLIAgentSession(
                spec=spec,
                sandbox=sandbox,
                task=rollout_task,
                config=config,
                agent_bg_job=agent_bg,
            )

            # ── Wait for agent ────────────────────────────────────────
            try:
                result.agent_exit_code = session.wait_for_completion(
                    timeout_s=agent_timeout_s
                )
            except TimeoutError as exc:
                result.error = f"Agent timeout: {exc}"

            # ── Grade submission ──────────────────────────────────────
            grade_result = self._grade_submission(sandbox, task)
            if grade_result is not None:
                result.reward = grade_result.reward
                result.resolved = grade_result.resolved
            else:
                # Fallback: legacy verify commands
                result.reward, result.resolved = self._legacy_verify(
                    sandbox, task
                )

            # ── Collect artifacts ──────────────────────────────────────
            result.files, result.files_extra = self._collect_files(sandbox)
            result.agent_log_tail = self._collect_agent_log(sandbox, session, agent)

        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            if sandbox is not None:
                result.agent_log_tail = self._safe_read(
                    sandbox, _AGENT_LOG_PATHS.get(agent, "")
                )[-2000:]
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
            elif sandbox is not None:
                try:
                    sandbox.kill()
                except Exception:
                    pass

        result.wall_s = round(time.time() - t0, 3)

        # ── Update state ──────────────────────────────────────────────
        self._state.rollouts_completed += 1
        self._state.last_reward = result.reward
        self._state.last_task_id = result.task_id or None
        self._state.last_instance_id = result.instance_id or None
        self._state.last_sandbox_id = result.sandbox_id or None

        return result.model_dump_json()

    # ── Task resolution ────────────────────────────────────────────────────

    def _resolve_task(
        self,
        *,
        instance_id: str,
        repo: str,
        base_commit: str,
        instruction: str,
        setup: list[str],
        verify: list[str],
        timeout_s: int,
        task_json: str,
        task_id: str,
    ) -> SWETask | str:
        """Build an SWETask from the provided arguments.

        Returns the task on success, or an error string on failure.
        """
        if task_json:
            try:
                raw = json.loads(task_json)
                # Try SWEGymTask first — it has richer metadata.
                if "problem_statement" in raw and "patch" in raw:
                    gym_task = SWEGymTask(**raw)
                    validate_swegym_task(gym_task)
                    return gym_task.to_swe_task()
                # Fall back to SWETask.
                task = SWETask(**raw)
                validate_swe_task(task)
                return task
            except Exception as exc:
                return f"Invalid task_json: {exc}"

        if not instruction:
            return "instruction is required"
        if not repo:
            return "repo is required"
        if not base_commit:
            return "base_commit is required"
        if not instance_id:
            instance_id = f"manual::{repo}::{base_commit[:12]}"

        # Derive per-task image.
        image = get_instance_image(instance_id) if instance_id else None

        try:
            task = SWETask(
                task_id=task_id or f"swegym::{instance_id}",
                source="swegym",
                instance_id=instance_id,
                repo=repo,
                base_commit=base_commit,
                instruction=instruction,
                setup=setup,
                verify=verify,
                timeout_s=timeout_s,
                sandbox_image=image,
            )
            validate_swe_task(task)
            return task
        except Exception as exc:
            return f"Task validation failed: {exc}"

    # ── Grading ────────────────────────────────────────────────────────────

    def _grade_submission(
        self,
        sandbox: Any,
        task: SWETask,
    ) -> GradeResult | None:
        """Run SWE-Gym-native grading when metadata includes test lists."""
        metadata = task.metadata
        if not metadata:
            return None

        required_keys = {"patch", "test_patch", "FAIL_TO_PASS", "version"}
        if not required_keys.issubset(metadata.keys()):
            return None

        try:
            gym_task = SWEGymTask(
                instance_id=task.instance_id,
                repo=task.repo,
                base_commit=task.base_commit,
                problem_statement=task.instruction,
                version=str(metadata["version"]),
                patch=str(metadata["patch"]),
                test_patch=str(metadata["test_patch"]),
                FAIL_TO_PASS=list(metadata["FAIL_TO_PASS"]),
                PASS_TO_PASS=list(metadata.get("PASS_TO_PASS", [])),
                hints_text=str(metadata.get("hints_text", "")),
                created_at=str(metadata.get("created_at", "")),
                timeout_s=task.timeout_s,
            )
        except Exception as exc:
            _log.warning("Could not reconstruct SWEGymTask for grading: %s", exc)
            return None

        touched_files = self._extract_paths_from_test_patch(gym_task.test_patch)
        try:
            self._revert_test_files(
                sandbox,
                base_commit=task.base_commit,
                paths=touched_files,
                strict=True,
            )
            self._apply_test_patch(sandbox, gym_task.test_patch)
            case_results = self._run_swegym_case_tests(sandbox, gym_task)
            return grade_from_case_results(gym_task, case_results)
        except Exception as exc:
            _log.warning("SWE-Gym grading failed, falling back: %s", exc)
            return None
        finally:
            self._revert_test_files(
                sandbox,
                base_commit=task.base_commit,
                paths=touched_files,
                strict=False,
            )

    def _apply_test_patch(self, sandbox: Any, test_patch: str) -> None:
        patch_path = f"{HOME}/.openenv_swe_test_patch.diff"
        sandbox.write_text(patch_path, test_patch)
        r = sandbox.exec(
            f"git apply --whitespace=nowarn {shlex.quote(patch_path)}",
            cwd=TESTBED,
            timeout=30,
        )
        if r.exit_code != 0:
            raise RuntimeError((r.stderr or r.stdout or "").strip())

    def _run_swegym_case_tests(
        self,
        sandbox: Any,
        task: SWEGymTask,
    ) -> dict[str, bool]:
        cases: list[str] = []
        seen: set[str] = set()
        for case in [*task.FAIL_TO_PASS, *task.PASS_TO_PASS]:
            if case in seen:
                continue
            seen.add(case)
            cases.append(case)

        results: dict[str, bool] = {}
        for case in cases:
            r = sandbox.exec(
                f"python -m pytest -q --maxfail=1 {shlex.quote(case)}",
                cwd=TESTBED,
                timeout=VERIFY_TIMEOUT_S,
            )
            results[case] = r.exit_code == 0
        return results

    @staticmethod
    def _extract_paths_from_test_patch(test_patch: str) -> list[str]:
        paths: list[str] = []
        for line in (test_patch or "").splitlines():
            if not line.startswith("+++ b/"):
                continue
            path = line[len("+++ b/") :].strip()
            if not path or path == "/dev/null":
                continue
            paths.append(path)
        return sorted(set(paths))

    def _revert_test_files(
        self,
        sandbox: Any,
        *,
        base_commit: str,
        paths: list[str],
        strict: bool,
    ) -> None:
        if not paths:
            return

        failures: list[str] = []
        for path in paths:
            has_file = sandbox.exec(
                f"git cat-file -e {shlex.quote(f'{base_commit}:{path}')}",
                cwd=TESTBED,
                timeout=10,
            )
            if has_file.exit_code == 0:
                cmd = (
                    "git checkout --quiet "
                    f"{shlex.quote(base_commit)} -- {shlex.quote(path)}"
                )
            else:
                cmd = f"rm -f -- {shlex.quote(path)}"
            result = sandbox.exec(cmd, cwd=TESTBED, timeout=20)
            if result.exit_code != 0:
                failures.append(
                    f"{path}: {(result.stderr or result.stdout or '').strip()}"
                )

        if failures and strict:
            raise RuntimeError("; ".join(failures))

    def _legacy_verify(
        self,
        sandbox: Any,
        task: SWETask,
    ) -> tuple[float | None, bool | None]:
        """Legacy verify: run verify commands, compute pass ratio."""
        verify_passed = 0
        for cmd in task.verify:
            cr = self._exec_command(sandbox, cmd, cwd=TESTBED)
            if cr.exit_code == 0:
                verify_passed += 1

        if task.verify:
            ratio = verify_passed / len(task.verify)
            return ratio, ratio == 1.0

        return None, None

    # ── Sandbox helpers ────────────────────────────────────────────────────

    def _create_backend(self, backend_name: str, image: str | None) -> Any:
        """Create a sandbox backend by name."""
        from openenv.core.harness.sandbox import create_sandbox_backend

        kwargs: dict[str, Any] = {}
        if image:
            kwargs["image"] = image
        return create_sandbox_backend(backend_name, **kwargs)

    def _stage_repo(self, sandbox: Any, task: SWETask) -> None:
        """Clone the repo and reset to base_commit in the sandbox.

        Only used when there is no per-task Docker image.  SWE-Gym
        per-task images already have the repo at ``/testbed``.
        """
        sandbox.exec(f"mkdir -p {TESTBED}", timeout=10)

        clone_url = f"https://github.com/{task.repo}.git"
        r = sandbox.exec(
            f"git clone --quiet {clone_url} {TESTBED}",
            timeout=SETUP_TIMEOUT_S,
        )
        if r.exit_code != 0:
            raise RuntimeError(
                f"git clone failed (exit {r.exit_code}): {r.stderr[:500]}"
            )

        r = sandbox.exec(
            f"git checkout --quiet {task.base_commit}",
            cwd=TESTBED,
            timeout=60,
        )
        if r.exit_code != 0:
            raise RuntimeError(
                f"git checkout failed (exit {r.exit_code}): {r.stderr[:500]}"
            )

    def _deploy_mcp_server(self, sandbox: Any, task: SWETask) -> None:
        """Write the MCP server script and config into the sandbox, then start it."""
        mcp_source = _SANDBOX_MCP_SERVER_SOURCE.read_text()
        sandbox.write_text(MCP_SERVER_PATH, mcp_source)

        mcp_config = json.dumps(
            {
                "workspace": TESTBED,
                "verify_commands": list(task.verify),
                "timeout_per_command_s": VERIFY_TIMEOUT_S,
                "output_limit": 16_000,
                "port": MCP_PORT,
            },
            indent=2,
        )
        sandbox.write_text(MCP_CONFIG_PATH, mcp_config)

        sandbox.exec(
            f"mkdir -p {HOME}/logs/verifier {HOME}/logs/agent",
            timeout=10,
        )

        sandbox.start_bg(
            f"python3 {MCP_SERVER_PATH}",
            envs={"SWE_MCP_CONFIG": MCP_CONFIG_PATH},
        )

        for attempt in range(10):
            r = sandbox.exec(
                f"curl -sf http://127.0.0.1:{MCP_PORT}/health 2>/dev/null || echo FAIL",
                timeout=5,
            )
            if "FAIL" not in (r.stdout or ""):
                return
            time.sleep(0.5)

        raise RuntimeError("In-sandbox MCP server did not start within 5s")

    def _build_agent_config(
        self,
        *,
        agent: str,
        base_url: str,
        api_key: str,
        model: str,
        agent_timeout_s: float,
    ) -> Any:
        """Build the agent-specific config dataclass."""
        from dataclasses import dataclass

        @dataclass
        class _AgentConfig:
            base_url: str = base_url
            api_key: str = api_key
            model: str = model
            agent_timeout_s: float = agent_timeout_s
            sandbox_home: str = HOME
            provider: str = ""
            thinking: str = "off"

        config = _AgentConfig()

        if agent == "pi":
            config.provider = self._infer_provider(base_url)

        return config

    def _build_agent_task(self, task: SWETask) -> Any:
        """Build a task object compatible with CLIAgentDriver."""
        from dataclasses import dataclass, field as dc_field

        @dataclass
        class _AgentTask:
            instruction: str = task.instruction
            setup_shell: str | None = None
            upload_files: dict[str, str] = dc_field(default_factory=dict)
            metadata: dict[str, Any] = dc_field(default_factory=dict)

        return _AgentTask(
            metadata={
                "task_id": task.task_id,
                "instance_id": task.instance_id,
                "repo": task.repo,
            },
        )

    @staticmethod
    def _infer_provider(base_url: str) -> str:
        url = (base_url or "").lower()
        if "router.huggingface.co" in url:
            return "huggingface"
        if "anthropic" in url:
            return "anthropic"
        if "googleapis.com" in url or "generativelanguage" in url:
            return "gemini"
        return "openai"

    def _exec_command(
        self, sandbox: Any, cmd: str, cwd: str | None = None
    ) -> SWECommandResult:
        """Execute a command and return a structured result."""
        t = time.time()
        try:
            kwargs: dict[str, Any] = {"timeout": VERIFY_TIMEOUT_S}
            if cwd:
                kwargs["cwd"] = cwd
            r = sandbox.exec(cmd, **kwargs)
            return SWECommandResult(
                cmd=cmd,
                exit_code=int(r.exit_code),
                stdout=(r.stdout or "")[-4000:],
                stderr=(r.stderr or "")[-4000:],
                duration_s=round(time.time() - t, 3),
            )
        except Exception as exc:  # noqa: BLE001
            return SWECommandResult(
                cmd=cmd,
                exit_code=-1,
                stderr=f"{type(exc).__name__}: {exc}",
                duration_s=round(time.time() - t, 3),
            )

    def _collect_files(self, sandbox: Any) -> tuple[dict[str, str], list[str]]:
        """Collect modified files from the workspace."""
        listing = sandbox.exec(
            f"cd {TESTBED} && git diff --name-only HEAD 2>/dev/null | head -32",
            timeout=10,
        )
        files: dict[str, str] = {}
        extras: list[str] = []
        for line in (listing.stdout or "").splitlines():
            rel_path = line.strip()
            if not rel_path:
                continue
            full_path = f"{TESTBED}/{rel_path}"
            try:
                content = sandbox.read_text(full_path)
                if len(content) <= 16_000:
                    files[rel_path] = content
                else:
                    files[rel_path] = content[:16_000] + "\n... [truncated]"
            except Exception:
                extras.append(rel_path)
        return files, extras

    def _collect_agent_log(self, sandbox: Any, session: Any, agent: str) -> str:
        """Collect agent log tail."""
        if session is not None and hasattr(session, "collect_artifacts"):
            try:
                artifacts = session.collect_artifacts()
                if isinstance(artifacts, dict) and "agent_log" in artifacts:
                    val = artifacts["agent_log"]
                    if isinstance(val, str):
                        return val[-4000:]
                    return json.dumps(val, default=str)[-4000:]
            except Exception:
                pass
        path = _AGENT_LOG_PATHS.get(agent, "")
        return self._safe_read(sandbox, path)[-4000:]

    @staticmethod
    def _safe_read(sandbox: Any, path: str) -> str:
        if not path:
            return ""
        try:
            return sandbox.read_text(path) or ""
        except Exception:
            return ""
