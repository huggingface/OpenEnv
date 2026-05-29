# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE harness session and session factory.

Integrates ``mini_swe_env`` with the ``CLIAgentDriver`` / ``ResourceSession``
harness infrastructure.  Pi runs in the sandbox with its built-in tools
(bash, edit, write, read, grep, find, ls) plus one host-side tool:
``answer``.

Session lifecycle::

    factory = SWESessionFactory(agent="pi", config=..., sandbox_backend=..., ...)
    session = factory.create(task=swe_gym_task.to_swe_task())

    # interception_gate mode:
    request = await session.next_request()
    await session.deliver(request, response_dict)
    ...
    # OR black_box mode:
    session.wait_for_completion(timeout_s=600)

    vr = session.verify(transcript=[])
    print(vr.env_reward)   # 1.0 or 0.0 (binary)
    session.close()

**Reward architecture**: The ``answer`` tool is a **host-side tool**
routed through the InterceptionServer's tool routing layer (``/vf/tools``).
When the agent calls ``answer()``, the request goes to the host, which
runs SWE-Gym-native grading (revert test files → apply test_patch → run
explicit FAIL_TO_PASS/PASS_TO_PASS tests), and returns the result to the
agent.
This is the same result ``verify()`` returns — one grading path, no
in-sandbox grading infrastructure.

This matches SWE-Gym's architecture where ``answer`` is server-side code
on the OpenReward platform.

**Requires core changes** — see ``CORE_CHANGES.md`` for the
InterceptionServer tool routing, models.json, workdir, and Docker
host-IP changes needed in ``openenv.core``.
"""

from __future__ import annotations

import asyncio
import json
import queue as _queue_mod
import logging
import shlex
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from openenv.core.harness import Message, ResourceSessionFactory, VerifyResult
from openenv.core.harness.agents import get_agent_spec
from openenv.core.harness.agents.cli_driver import (
    CLIAgentDriver,
    CLIAgentSession,
    build_interception_rollout_url,
)
from openenv.core.harness.agents.interception_server import InterceptionServer
from openenv.core.harness.sandbox import SandboxBackend, SandboxHandle

from .grading import grade_from_case_results
from .models import SWEGymTask, SWETask, coerce_swe_task, validate_swe_task


_log = logging.getLogger(__name__)

# ── Sandbox filesystem layout (SWE-Gym convention) ─────────────────────────

HOME = "/home/user"
TESTBED = "/testbed"

VERIFY_TIMEOUT_S = 300
SETUP_TIMEOUT_S = 600

_ANSWER_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Submit your final answer for SWE grading.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}


# ── SWE instruction template ──────────────────────────────────────────────

_SWE_INSTRUCTION_TEMPLATE = """<pr_description>
Consider the following PR description:
{problem_statement}
</pr_description>

<instructions>
# Task Instructions

## Overview
You're a software engineer working on a codebase at /testbed.
Your task is to fix the issue described in the PR description above
by making changes to the source code (non-test files).

## Important Boundaries
- MODIFY: Regular source code files in /testbed
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow
1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Submitting Your Answer
When you've completed your work and verified your fix, call the `answer`
tool to submit your solution for grading. This runs the test suite and
returns whether the issue is resolved.

You cannot continue working after submitting — make sure your fix is
tested before calling `answer`.
</instructions>"""


def _wrap_instruction(problem_statement: str) -> str:
    """Wrap a problem statement with SWE-Gym-style task instructions.

    Tells the agent about the workflow, boundaries, and crucially
    about the ``answer`` tool for submission.
    """
    return _SWE_INSTRUCTION_TEMPLATE.format(
        problem_statement=problem_statement,
    )


@dataclass
class SWEAgentConfig:
    """Minimal config for the CLI agent driver."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    agent_timeout_s: float = 600.0
    sandbox_home: str = HOME
    workdir: str = TESTBED
    provider: str = ""
    thinking: str = "off"


@dataclass
class _SWEAgentTask:
    """Internal task shape passed to CLIAgentDriver (not SWETask)."""

    instruction: str = ""
    setup_shell: str | None = None
    upload_files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── SWE Session ────────────────────────────────────────────────────────────


class SWESession(CLIAgentSession):
    """Per-rollout session with SWE-specific verify and reward logic.

    Extends :class:`CLIAgentSession` with:
    - ``verify()`` — returns the reward produced by the host-side
      ``answer`` tool, or grades the repo state post-hoc if no answer
      was called.
    - SWE task metadata.

    **Reward architecture**: The ``answer`` tool is optional. If the
    agent calls it, grading runs immediately and returns feedback.
    If the agent never calls ``answer()`` (timeout, max turns, natural
    exit), ``verify()`` grades the repo state post-hoc — the agent's
    edits are evaluated regardless of explicit submission.
    """

    def __init__(
        self,
        *,
        swe_task: SWETask,
        verify_timeout_s: int = VERIFY_TIMEOUT_S,
        grade_fn: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._swe_task = swe_task
        self._verify_timeout_s = verify_timeout_s
        self._answer_reward: float | None = None  # set by host-side answer tool
        self._answer_reward_source: str | None = None
        self._answer_called = False
        self._answer_bridged = False
        # Callback for post-hoc grading: (sandbox, swe_task) -> (reward, resolved)
        self._grade_fn = grade_fn

    @property
    def swe_task(self) -> SWETask:
        return self._swe_task

    @property
    def answer_reward(self) -> float | None:
        return self._answer_reward

    @property
    def answer_reward_source(self) -> str | None:
        return self._answer_reward_source

    @property
    def answer_called(self) -> bool:
        return self._answer_called

    @property
    def answer_bridged(self) -> bool:
        return self._answer_bridged

    def mark_answer_called(self) -> None:
        self._answer_called = True

    def mark_answer_bridged(self) -> None:
        self._answer_bridged = True

    def set_answer_reward(
        self,
        reward: float,
        *,
        source: str = "host_answer_tool",
    ) -> None:
        """Called by the host-side answer tool handler to store the reward."""
        self._answer_reward = reward
        self._answer_reward_source = source

    def initial_messages(self) -> list[Message]:
        """Return the SWE instruction as the initial prompt."""
        return [{"role": "user", "content": self._swe_task.instruction}]

    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        """Return the reward computed by the host-side ``answer`` tool.

        If the agent called ``answer()`` during the rollout, the
        InterceptionServer's tool handler already computed the reward
        and stored it via ``set_answer_reward()``.

        If the agent never called ``answer()`` (timeout, crash),
        falls back to verify commands (legacy) or defaults to 0.0.
        """
        # 1. Primary: reward from host-side answer tool.
        if self._answer_reward is not None:
            return VerifyResult(
                env_reward=self._answer_reward,
                done=True,
                metrics={
                    "instance_id": self._swe_task.instance_id,
                    "reward_source": self._answer_reward_source
                    or "host_answer_tool",
                    "answer_called": self._answer_called,
                    "answer_bridged": self._answer_bridged,
                },
                artifacts={
                    "task_id": self._swe_task.task_id,
                },
            )

        # 2. Guardrail: answer was attempted but host-side reward not recorded.
        if self._answer_called:
            return VerifyResult(
                env_reward=0.0,
                done=True,
                metrics={
                    "instance_id": self._swe_task.instance_id,
                    "reward_source": "answer_called_missing_host_reward",
                    "answer_called": True,
                    "answer_bridged": self._answer_bridged,
                },
                artifacts={
                    "task_id": self._swe_task.task_id,
                },
            )

        # 3. Fallback: run verify commands (legacy tasks with shell commands).
        if self._swe_task.verify:
            passed = 0
            verify_details: list[dict[str, Any]] = []

            for cmd in self._swe_task.verify:
                t0 = time.time()
                try:
                    r = self.sandbox.exec(
                        cmd, cwd=TESTBED, timeout=self._verify_timeout_s
                    )
                    detail = {
                        "cmd": cmd,
                        "exit_code": r.exit_code,
                        "stdout_tail": (r.stdout or "")[-2000:],
                        "stderr_tail": (r.stderr or "")[-2000:],
                        "duration_s": round(time.time() - t0, 3),
                    }
                    if r.exit_code == 0:
                        passed += 1
                except Exception as exc:
                    detail = {
                        "cmd": cmd,
                        "exit_code": -1,
                        "error": f"{type(exc).__name__}: {exc}",
                        "duration_s": round(time.time() - t0, 3),
                    }
                verify_details.append(detail)

            fallback_reward = passed / len(self._swe_task.verify)
            return VerifyResult(
                env_reward=fallback_reward,
                done=True,
                metrics={
                    "verify_passed": passed,
                    "verify_total": len(self._swe_task.verify),
                    "instance_id": self._swe_task.instance_id,
                    "reward_source": "verify_commands",
                    "answer_called": False,
                    "answer_bridged": False,
                },
                artifacts={
                    "verify_details": verify_details,
                    "task_id": self._swe_task.task_id,
                },
            )

        # 4. Post-hoc grading: grade the repo state regardless of answer().
        # whether or not it explicitly submitted.
        if self._grade_fn is not None:
            try:
                reward, resolved = self._grade_fn(self.sandbox, self._swe_task)
                source = "post_hoc_grade"
                return VerifyResult(
                    env_reward=float(reward),
                    done=True,
                    metrics={
                        "instance_id": self._swe_task.instance_id,
                        "reward_source": source,
                        "resolved": resolved,
                        "answer_called": False,
                        "answer_bridged": False,
                    },
                    artifacts={
                        "task_id": self._swe_task.task_id,
                    },
                )
            except Exception:
                _log.exception(
                    "post-hoc grading failed for %s", self._swe_task.instance_id
                )

        # 5. No grading possible (no grade_fn, no verify cmds, no answer).
        return VerifyResult(
            env_reward=0.0,
            done=True,
            metrics={
                "instance_id": self._swe_task.instance_id,
                "reward_source": "default_no_answer",
                "answer_called": False,
                "answer_bridged": False,
            },
            artifacts={
                "task_id": self._swe_task.task_id,
            },
        )


# ── Tool-call parsing (kept for backward compatibility) ────────────────────


def parse_terminal_call(text: str) -> dict[str, Any] | None:
    """Parse a terminal tool-call from text.

    Handles multiple formats:
    - ``{"command": "..."}``
    - ``{"final_answer": "..."}``
    - ``terminal(command="...")`` (Python-style)

    Returns parsed arguments dict or None if not a terminal call.
    """
    text = text.strip()
    if not text:
        return None

    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and ("command" in data or "final_answer" in data):
                return data
        except json.JSONDecodeError:
            pass

    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                try:
                    data = json.loads(block)
                    if isinstance(data, dict) and (
                        "command" in data or "final_answer" in data
                    ):
                        return data
                except json.JSONDecodeError:
                    continue

    for key in ("command", "final_answer"):
        prefix = f"terminal({key}="
        if prefix in text:
            idx = text.index(prefix) + len(prefix)
            rest = text[idx:]
            if rest.startswith('"') or rest.startswith("'"):
                quote = rest[0]
                end = rest.find(quote, 1)
                if end > 0:
                    return {key: rest[1:end]}

    return None


# ── SWE Session Factory ───────────────────────────────────────────────────


class SWESessionFactory(ResourceSessionFactory):
    """Creates isolated SWE sessions.

    The ``answer`` tool is registered as a host-side tool on the
    InterceptionServer (via ``/vf/tools``).  No in-sandbox grading
    scripts or extensions are deployed.

    Compatible with :func:`build_harness_rollout_func`.
    """

    def __init__(
        self,
        *,
        agent: str = "pi",
        config: SWEAgentConfig,
        sandbox_backend: SandboxBackend,
        mode: Literal["black_box", "interception_gate"] = "black_box",
        install_timeout_s: int = 300,
        setup_timeout_s: int = SETUP_TIMEOUT_S,
        verify_timeout_s: int = VERIFY_TIMEOUT_S,
        interception_server: InterceptionServer | None = None,
        interception_base_url: str | None = None,
    ) -> None:
        if mode not in {"black_box", "interception_gate"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        if mode == "interception_gate":
            if interception_server is None:
                raise ValueError(
                    "interception_gate mode requires an InterceptionServer."
                )
            if interception_base_url is None:
                raise ValueError(
                    "interception_gate mode requires interception_base_url."
                )

        self._config = config
        self._backend = sandbox_backend
        self._mode = mode
        self._verify_timeout_s = verify_timeout_s
        self._interception_server = interception_server
        self._interception_base_url = interception_base_url

        self._spec = get_agent_spec(agent)
        self._driver = CLIAgentDriver(
            spec=self._spec,
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
    ) -> SWESession:
        """Create one SWE session.

        ``task`` can be an ``SWETask``, ``SWEGymTask``, or a dict.
        """
        if isinstance(task, SWEGymTask):
            swe_task = task.to_swe_task()
        elif isinstance(task, SWETask):
            swe_task = task
        else:
            swe_task = coerce_swe_task(task)
        validate_swe_task(swe_task)

        sandbox_timeout = int(self._config.agent_timeout_s) + 600
        sandbox = self._backend.create(
            timeout_s=sandbox_timeout,
            metadata=(
                {"episode_id": episode_id, "instance_id": swe_task.instance_id}
                if episode_id
                else {"instance_id": swe_task.instance_id}
            ),
            image=swe_task.sandbox_image,
        )

        try:
            if not swe_task.sandbox_image:
                self._prepare_repo(sandbox, swe_task)

            self._run_setup(sandbox, swe_task)

            agent_task = self._build_agent_task(swe_task)
            self._driver._bootstrap_sandbox(sandbox, agent_task, self._config)

        except Exception as exc:
            _log.error("SWESessionFactory.create: bootstrap failed: %r", exc)
            sandbox.kill()
            raise

        base_url_override: str | None = None
        interception_rollout_id: str | None = None
        interception_queue: _queue_mod.Queue[str] | None = None

        if self._mode == "interception_gate":
            assert self._interception_server is not None
            assert self._interception_base_url is not None
            rollout_id = episode_id or f"rollout_{uuid.uuid4().hex[:8]}"
            interception_rollout_id = rollout_id
            interception_queue = self._interception_server.register_rollout(rollout_id)
            base_url_override = build_interception_rollout_url(
                self._interception_base_url,
                rollout_id,
            )

        agent_task = self._build_agent_task(swe_task)
        agent_bg = self._driver._start_agent(
            sandbox, agent_task, self._config, base_url_override=base_url_override
        )

        session = SWESession(
            swe_task=swe_task,
            verify_timeout_s=self._verify_timeout_s,
            grade_fn=self._grade_answer_submission,
            spec=self._spec,
            sandbox=sandbox,
            task=agent_task,
            config=self._config,
            base_url_override=base_url_override,
            agent_bg_job=agent_bg,
            interception_server=self._interception_server,
            interception_rollout_id=interception_rollout_id,
            interception_queue=interception_queue,
        )

        if self._mode == "interception_gate":
            self._register_answer_tool(session)

        return session

    # ── Bootstrap helpers ──────────────────────────────────────────────────

    def _prepare_repo(self, sandbox: SandboxHandle, task: SWETask) -> None:
        """Clone the repo and reset to base_commit."""
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

    def _run_setup(self, sandbox: SandboxHandle, task: SWETask) -> None:
        """Run task setup commands in the workspace."""
        for cmd in task.setup:
            r = sandbox.exec(cmd, cwd=TESTBED, timeout=SETUP_TIMEOUT_S)
            if r.exit_code != 0:
                raise RuntimeError(
                    f"Setup command failed (exit {r.exit_code}): "
                    f"{cmd[:120]}\nstderr: {(r.stderr or '')[:500]}"
                )

    def _build_agent_task(self, swe_task: SWETask) -> _SWEAgentTask:
        """Convert SWETask into the shape CLIAgentDriver expects.

        Wraps the raw problem statement with SWE-Gym-style instructions
        that tell the agent about the ``answer`` tool.
        """
        return _SWEAgentTask(
            instruction=_wrap_instruction(swe_task.instruction),
            setup_shell=None,
            metadata={
                "task_id": swe_task.task_id,
                "instance_id": swe_task.instance_id,
                "repo": swe_task.repo,
            },
        )

    def _register_answer_tool(self, session: SWESession) -> None:
        """Register the host-side ``answer`` tool for one interception rollout."""

        async def _answer_handler(arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments

            session.mark_answer_called()
            session.mark_answer_bridged()

            if session.answer_reward is not None:
                resolved = session.answer_reward >= 1.0
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"✅ Resolved: {str(resolved).lower()}",
                        }
                    ]
                }

            reward, resolved = await asyncio.to_thread(
                self._grade_answer_submission,
                session.sandbox,
                session.swe_task,
            )
            session.set_answer_reward(reward, source="host_answer_tool")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"✅ Resolved: {str(resolved).lower()}",
                    }
                ]
            }

        session.register_tool_handler(
            "answer",
            _answer_handler,
            tool_definition=_ANSWER_TOOL_DEFINITION,
        )

    def _grade_answer_submission(
        self,
        sandbox: SandboxHandle,
        swe_task: SWETask,
    ) -> tuple[float, bool]:
        """Compute answer-tool reward on host and return ``(reward, resolved)``."""
        try:
            metadata = swe_task.metadata or {}
            required = {"version", "patch", "test_patch", "FAIL_TO_PASS"}
            if required.issubset(metadata):
                return self._grade_with_swegym_metadata(sandbox, swe_task)
            return self._grade_with_verify_commands(sandbox, swe_task)
        except Exception:
            _log.exception("answer-tool grading failed for %s", swe_task.instance_id)
            return 0.0, False

    def _grade_with_swegym_metadata(
        self,
        sandbox: SandboxHandle,
        swe_task: SWETask,
    ) -> tuple[float, bool]:
        """Grade SWE-Gym tasks directly from FAIL/PASS test-case outcomes."""
        metadata = swe_task.metadata
        assert metadata is not None

        gym_task = SWEGymTask(
            instance_id=swe_task.instance_id,
            repo=swe_task.repo,
            base_commit=swe_task.base_commit,
            problem_statement=swe_task.instruction,
            version=str(metadata["version"]),
            patch=str(metadata["patch"]),
            test_patch=str(metadata["test_patch"]),
            FAIL_TO_PASS=[str(t) for t in metadata["FAIL_TO_PASS"]],
            PASS_TO_PASS=[str(t) for t in metadata.get("PASS_TO_PASS", [])],
            hints_text=str(metadata.get("hints_text", "")),
            created_at=str(metadata.get("created_at", "")),
            timeout_s=swe_task.timeout_s,
        )

        touched_files = self._extract_paths_from_test_patch(gym_task.test_patch)
        self._revert_test_files(
            sandbox,
            base_commit=swe_task.base_commit,
            paths=touched_files,
            strict=True,
        )

        self._apply_test_patch(sandbox, gym_task.test_patch)
        case_results = self._run_swegym_case_tests(sandbox, gym_task)
        grade = grade_from_case_results(gym_task, case_results)

        # Best-effort cleanup in case grading was interrupted.
        self._revert_test_files(
            sandbox,
            base_commit=swe_task.base_commit,
            paths=touched_files,
            strict=False,
        )

        return float(grade.reward), bool(grade.resolved)

    def _apply_test_patch(self, sandbox: SandboxHandle, test_patch: str) -> None:
        patch_path = f"{HOME}/.openenv_swe_test_patch.diff"
        sandbox.write_text(patch_path, test_patch)
        result = sandbox.exec(
            f"git apply --whitespace=nowarn {shlex.quote(patch_path)}",
            cwd=TESTBED,
            timeout=30,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                "failed to apply SWE-Gym test_patch: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )

    def _run_swegym_case_tests(
        self,
        sandbox: SandboxHandle,
        gym_task: SWEGymTask,
    ) -> dict[str, bool]:
        cases: list[str] = []
        seen: set[str] = set()
        for case in [*gym_task.FAIL_TO_PASS, *gym_task.PASS_TO_PASS]:
            if case in seen:
                continue
            seen.add(case)
            cases.append(case)

        results: dict[str, bool] = {}
        for case in cases:
            cmd = f"python -m pytest -q --maxfail=1 {shlex.quote(case)}"
            run = sandbox.exec(cmd, cwd=TESTBED, timeout=self._verify_timeout_s)
            results[case] = run.exit_code == 0
        return results

    def _grade_with_verify_commands(
        self,
        sandbox: SandboxHandle,
        swe_task: SWETask,
    ) -> tuple[float, bool]:
        """Legacy fallback for non-SWE-Gym tasks."""
        if not swe_task.verify:
            return 0.0, False
        passed = 0
        for cmd in swe_task.verify:
            r = sandbox.exec(cmd, cwd=TESTBED, timeout=self._verify_timeout_s)
            if r.exit_code == 0:
                passed += 1
        reward = passed / len(swe_task.verify)
        return reward, reward >= 1.0

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
        sandbox: SandboxHandle,
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

        if not failures:
            return

        msg = "failed to revert test files before/after grading: " + "; ".join(failures)
        if strict:
            raise RuntimeError(msg)
        _log.warning(msg)


__all__ = [
    "SWEAgentConfig",
    "SWESession",
    "SWESessionFactory",
    "_wrap_instruction",
    "parse_terminal_call",
    "HOME",
    "TESTBED",
]
