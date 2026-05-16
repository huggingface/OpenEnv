# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE harness session and session factory (v2 — Pi answer extension).

Integrates ``mini_swe_env`` with the ``CLIAgentDriver`` / ``ResourceSession``
harness infrastructure.  Pi runs in the sandbox with its built-in tools
(bash, edit, write, read, grep, find, ls) plus one extension-registered
tool: ``answer``.

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

**Reward integrity**: The authoritative reward is computed **host-side**
by ``SWESession.verify()``, which extracts the agent's diff from the
sandbox, runs the swebench eval script in the sandbox, downloads the
test log, and grades it on the host via
``swebench.harness.grading.get_eval_report()``.  No ``reward.txt`` is
used — the agent cannot influence the training reward.

The in-sandbox ``answer`` extension gives the agent a fast feedback
signal ("Resolved: true/false") but this is purely informational and
not used for the training reward.

The factory handles:
  1. Sandbox creation (per-task Docker image, /testbed ready)
  2. Deploy Pi ``answer`` extension (swe-answer.ts + swe-grade.sh)
  3. Deploy eval script (from swebench via ``make_eval_script``)
  4. Deploy test patch file
  5. Run task setup commands
  6. Agent bootstrap + launch
  7. Interception gate rollout registration (when mode="interception_gate")

The session handles:
  - ``verify()`` — host-side grading via swebench, returns binary VerifyResult
  - ``initial_messages()`` — instruction prompt
  - Interception gate: ``next_request()`` / ``deliver()``
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openenv.core.harness import Message, ResourceSessionFactory, VerifyResult
from openenv.core.harness.agents import get_agent_spec
from openenv.core.harness.agents.cli_driver import (
    CLIAgentDriver,
    CLIAgentSession,
)
from openenv.core.harness.agents.interception_server import InterceptionServer
from openenv.core.harness.sandbox import SandboxBackend, SandboxHandle

from .models import SWEGymTask, SWETask, coerce_swe_task, validate_swe_task


_log = logging.getLogger(__name__)

# ── Sandbox filesystem layout (SWE-Gym convention) ─────────────────────────

HOME = "/home/user"
TESTBED = "/testbed"
EVAL_LOG_FILE = f"{HOME}/logs/verifier/eval.log"

# Extension + grading script paths in sandbox.
EXTENSION_DIR = f"{HOME}/.pi/agent/extensions"
EXTENSION_PATH = f"{EXTENSION_DIR}/swe-answer.ts"
GRADE_SCRIPT_PATH = f"{HOME}/swe-grade.sh"
EVAL_SCRIPT_PATH = f"{HOME}/swe_eval.sh"
TEST_PATCH_PATH = f"{HOME}/swe_test.patch"

VERIFY_TIMEOUT_S = 300
SETUP_TIMEOUT_S = 600

# Source files for the answer extension and grading script.
_EXTENSIONS_DIR = Path(__file__).parent / "extensions"
_SWE_ANSWER_TS = _EXTENSIONS_DIR / "swe-answer.ts"
_SWE_GRADE_SH = _EXTENSIONS_DIR / "swe-grade.sh"


@dataclass
class SWEAgentConfig:
    """Minimal config for the CLI agent driver."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    agent_timeout_s: float = 600.0
    sandbox_home: str = HOME
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
    - ``verify()`` that grades host-side via ``swebench.harness.grading``
      (never trusts in-sandbox files for reward).
    - Falls back to running verify commands for legacy tasks.
    - SWE task metadata.

    **Reward integrity**: The training reward is always computed on the
    host.  The agent cannot influence it by writing files in the sandbox.
    """

    def __init__(
        self,
        *,
        swe_task: SWETask,
        verify_timeout_s: int = VERIFY_TIMEOUT_S,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._swe_task = swe_task
        self._verify_timeout_s = verify_timeout_s

    @property
    def swe_task(self) -> SWETask:
        return self._swe_task

    def initial_messages(self) -> list[Message]:
        """Return the SWE instruction as the initial prompt."""
        return [{"role": "user", "content": self._swe_task.instruction}]

    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        """Compute the training reward host-side.

        For SWE-Gym tasks (metadata has ``test_patch``, ``FAIL_TO_PASS``,
        etc.), the flow is:

        1. Revert any test files the agent may have modified back to
           ``base_commit`` (anti-reward-hacking).
        2. Apply the task's ``test_patch`` in the sandbox.
        3. Run the eval script in the sandbox, capturing the log.
        4. Revert the ``test_patch``.
        5. Download the log to the host.
        6. Grade on the host via ``swebench.harness.grading.get_eval_report()``.

        For legacy tasks with shell ``verify`` commands, falls back to
        running them and computing pass ratio.

        If neither path applies, reward defaults to 0.0.
        """
        # 1. Try host-side swebench grading (primary path).
        grade_result = self._host_side_grade()
        if grade_result is not None:
            return VerifyResult(
                env_reward=grade_result.reward,
                done=True,
                metrics={
                    "instance_id": self._swe_task.instance_id,
                    "reward_source": "host_swebench",
                    "resolved": grade_result.resolved,
                    "patch_applied": grade_result.patch_applied,
                },
                artifacts={
                    "task_id": self._swe_task.task_id,
                    "tests_status": grade_result.tests_status,
                },
            )

        # 2. Fallback: run verify commands (legacy tasks with shell commands).
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
                },
                artifacts={
                    "verify_details": verify_details,
                    "task_id": self._swe_task.task_id,
                },
            )

        # 3. No grading source available.
        return VerifyResult(
            env_reward=0.0,
            done=True,
            metrics={
                "instance_id": self._swe_task.instance_id,
                "reward_source": "default_no_grading",
            },
            artifacts={
                "task_id": self._swe_task.task_id,
            },
        )

    def _host_side_grade(self) -> Any:
        """Run swebench grading host-side.

        Returns a :class:`GradeResult` or ``None`` if the task lacks
        SWE-Gym metadata.
        """
        metadata = self._swe_task.metadata
        if not metadata:
            return None

        required_keys = {"patch", "test_patch", "FAIL_TO_PASS", "version"}
        if not required_keys.issubset(metadata.keys()):
            return None

        try:
            from .grading import grade_from_test_output
            from .models import SWEGymTask

            gym_task = SWEGymTask(
                instance_id=self._swe_task.instance_id,
                repo=self._swe_task.repo,
                base_commit=self._swe_task.base_commit,
                problem_statement=self._swe_task.instruction,
                version=str(metadata["version"]),
                patch=str(metadata["patch"]),
                test_patch=str(metadata["test_patch"]),
                FAIL_TO_PASS=list(metadata["FAIL_TO_PASS"]),
                PASS_TO_PASS=list(metadata.get("PASS_TO_PASS", [])),
                hints_text=str(metadata.get("hints_text", "")),
                created_at=str(metadata.get("created_at", "")),
                timeout_s=self._swe_task.timeout_s,
            )
        except Exception as exc:
            _log.warning("Could not reconstruct SWEGymTask: %s", exc)
            return None

        try:
            # Step 1: Revert test files to base_commit (anti-reward-hacking).
            self._revert_test_files(gym_task)

            # Step 2: Apply the known-good test_patch.
            self.sandbox.exec(
                f"cd {TESTBED} && git apply --allow-empty {TEST_PATCH_PATH}",
                timeout=30,
            )

            # Step 3: Run the eval script, capturing output.
            r = self.sandbox.exec(
                f"bash {EVAL_SCRIPT_PATH} 2>&1",
                cwd=TESTBED,
                timeout=self._verify_timeout_s,
            )
            test_output = (r.stdout or "") + "\n" + (r.stderr or "")

            # Step 4: Revert the test_patch.
            self.sandbox.exec(
                f"cd {TESTBED} && git apply --allow-empty -R {TEST_PATCH_PATH}",
                timeout=30,
            )

            # Step 5: Extract agent's patch for the grading report.
            diff_r = self.sandbox.exec(
                f"cd {TESTBED} && git diff HEAD",
                timeout=30,
            )
            model_patch = diff_r.stdout or ""

            # Step 6: Grade on host.
            # Prepend patch-applied marker expected by swebench grading.
            test_output = f">>>>> Applied Patch (pred)\n{test_output}"
            return grade_from_test_output(
                gym_task,
                test_output,
                model_patch=model_patch if model_patch.strip() else None,
            )

        except Exception as exc:
            _log.warning("Host-side grading failed: %s", exc)
            return None

    def _revert_test_files(self, gym_task: SWEGymTask) -> None:
        """Revert any test files the agent may have modified.

        Uses ``git checkout <base_commit> -- <path>`` for each file in
        the test_patch, then removes any new test files the agent added.
        This prevents reward hacking by weakening test assertions.
        """
        test_patch = gym_task.test_patch
        if not test_patch:
            return

        # Parse file paths from the test patch.
        modified_files: list[str] = []
        for line in test_patch.splitlines():
            if line.startswith("+++ b/"):
                path = line[6:]
                modified_files.append(path)
            elif line.startswith("--- a/"):
                path = line[6:]
                if path != "/dev/null":
                    modified_files.append(path)

        if not modified_files:
            return

        # Revert each test file to base_commit.
        base = gym_task.base_commit
        for path in set(modified_files):
            self.sandbox.exec(
                f"cd {TESTBED} && git checkout {base} -- {path} 2>/dev/null || true",
                timeout=10,
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
    """Creates isolated SWE sessions with Pi answer extension.

    Deploys the ``answer`` tool extension + grading script into each
    sandbox so Pi gets fast feedback on its submission.  The
    authoritative training reward is computed host-side by
    ``SWESession.verify()``.

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

        self._agent_name = agent
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
        """Create one SWE session with answer extension deployed.

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
                self._stage_repo(sandbox, swe_task)

            self._deploy_answer_extension(sandbox, swe_task)
            self._run_setup(sandbox, swe_task)

            agent_task = self._build_agent_task(swe_task)
            self._driver._bootstrap_sandbox(sandbox, agent_task, self._config)

        except Exception as exc:
            _log.error("SWESessionFactory.create: bootstrap failed: %r", exc)
            sandbox.kill()
            raise

        base_url_override: str | None = None
        interception_rollout_id: str | None = None
        interception_queue: asyncio.Queue | None = None

        if self._mode == "interception_gate":
            assert self._interception_server is not None
            assert self._interception_base_url is not None
            rollout_id = episode_id or f"rollout_{uuid.uuid4().hex[:8]}"
            interception_rollout_id = rollout_id
            interception_queue = self._interception_server.register_rollout(rollout_id)
            base_url_override = (
                f"{self._interception_base_url.rstrip('/')}/rollout/{rollout_id}/v1"
            )

        agent_task = self._build_agent_task(swe_task)
        agent_bg = self._driver._start_agent(
            sandbox, agent_task, self._config, base_url_override=base_url_override
        )

        return SWESession(
            swe_task=swe_task,
            verify_timeout_s=self._verify_timeout_s,
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

    # ── Bootstrap helpers ──────────────────────────────────────────────────

    def _stage_repo(self, sandbox: SandboxHandle, task: SWETask) -> None:
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

    def _deploy_answer_extension(
        self, sandbox: SandboxHandle, task: SWETask
    ) -> None:
        """Deploy the Pi answer extension and grading infrastructure.

        Writes into the sandbox:
          - swe-answer.ts  → ~/.pi/agent/extensions/  (Pi auto-discovers)
          - swe-grade.sh   → ~/swe-grade.sh
          - swe_eval.sh    → ~/swe_eval.sh   (from swebench, if available)
          - swe_test.patch → ~/swe_test.patch (test patch, if available)
        """
        sandbox.exec(
            f"mkdir -p {EXTENSION_DIR} {HOME}/logs/verifier {HOME}/logs/agent",
            timeout=10,
        )

        sandbox.write_text(EXTENSION_PATH, _SWE_ANSWER_TS.read_text())

        sandbox.write_text(GRADE_SCRIPT_PATH, _SWE_GRADE_SH.read_text())
        sandbox.exec(f"chmod +x {GRADE_SCRIPT_PATH}", timeout=5)

        test_patch = task.metadata.get("test_patch", "")
        if test_patch:
            sandbox.write_text(TEST_PATCH_PATH, test_patch)

        eval_script = self._generate_eval_script(task)
        if eval_script:
            sandbox.write_text(EVAL_SCRIPT_PATH, eval_script)
            sandbox.exec(f"chmod +x {EVAL_SCRIPT_PATH}", timeout=5)

        # Set environment variables for the grading script.
        env_file = f"{HOME}/.swe_env"
        env_content = "\n".join([
            f"export SWE_INSTANCE_ID={_shell_quote(task.instance_id)}",
            f"export SWE_TESTBED={TESTBED}",
            f"export SWE_TEST_PATCH={TEST_PATCH_PATH}",
            f"export SWE_EVAL_SCRIPT={EVAL_SCRIPT_PATH}",
            f"export SWE_LOG_FILE={EVAL_LOG_FILE}",
            f"export SWE_GRADE_SCRIPT={GRADE_SCRIPT_PATH}",
        ])
        sandbox.write_text(env_file, env_content)

        sandbox.exec(
            f'echo "source {env_file}" >> {HOME}/.bashrc',
            timeout=5,
        )

    def _generate_eval_script(self, task: SWETask) -> str | None:
        """Try to generate a swebench eval script for this task."""
        metadata = task.metadata
        required_keys = {"patch", "test_patch", "FAIL_TO_PASS", "version"}
        if not metadata or not required_keys.issubset(metadata.keys()):
            return None

        try:
            from .grading import make_eval_script
            from .models import SWEGymTask

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
            return make_eval_script(gym_task)
        except Exception as exc:
            _log.debug("Could not generate eval script: %s", exc)
            return None

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
        """Convert SWETask into the shape CLIAgentDriver expects."""
        return _SWEAgentTask(
            instruction=swe_task.instruction,
            setup_shell=None,
            metadata={
                "task_id": swe_task.task_id,
                "instance_id": swe_task.instance_id,
                "repo": swe_task.repo,
            },
        )


def _shell_quote(s: str) -> str:
    """Simple shell quoting for env var values."""
    return "'" + s.replace("'", "'\\''") + "'"


__all__ = [
    "SWEAgentConfig",
    "SWESession",
    "SWESessionFactory",
    "parse_terminal_call",
    # Filesystem constants (useful for tests).
    "EVAL_LOG_FILE",
    "EVAL_SCRIPT_PATH",
    "EXTENSION_DIR",
    "EXTENSION_PATH",
    "GRADE_SCRIPT_PATH",
    "HOME",
    "TEST_PATCH_PATH",
    "TESTBED",
]
