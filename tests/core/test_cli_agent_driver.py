# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the CLI agent driver abstraction (Phase 2).

Covers:
  - Agent spec + event protocols (base.py)
  - Agent registry (__init__.py)
  - CLIAgentDriver / CLIAgentSession / CLIAgentSessionFactory (cli_driver.py)
  - OpenCode adapter spec (opencode.py)

All tests run without external dependencies (no E2B, no LLM, no network).
"""

from __future__ import annotations

import asyncio
import json
import queue as _queue_mod
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from openenv.core.harness.sandbox.base import ExecResult, SandboxHandle


# Fake sandbox infrastructure (mirrors test_coding_agent_env.py pattern)


@dataclass
class FakeBgJob:
    cmd: str = ""
    envs: dict[str, str] | None = None
    _exit_code: int = 0

    @property
    def pid(self) -> int:
        return 12345

    def wait(self, timeout: float | None = None) -> int:
        return self._exit_code

    def kill(self) -> None:
        pass


class FakeSandbox:
    """In-memory sandbox for unit testing."""

    def __init__(
        self,
        *,
        install_check_succeeds: bool = False,
        healthz_succeeds: bool = True,
    ) -> None:
        self.sandbox_id = "fake-sandbox-001"
        self.written: dict[str, str] = {}
        self.executed: list[str] = []
        self.bg_commands: list[tuple[str, dict[str, str] | None]] = []
        self._install_check_succeeds = install_check_succeeds
        self._healthz_succeeds = healthz_succeeds
        self._killed = False

    def exec(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> ExecResult:
        self.executed.append(cmd)
        if cmd == "echo ok":
            return ExecResult(exit_code=0, stdout="ok", stderr="")
        # install check — only standalone version-check commands (short, just
        # binary + --version) should be treated as install probes. Multi-part
        # setup scripts that happen to end with --version should succeed.
        if "--version" in cmd and len(cmd) < 80 and "&&" not in cmd:
            if self._install_check_succeeds:
                return ExecResult(exit_code=0, stdout="1.0.0", stderr="")
            return ExecResult(exit_code=127, stdout="", stderr="not found")
        # healthz check
        if "healthz" in cmd:
            if self._healthz_succeeds:
                return ExecResult(exit_code=0, stdout='{"status":"ok"}', stderr="")
            return ExecResult(exit_code=7, stdout="", stderr="connection refused")
        # All other commands succeed
        return ExecResult(exit_code=0, stdout="", stderr="")

    def start_bg(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> FakeBgJob:
        self.bg_commands.append((cmd, envs))
        return FakeBgJob(cmd=cmd, envs=envs)

    def write_text(self, path: str, content: str) -> None:
        self.written[path] = content

    def read_text(self, path: str) -> str:
        if path not in self.written:
            raise FileNotFoundError(f"No such file: {path}")
        return self.written[path]

    def exists(self, path: str) -> bool:
        return path in self.written

    def kill(self) -> None:
        self._killed = True


class FakeSandboxBackend:
    """Backend that returns FakeSandbox instances."""

    def __init__(
        self,
        *,
        install_check_succeeds: bool = False,
        healthz_succeeds: bool = True,
    ) -> None:
        self._install_check_succeeds = install_check_succeeds
        self._healthz_succeeds = healthz_succeeds
        self.created: list[FakeSandbox] = []

    def create(
        self,
        *,
        timeout_s: int = 900,
        envs: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> SandboxHandle:
        sbx = FakeSandbox(
            install_check_succeeds=self._install_check_succeeds,
            healthz_succeeds=self._healthz_succeeds,
        )
        self.created.append(sbx)
        return sbx


@dataclass
class FakeTask:
    instruction: str = "Write hello.py"
    setup_shell: str | None = None
    upload_files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeConfig:
    base_url: str = "https://api.example.com/v1"
    api_key: str = "sk-test-key"
    model: str = "test-model"
    agent_timeout_s: float = 300.0
    sandbox_home: str = "/home/user"
    workdir: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)


# PR 2.1: Agent Spec and Event Parser Protocols


class TestAgentSpecProtocols:
    """Tests for base.py data models."""

    def test_mcp_config_spec_frozen(self):
        from openenv.core.harness.agents.base import MCPConfigSpec

        spec = MCPConfigSpec(method="config_file", path_template="{workdir}/mcp.json")
        assert spec.method == "config_file"
        assert spec.path_template == "{workdir}/mcp.json"
        with pytest.raises(AttributeError):
            spec.method = "cli_flags"  # type: ignore[misc]

    def test_artifact_spec_defaults(self):
        from openenv.core.harness.agents.base import ArtifactSpec

        a = ArtifactSpec(path="/logs/agent/out.log")
        assert a.format == "text"
        assert a.optional is True

    def test_artifact_spec_json(self):
        from openenv.core.harness.agents.base import ArtifactSpec

        a = ArtifactSpec(path="/data/traj.json", format="json", optional=False)
        assert a.format == "json"
        assert a.optional is False

    def test_agent_event_creation(self):
        from openenv.core.harness.agents.base import AgentEvent

        e = AgentEvent(
            type="tool_call", data={"name": "bash"}, raw='{"type":"tool_call"}'
        )
        assert e.type == "tool_call"
        assert e.data["name"] == "bash"

    def test_cli_agent_spec_minimal(self):
        from openenv.core.harness.agents.base import CLIAgentSpec, MCPConfigSpec

        spec = CLIAgentSpec(
            name="test-agent",
            install_check_cmd=["test-agent", "--version"],
            base_command=["test-agent", "run"],
            mcp_config=MCPConfigSpec(method="cli_flags"),
        )
        assert spec.name == "test-agent"
        assert spec.default_timeout_s == 600.0
        assert spec.setup is None
        assert spec.files is None
        assert spec.artifacts is None
        assert spec.env is None
        assert spec.extension_dir_template is None
        assert spec.build_command is None

    def test_cli_agent_spec_full(self):
        from openenv.core.harness.agents.base import (
            ArtifactSpec,
            CLIAgentSpec,
            MCPConfigSpec,
        )

        spec = CLIAgentSpec(
            name="full-agent",
            install_check_cmd=["full-agent", "--version"],
            base_command=["full-agent", "exec"],
            mcp_config=MCPConfigSpec(
                method="config_file", path_template="{workdir}/mcp.json"
            ),
            default_timeout_s=900.0,
            setup="npm install -g full-agent",
            files={
                "/task.txt": "hello",
                "/dynamic.txt": lambda task, config: task.instruction,
            },
            artifacts={
                "log": ArtifactSpec(path="/logs/out.log"),
                "traj": ArtifactSpec(path="/logs/traj.json", format="json"),
            },
            env={"API_KEY": "{api_key}", "MODEL": "{model}"},
            build_command=lambda spec, config, task, mcp: "full-agent exec",
            build_mcp_config=lambda spec, tools, workdir: "{}",
            parse_events=lambda line: None,
        )
        assert spec.name == "full-agent"
        assert spec.artifacts is not None
        assert len(spec.artifacts) == 2
        assert spec.files is not None
        assert callable(spec.files["/dynamic.txt"])


# PR 2.2: Agent Registry


class TestAgentRegistry:
    """Tests for the agent registry."""

    def test_register_and_lookup(self):
        from openenv.core.harness.agents import (
            get_agent_spec,
            list_agents,
            register_agent,
            unregister_agent,
        )
        from openenv.core.harness.agents.base import CLIAgentSpec, MCPConfigSpec

        spec = CLIAgentSpec(
            name="test-registry-agent",
            install_check_cmd=["tra", "--version"],
            base_command=["tra", "run"],
            mcp_config=MCPConfigSpec(method="cli_flags"),
        )
        try:
            register_agent(spec)
            assert "test-registry-agent" in list_agents()
            assert get_agent_spec("test-registry-agent") is spec
        finally:
            unregister_agent("test-registry-agent")

    def test_duplicate_registration_same_object_ok(self):
        from openenv.core.harness.agents import register_agent, unregister_agent
        from openenv.core.harness.agents.base import CLIAgentSpec, MCPConfigSpec

        spec = CLIAgentSpec(
            name="test-dup-ok",
            install_check_cmd=["x"],
            base_command=["x"],
            mcp_config=MCPConfigSpec(method="cli_flags"),
        )
        try:
            register_agent(spec)
            register_agent(spec)  # same object — should be fine
        finally:
            unregister_agent("test-dup-ok")

    def test_duplicate_registration_different_object_raises(self):
        from openenv.core.harness.agents import register_agent, unregister_agent
        from openenv.core.harness.agents.base import CLIAgentSpec, MCPConfigSpec

        spec1 = CLIAgentSpec(
            name="test-dup-fail",
            install_check_cmd=["x"],
            base_command=["x"],
            mcp_config=MCPConfigSpec(method="cli_flags"),
        )
        spec2 = CLIAgentSpec(
            name="test-dup-fail",
            install_check_cmd=["y"],
            base_command=["y"],
            mcp_config=MCPConfigSpec(method="cli_flags"),
        )
        try:
            register_agent(spec1)
            with pytest.raises(ValueError, match="already registered"):
                register_agent(spec2)
        finally:
            unregister_agent("test-dup-fail")

    def test_unknown_agent_raises_keyerror(self):
        from openenv.core.harness.agents import get_agent_spec

        with pytest.raises(KeyError, match="Unknown agent"):
            get_agent_spec("nonexistent-agent-xyz")

    def test_unregister_returns_spec(self):
        from openenv.core.harness.agents import register_agent, unregister_agent
        from openenv.core.harness.agents.base import CLIAgentSpec, MCPConfigSpec

        spec = CLIAgentSpec(
            name="test-unreg",
            install_check_cmd=["x"],
            base_command=["x"],
            mcp_config=MCPConfigSpec(method="cli_flags"),
        )
        register_agent(spec)
        removed = unregister_agent("test-unreg")
        assert removed is spec
        assert unregister_agent("test-unreg") is None

    def test_auto_import_opencode(self):
        """Auto-import triggers registration of built-in agents."""
        from openenv.core.harness.agents import get_agent_spec

        spec = get_agent_spec("opencode")
        assert spec.name == "opencode"


# PR 2.3: CLIAgentDriver / CLIAgentSession / CLIAgentSessionFactory


def _make_test_spec(**overrides: Any):
    from openenv.core.harness.agents.base import (
        ArtifactSpec,
        CLIAgentSpec,
        MCPConfigSpec,
    )

    defaults: dict[str, Any] = dict(
        name="test-agent",
        install_check_cmd=["test-agent", "--version"],
        base_command=["test-agent", "run", "--json"],
        mcp_config=MCPConfigSpec(
            method="config_file", path_template="{workdir}/mcp.json"
        ),
        setup="apt-get install -y test-agent",
        files={
            "/home/user/task/instruction.txt": lambda task, config: task.instruction,
        },
        artifacts={
            "agent_log": ArtifactSpec(path="/home/user/logs/agent.log"),
        },
        env={
            "API_KEY": "{api_key}",
            "BASE_URL": "{base_url}",
            "MODEL": "{model}",
        },
        build_command=lambda spec, config, task, mcp: (
            f"test-agent run --json '{task.instruction}' 2>&1 | tee /home/user/logs/agent.log"
        ),
        build_mcp_config=lambda spec, tools, workdir: json.dumps({"tools": []}),
        parse_events=lambda line: None,
    )
    defaults.update(overrides)
    return CLIAgentSpec(**defaults)


class TestCLIAgentDriver:
    """Tests for the shared CLI agent driver."""

    def test_create_session_full_lifecycle(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec()
        backend = FakeSandboxBackend()
        driver = CLIAgentDriver(spec=spec, sandbox_backend=backend, mode="black_box")

        task = FakeTask(instruction="Write hello.py")
        config = FakeConfig()
        session = driver.create_session(task=task, config=config)

        # Verify sandbox was created
        assert len(backend.created) == 1
        sbx = backend.created[0]

        # Verify sandbox readiness was probed
        assert "echo ok" in sbx.executed

        # Verify install was attempted (agent not pre-installed)
        assert any("apt-get install" in cmd for cmd in sbx.executed)

        # Verify files were uploaded
        assert "/home/user/task/instruction.txt" in sbx.written
        assert sbx.written["/home/user/task/instruction.txt"] == "Write hello.py"

        # Verify MCP config was written
        assert "/home/user/workdir/mcp.json" in sbx.written

        # Verify agent was launched as bg process
        assert len(sbx.bg_commands) == 1
        bg_cmd, bg_envs = sbx.bg_commands[0]
        assert "test-agent run" in bg_cmd

        # Verify env vars were resolved
        assert bg_envs is not None
        assert bg_envs["API_KEY"] == "sk-test-key"
        assert bg_envs["BASE_URL"] == "https://api.example.com/v1"
        assert bg_envs["MODEL"] == "test-model"

        # Session API
        assert session.initial_messages() == [
            {"role": "user", "content": "Write hello.py"}
        ]
        assert session.list_tools() == []
        assert session.call_tool("x", {}).error is not None
        assert session.wait_for_completion() == 0

        session.close()
        assert sbx._killed

    def test_create_session_honors_configured_workdir_for_mcp_file(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec()
        backend = FakeSandboxBackend()
        driver = CLIAgentDriver(spec=spec, sandbox_backend=backend, mode="black_box")

        config = FakeConfig(workdir="/testbed")
        session = driver.create_session(task=FakeTask(), config=config)

        sbx = backend.created[0]
        assert "/testbed/mcp.json" in sbx.written
        session.close()

    def test_create_session_creates_extension_dir_when_spec_declares_one(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec(extension_dir_template="{home}/.agent/extensions")
        backend = FakeSandboxBackend()
        driver = CLIAgentDriver(spec=spec, sandbox_backend=backend, mode="black_box")

        session = driver.create_session(task=FakeTask(), config=FakeConfig())
        sbx = backend.created[0]
        assert any(
            cmd.startswith("mkdir -p /home/user/.agent/extensions")
            for cmd in sbx.executed
        )
        session.close()

    def test_create_session_skips_install_when_prebaked(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec()
        backend = FakeSandboxBackend(install_check_succeeds=True)
        driver = CLIAgentDriver(spec=spec, sandbox_backend=backend, mode="black_box")

        session = driver.create_session(
            task=FakeTask(),
            config=FakeConfig(),
        )

        sbx = backend.created[0]
        # install should have been skipped
        assert not any("apt-get install" in cmd for cmd in sbx.executed)
        session.close()

    def test_create_session_interception_gate_requires_server(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec()
        with pytest.raises(ValueError, match="InterceptionServer"):
            CLIAgentDriver(
                spec=spec,
                sandbox_backend=FakeSandboxBackend(),
                mode="interception_gate",
            )

    def test_create_session_uploads_task_files(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec()
        backend = FakeSandboxBackend()
        driver = CLIAgentDriver(spec=spec, sandbox_backend=backend, mode="black_box")

        task = FakeTask(
            instruction="Write code",
            upload_files={"/extra/data.json": '{"key": "value"}'},
        )
        session = driver.create_session(task=task, config=FakeConfig())

        sbx = backend.created[0]
        assert sbx.written["/extra/data.json"] == '{"key": "value"}'
        session.close()

    def test_opencode_black_box_api_key_stays_out_of_command_argv(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        secret = "sk-test '$(leak)"
        config = FakeConfig(api_key=secret)
        backend = FakeSandboxBackend()
        driver = CLIAgentDriver(
            spec=OPENCODE_SPEC,
            sandbox_backend=backend,
            mode="black_box",
        )

        session = driver.create_session(task=FakeTask(), config=config)
        sbx = backend.created[0]
        cmd, envs = sbx.bg_commands[-1]
        assert secret not in cmd
        assert envs is not None
        assert envs["OPENAI_API_KEY"] == secret
        session.close()

    def test_opencode_interception_gate_uses_server_secret_not_user_key(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver
        from openenv.core.harness.agents.interception_server import InterceptionServer
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        secret = "sk-test '$(leak)"
        config = FakeConfig(api_key=secret)
        backend = FakeSandboxBackend()
        server = InterceptionServer(port=0, secret="gate-secret")
        driver = CLIAgentDriver(
            spec=OPENCODE_SPEC,
            sandbox_backend=backend,
            mode="interception_gate",
            interception_server=server,
            interception_base_url="http://127.0.0.1:8765",
        )

        session = driver.create_session(task=FakeTask(), config=config)
        sbx = backend.created[0]
        cmd, envs = sbx.bg_commands[-1]
        assert secret not in cmd
        assert envs is not None
        assert envs["OPENAI_API_KEY"] == "gate-secret"
        session.close()

    def test_pi_interception_gate_writes_models_json_and_uses_openenv_provider(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver
        from openenv.core.harness.agents.interception_server import InterceptionServer
        from openenv.core.harness.agents.pi import PI_SPEC

        backend = FakeSandboxBackend()
        server = InterceptionServer(port=0, secret="gate-secret")
        driver = CLIAgentDriver(
            spec=PI_SPEC,
            sandbox_backend=backend,
            mode="interception_gate",
            interception_server=server,
            interception_base_url="http://127.0.0.1:8765",
        )

        session = driver.create_session(task=FakeTask(), config=FakeConfig())
        sbx = backend.created[0]

        # Command should force the custom provider backed by models.json.
        cmd, _envs = sbx.bg_commands[-1]
        assert "--provider openenv" in cmd

        home_models = "/home/user/.pi/agent/models.json"
        root_models = "/root/.pi/agent/models.json"
        assert home_models in sbx.written
        # /root/ path is only written when sandbox_home == "/root"
        assert root_models not in sbx.written

        cfg = json.loads(sbx.written[home_models])
        provider = cfg["providers"]["openenv"]
        assert provider["api"] == "openai-completions"
        assert provider["apiKey"] == "gate-secret"
        assert provider["models"][0]["id"] == "test-model"
        assert "/rollout/" in provider["baseUrl"]
        assert provider["baseUrl"].endswith("/v1")

        session.close()

    def test_create_session_runs_task_setup_shell(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec()
        backend = FakeSandboxBackend()
        driver = CLIAgentDriver(spec=spec, sandbox_backend=backend, mode="black_box")

        task = FakeTask(
            instruction="Write code",
            setup_shell="pip install pandas",
        )
        session = driver.create_session(task=task, config=FakeConfig())

        sbx = backend.created[0]
        assert "pip install pandas" in sbx.executed
        session.close()

    def test_create_session_with_verifier(self):
        from openenv.core.harness import VerifyResult
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec()
        backend = FakeSandboxBackend()
        driver = CLIAgentDriver(spec=spec, sandbox_backend=backend, mode="black_box")

        def verifier(sandbox, task):
            return VerifyResult(env_reward=1.0, done=True, metrics={"correct": True})

        session = driver.create_session(
            task=FakeTask(),
            config=FakeConfig(),
            verifier=verifier,
        )

        result = session.verify([])
        assert result.env_reward == 1.0
        assert result.metrics["correct"] is True
        session.close()

    def test_session_verify_without_verifier(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec()
        backend = FakeSandboxBackend()
        driver = CLIAgentDriver(spec=spec, sandbox_backend=backend, mode="black_box")

        session = driver.create_session(task=FakeTask(), config=FakeConfig())

        result = session.verify([])
        assert result.env_reward is None
        assert result.done is True
        session.close()

    def test_invalid_mode_raises(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec()
        with pytest.raises(ValueError, match="Unknown mode"):
            CLIAgentDriver(
                spec=spec,
                sandbox_backend=FakeSandboxBackend(),
                mode="invalid",  # type: ignore[arg-type]
            )


class TestCLIAgentSession:
    """Tests for CLIAgentSession."""

    def test_collect_artifacts_text(self):
        from openenv.core.harness.agents.base import ArtifactSpec
        from openenv.core.harness.agents.cli_driver import CLIAgentSession

        spec = _make_test_spec(
            artifacts={
                "log": ArtifactSpec(path="/logs/out.log"),
            },
        )
        sbx = FakeSandbox()
        sbx.written["/logs/out.log"] = "line1\nline2\n"

        session = CLIAgentSession(
            spec=spec,
            sandbox=sbx,
            task=FakeTask(),
            config=FakeConfig(),
        )
        arts = session.collect_artifacts()
        assert arts["log"] == "line1\nline2\n"

    def test_collect_artifacts_json(self):
        from openenv.core.harness.agents.base import ArtifactSpec
        from openenv.core.harness.agents.cli_driver import CLIAgentSession

        spec = _make_test_spec(
            artifacts={
                "traj": ArtifactSpec(path="/logs/traj.json", format="json"),
            },
        )
        sbx = FakeSandbox()
        sbx.written["/logs/traj.json"] = json.dumps({"steps": [1, 2, 3]})

        session = CLIAgentSession(
            spec=spec,
            sandbox=sbx,
            task=FakeTask(),
            config=FakeConfig(),
        )
        arts = session.collect_artifacts()
        assert arts["traj"] == {"steps": [1, 2, 3]}

    def test_collect_artifacts_jsonl(self):
        from openenv.core.harness.agents.base import ArtifactSpec
        from openenv.core.harness.agents.cli_driver import CLIAgentSession

        spec = _make_test_spec(
            artifacts={
                "events": ArtifactSpec(path="/logs/events.jsonl", format="jsonl"),
            },
        )
        sbx = FakeSandbox()
        sbx.written["/logs/events.jsonl"] = '{"a":1}\n{"b":2}\n'

        session = CLIAgentSession(
            spec=spec,
            sandbox=sbx,
            task=FakeTask(),
            config=FakeConfig(),
        )
        arts = session.collect_artifacts()
        assert arts["events"] == [{"a": 1}, {"b": 2}]

    def test_collect_artifacts_missing_optional(self):
        from openenv.core.harness.agents.base import ArtifactSpec
        from openenv.core.harness.agents.cli_driver import CLIAgentSession

        spec = _make_test_spec(
            artifacts={
                "log": ArtifactSpec(path="/missing/file.log", optional=True),
            },
        )
        sbx = FakeSandbox()
        session = CLIAgentSession(
            spec=spec,
            sandbox=sbx,
            task=FakeTask(),
            config=FakeConfig(),
        )
        arts = session.collect_artifacts()
        assert "log" not in arts

    def test_collect_artifacts_missing_required_raises(self):
        from openenv.core.harness.agents.base import ArtifactSpec
        from openenv.core.harness.agents.cli_driver import CLIAgentSession

        spec = _make_test_spec(
            artifacts={
                "log": ArtifactSpec(path="/missing/file.log", optional=False),
            },
        )
        sbx = FakeSandbox()
        session = CLIAgentSession(
            spec=spec,
            sandbox=sbx,
            task=FakeTask(),
            config=FakeConfig(),
        )
        with pytest.raises(FileNotFoundError):
            session.collect_artifacts()

    def test_close_kills_sandbox_and_jobs(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentSession

        spec = _make_test_spec()
        sbx = FakeSandbox()
        agent_job = FakeBgJob()

        session = CLIAgentSession(
            spec=spec,
            sandbox=sbx,
            task=FakeTask(),
            config=FakeConfig(),
            agent_bg_job=agent_job,
        )
        session.close()
        assert sbx._killed
        assert session._agent_bg_job is None

    @pytest.mark.asyncio
    async def test_next_request_handles_missing_intercept_without_keyerror(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentSession
        from openenv.core.harness.agents.interception_server import InterceptionServer

        spec = _make_test_spec()
        sbx = FakeSandbox()
        q: _queue_mod.Queue[str] = _queue_mod.Queue()
        q.put("req_missing")

        session = CLIAgentSession(
            spec=spec,
            sandbox=sbx,
            task=FakeTask(),
            config=FakeConfig(),
            agent_bg_job=FakeBgJob(),
            interception_server=InterceptionServer(secret="s"),
            interception_rollout_id="rollout-1",
            interception_queue=q,
        )

        # Missing request IDs can happen if unregister_rollout races with queue.get().
        assert await session.next_request(timeout_s=0.2) is None

    def test_next_request_soak_cross_loop_queue_get(self):
        """Soak test cross-loop request dequeueing via queue.Queue.

        Exercises the worker pattern that used to be unsafe with asyncio.Queue:
        repeatedly call next_request() from fresh event loops (asyncio.run)
        while request IDs are pushed from another thread.
        """
        from openenv.core.harness.agents.cli_driver import CLIAgentSession
        from openenv.core.harness.agents.interception_server import InterceptionServer

        spec = _make_test_spec()
        sbx = FakeSandbox()
        server = InterceptionServer(secret="s")
        request_queue = server.register_rollout("rollout-soak")

        session = CLIAgentSession(
            spec=spec,
            sandbox=sbx,
            task=FakeTask(),
            config=FakeConfig(),
            interception_server=server,
            interception_rollout_id="rollout-soak",
            interception_queue=request_queue,
        )

        total_requests = 200
        consumed: list[str] = []
        failures: list[BaseException] = []

        def _consumer() -> None:
            try:
                for _ in range(total_requests):
                    intercept = asyncio.run(session.next_request(timeout_s=2.0))
                    assert intercept is not None
                    request_id = intercept["request_id"]
                    consumed.append(request_id)
                    with server._state_lock:
                        server.intercepts.pop(request_id, None)
            except BaseException as exc:  # pragma: no cover - assertion path
                failures.append(exc)

        def _producer() -> None:
            try:
                for i in range(total_requests):
                    request_id = f"req_soak_{i:04d}"
                    with server._state_lock:
                        server.intercepts[request_id] = {
                            "request_id": request_id,
                            "messages": [{"role": "user", "content": "ping"}],
                        }
                    request_queue.put_nowait(request_id)
                    if i % 10 == 0:
                        time.sleep(0.001)
            except BaseException as exc:  # pragma: no cover - unexpected
                failures.append(exc)

        consumer_t = threading.Thread(target=_consumer, name="soak-consumer")
        producer_t = threading.Thread(target=_producer, name="soak-producer")

        consumer_t.start()
        producer_t.start()

        producer_t.join(timeout=10)
        consumer_t.join(timeout=15)

        assert not producer_t.is_alive(), "producer thread hung"
        assert not consumer_t.is_alive(), "consumer thread hung"
        assert not failures
        assert len(consumed) == total_requests
        assert len(set(consumed)) == total_requests

        session.close()


class TestCLIAgentSessionFactory:
    """Tests for the ResourceSessionFactory wrapper."""

    def test_factory_creates_sessions(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentSessionFactory

        spec = _make_test_spec()
        backend = FakeSandboxBackend()

        factory = CLIAgentSessionFactory(
            spec=spec,
            config=FakeConfig(),
            sandbox_backend=backend,
            mode="black_box",
        )

        session = factory.create(task=FakeTask())
        assert len(backend.created) == 1
        assert session.initial_messages()[0]["content"] == "Write hello.py"
        session.close()

    def test_factory_with_verifier(self):
        from openenv.core.harness import VerifyResult
        from openenv.core.harness.agents.cli_driver import CLIAgentSessionFactory

        spec = _make_test_spec()
        backend = FakeSandboxBackend()

        def verifier(sandbox, task):
            return VerifyResult(env_reward=0.5, done=True)

        factory = CLIAgentSessionFactory(
            spec=spec,
            config=FakeConfig(),
            sandbox_backend=backend,
            mode="black_box",
            verifier=verifier,
        )

        session = factory.create(task=FakeTask())
        result = session.verify([])
        assert result.env_reward == 0.5
        session.close()

    def test_factory_implements_resource_session_factory(self):
        from openenv.core.harness import ResourceSessionFactory
        from openenv.core.harness.agents.cli_driver import CLIAgentSessionFactory

        assert issubclass(CLIAgentSessionFactory, ResourceSessionFactory)

    def test_session_implements_resource_session(self):
        from openenv.core.harness import ResourceSession
        from openenv.core.harness.agents.cli_driver import CLIAgentSession

        assert issubclass(CLIAgentSession, ResourceSession)


# PR 2.4: OpenCode Adapter Spec


class TestOpenCodeSpec:
    """Tests for the OpenCode declarative spec."""

    def test_spec_is_registered(self):
        from openenv.core.harness.agents import get_agent_spec

        spec = get_agent_spec("opencode")
        assert spec.name == "opencode"

    def test_spec_fields(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        assert OPENCODE_SPEC.name == "opencode"
        assert OPENCODE_SPEC.install_check_cmd == [
            "/home/user/.opencode/bin/opencode",
            "--version",
        ]
        assert OPENCODE_SPEC.default_timeout_s == 900.0
        assert OPENCODE_SPEC.mcp_config.method == "config_file"
        assert OPENCODE_SPEC.mcp_config.path_template is not None
        assert "{home}" in OPENCODE_SPEC.mcp_config.path_template
        assert OPENCODE_SPEC.artifacts is not None
        assert "agent_log" in OPENCODE_SPEC.artifacts
        assert OPENCODE_SPEC.artifacts["agent_log"].format == "jsonl"

    def test_build_command(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        @dataclass
        class OcConfig:
            sandbox_home: str = "/home/user"
            run_format: str = "json"

        assert OPENCODE_SPEC.build_command is not None
        cmd = OPENCODE_SPEC.build_command(
            OPENCODE_SPEC,
            OcConfig(),
            FakeTask(instruction="Write hello.py"),
            None,
        )
        assert "opencode run" in cmd
        assert "--format json" in cmd
        assert "/home/user/task/instruction.md" in cmd

    def test_build_command_quotes_paths(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        @dataclass
        class OcConfig:
            sandbox_home: str = "/home/user with space"
            run_format: str = "json"

        assert OPENCODE_SPEC.build_command is not None
        cmd = OPENCODE_SPEC.build_command(
            OPENCODE_SPEC,
            OcConfig(),
            FakeTask(instruction="Write hello.py"),
            None,
        )
        assert "cd '/home/user with space/workdir'" in cmd
        assert "cat '/home/user with space/task/instruction.md'" in cmd
        assert "tee '/home/user with space/logs/agent/opencode.jsonl'" in cmd

    def test_build_mcp_config(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        assert OPENCODE_SPEC.build_mcp_config is not None
        config_str = OPENCODE_SPEC.build_mcp_config(
            OPENCODE_SPEC,
            [],
            "/home/user/workdir",
        )
        # OpenCode returns empty string because the config is written
        # via spec.files using _build_opencode_config_file instead.
        assert config_str == ""

    def test_parse_events_assistant(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        assert OPENCODE_SPEC.parse_events is not None
        line = json.dumps({"type": "assistant", "content": "hello"})
        event = OPENCODE_SPEC.parse_events(line)
        assert event is not None
        assert event.type == "assistant"

    def test_parse_events_tool_call(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        assert OPENCODE_SPEC.parse_events is not None
        line = json.dumps({"type": "tool_call", "name": "bash", "args": {}})
        event = OPENCODE_SPEC.parse_events(line)
        assert event is not None
        assert event.type == "tool_call"

    def test_parse_events_error(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        assert OPENCODE_SPEC.parse_events is not None
        line = json.dumps({"type": "error", "message": "boom"})
        event = OPENCODE_SPEC.parse_events(line)
        assert event is not None
        assert event.type == "error"

    def test_parse_events_done(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        assert OPENCODE_SPEC.parse_events is not None
        line = json.dumps({"type": "done"})
        event = OPENCODE_SPEC.parse_events(line)
        assert event is not None
        assert event.type == "done"

    def test_parse_events_invalid_json(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        assert OPENCODE_SPEC.parse_events is not None
        assert OPENCODE_SPEC.parse_events("not json") is None
        assert OPENCODE_SPEC.parse_events("") is None

    def test_build_env_vars(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        config = FakeConfig()
        config.extra_env = {"EXTRA": "val"}
        assert OPENCODE_SPEC.build_env_vars is not None
        envs = OPENCODE_SPEC.build_env_vars(OPENCODE_SPEC, config)
        assert envs["OPENAI_BASE_URL"] == "https://api.example.com/v1"
        assert envs["OPENAI_API_KEY"] == "sk-test-key"
        assert envs["OPENCODE_CONFIG"] == "/home/user/.config/opencode/opencode.json"
        assert envs["EXTRA"] == "val"

    def test_files_instruction_resolver(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        task = FakeTask(instruction="Build a REST API")
        config = FakeConfig()
        assert OPENCODE_SPEC.files is not None
        instruction_fn = OPENCODE_SPEC.files["/home/user/task/instruction.md"]
        assert callable(instruction_fn)
        assert instruction_fn(task, config) == "Build a REST API"

    def test_files_system_prompt_resolver(self):
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        task = FakeTask()
        config = FakeConfig()
        assert OPENCODE_SPEC.files is not None
        system_fn = OPENCODE_SPEC.files["/home/user/task/system.md"]
        assert callable(system_fn)
        # No system prompt on FakeConfig → returns None
        assert system_fn(task, config) is None

    def test_opencode_driver_integration(self):
        """End-to-end: create a session using the OpenCode spec via the driver."""
        from openenv.core.harness.agents.cli_driver import CLIAgentSessionFactory
        from openenv.core.harness.agents.opencode import OPENCODE_SPEC

        backend = FakeSandboxBackend()
        factory = CLIAgentSessionFactory(
            spec=OPENCODE_SPEC,
            config=FakeConfig(),
            sandbox_backend=backend,
            mode="black_box",
        )

        session = factory.create(task=FakeTask(instruction="Hello"))
        assert session.spec.name == "opencode"
        assert session.initial_messages()[0]["content"] == "Hello"

        sbx = backend.created[0]
        # Instruction file should have been written
        assert sbx.written.get("/home/user/task/instruction.md") == "Hello"

        session.close()


class TestPiSpec:
    def test_build_command_quotes_paths(self):
        from openenv.core.harness.agents.pi import PI_SPEC

        @dataclass
        class PiConfig:
            sandbox_home: str = "/home/user with space"
            provider: str = "openai"
            model: str = "model/name"
            thinking: str = "off"

        assert PI_SPEC.build_command is not None
        cmd = PI_SPEC.build_command(
            PI_SPEC,
            PiConfig(),
            FakeTask(instruction="Write hello.py"),
            None,
        )
        assert "cd '/home/user with space/workdir'" in cmd
        assert "-p @'/home/user with space/task/instruction.txt'" in cmd
        assert "tee '/home/user with space/logs/agent/pi.txt'" in cmd

    def test_build_command_uses_config_workdir_when_present(self):
        from openenv.core.harness.agents.pi import PI_SPEC

        @dataclass
        class PiConfig:
            sandbox_home: str = "/home/user"
            workdir: str = "/testbed"
            provider: str = "openai"
            model: str = "model/name"
            thinking: str = "off"

        assert PI_SPEC.build_command is not None
        cmd = PI_SPEC.build_command(
            PI_SPEC,
            PiConfig(),
            FakeTask(instruction="Write hello.py"),
            None,
        )
        assert "cd /testbed" in cmd

    def test_spec_declares_extension_dir_template(self):
        from openenv.core.harness.agents.pi import PI_SPEC

        assert PI_SPEC.extension_dir_template == "{home}/.pi/agent/extensions"


# Env var resolution


class TestEnvVarResolution:
    """Tests for environment variable placeholder resolution."""

    def test_resolve_placeholders(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec(
            env={
                "KEY": "{api_key}",
                "URL": "{base_url}",
                "MDL": "{model}",
                "STATIC": "fixed_value",
            },
            build_env_vars=None,  # use placeholder resolution
        )
        driver = CLIAgentDriver(
            spec=spec,
            sandbox_backend=FakeSandboxBackend(),
            mode="black_box",
        )
        envs = driver._resolve_env_vars(FakeConfig())
        assert envs["KEY"] == "sk-test-key"
        assert envs["URL"] == "https://api.example.com/v1"
        assert envs["MDL"] == "test-model"
        assert envs["STATIC"] == "fixed_value"

    def test_resolve_with_proxy_override(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec(
            env={"URL": "{base_url}"},
            build_env_vars=None,
        )
        driver = CLIAgentDriver(
            spec=spec,
            sandbox_backend=FakeSandboxBackend(),
            mode="black_box",
        )
        envs = driver._resolve_env_vars(
            FakeConfig(),
            base_url_override="http://127.0.0.1:7000/v1",
        )
        assert envs["URL"] == "http://127.0.0.1:7000/v1"

    def test_build_env_vars_hook_takes_precedence(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        def custom_env(spec, config):
            return {"CUSTOM": "yes", "MODEL": config.model}

        spec = _make_test_spec(
            env={"SHOULD_NOT": "appear"},
            build_env_vars=custom_env,
        )
        driver = CLIAgentDriver(
            spec=spec,
            sandbox_backend=FakeSandboxBackend(),
            mode="black_box",
        )
        envs = driver._resolve_env_vars(FakeConfig())
        assert envs == {"CUSTOM": "yes", "MODEL": "test-model"}
        assert "SHOULD_NOT" not in envs

    def test_empty_env_dict(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec(env=None, build_env_vars=None)
        driver = CLIAgentDriver(
            spec=spec,
            sandbox_backend=FakeSandboxBackend(),
            mode="black_box",
        )
        envs = driver._resolve_env_vars(FakeConfig())
        assert envs == {}


# Multiple setup commands


class TestMultiStepSetup:
    """Tests for specs with multi-step setup commands."""

    def test_list_of_setup_commands(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec(
            setup=[
                "apt-get update",
                "apt-get install -y nodejs",
                "npm install -g test-agent",
            ],
        )
        backend = FakeSandboxBackend()
        driver = CLIAgentDriver(spec=spec, sandbox_backend=backend, mode="black_box")

        session = driver.create_session(task=FakeTask(), config=FakeConfig())
        sbx = backend.created[0]

        # All three setup commands should have been executed
        assert any("apt-get update" in cmd for cmd in sbx.executed)
        assert any("apt-get install" in cmd for cmd in sbx.executed)
        assert any("npm install" in cmd for cmd in sbx.executed)
        session.close()

    def test_no_setup_and_not_installed_raises(self):
        from openenv.core.harness.agents.cli_driver import CLIAgentDriver

        spec = _make_test_spec(setup=None)
        backend = FakeSandboxBackend(install_check_succeeds=False)
        driver = CLIAgentDriver(spec=spec, sandbox_backend=backend, mode="black_box")

        with pytest.raises(RuntimeError, match="not installed"):
            driver.create_session(task=FakeTask(), config=FakeConfig())
