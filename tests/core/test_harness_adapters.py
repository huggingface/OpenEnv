# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for currently implemented harness adapters (OpenCode + Pi)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class FakeTask:
    instruction: str = "Write hello.py"
    setup_shell: str | None = None
    upload_files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeConfig:
    base_url: str = "https://api.example.com/v1"
    api_key: str = "sk-test"
    model: str = "test-model"
    agent_timeout_s: float = 300.0
    sandbox_home: str = "/home/user"
    system_prompt: str | None = None


class TestPiSpec:
    def test_registered(self):
        from openenv.core.harness.agents import get_agent_spec

        spec = get_agent_spec("pi")
        assert spec.name == "pi"

    def test_fields(self):
        from openenv.core.harness.agents.pi import PI_SPEC

        assert PI_SPEC.install_check_cmd == ["pi", "--version"]
        assert PI_SPEC.mcp_config.method == "config_file"
        assert PI_SPEC.mcp_config.path_template is not None
        assert ".mcp.json" in PI_SPEC.mcp_config.path_template
        assert PI_SPEC.build_env_vars is not None

    def test_build_env_vars_provider_specific_api_key(self):
        from openenv.core.harness.agents.pi import PI_SPEC

        @dataclass
        class PiConfig:
            provider: str
            api_key: str = "secret"
            base_url: str = "https://api.example.com/v1"
            extra_env: dict[str, str] = field(default_factory=dict)

        assert PI_SPEC.build_env_vars is not None

        hf_env = PI_SPEC.build_env_vars(PI_SPEC, PiConfig(provider="huggingface"))
        assert hf_env["HF_TOKEN"] == "secret"
        assert "OPENAI_API_KEY" not in hf_env

        oa_env = PI_SPEC.build_env_vars(PI_SPEC, PiConfig(provider="openai"))
        assert oa_env["OPENAI_API_KEY"] == "secret"
        assert "HF_TOKEN" not in oa_env

    def test_build_env_vars_rejects_unknown_provider(self):
        from openenv.core.harness.agents.pi import PI_SPEC

        @dataclass
        class PiConfig:
            provider: str = "unknown"
            api_key: str = "secret"
            base_url: str = "https://api.example.com/v1"
            extra_env: dict[str, str] = field(default_factory=dict)

        assert PI_SPEC.build_env_vars is not None
        with pytest.raises(ValueError, match="Unsupported pi provider"):
            PI_SPEC.build_env_vars(PI_SPEC, PiConfig())

    def test_build_command(self):
        from openenv.core.harness.agents.pi import PI_SPEC

        assert PI_SPEC.build_command is not None
        cmd = PI_SPEC.build_command(PI_SPEC, FakeConfig(), FakeTask(), None)
        assert "pi --no-session" in cmd
        assert "--no-context-files" in cmd

    def test_build_mcp_config(self):
        from openenv.core.harness.agents.pi import PI_SPEC

        assert PI_SPEC.build_mcp_config is not None
        content = PI_SPEC.build_mcp_config(PI_SPEC, [], "/workdir")
        assert "mcpServers" in json.loads(content)


class TestOpenCodeSpec:
    def test_registered(self):
        from openenv.core.harness.agents import get_agent_spec

        spec = get_agent_spec("opencode")
        assert spec.name == "opencode"


class TestRegistryAutoImport:
    @pytest.mark.parametrize("name", ["pi", "opencode"])
    def test_auto_import(self, name):
        from openenv.core.harness.agents import get_agent_spec

        spec = get_agent_spec(name)
        assert spec.name == name

    def test_list_agents_includes_current(self):
        import openenv.core.harness.agents.opencode  # noqa: F401
        import openenv.core.harness.agents.pi  # noqa: F401
        from openenv.core.harness.agents import list_agents

        agents = list_agents()
        for name in ["opencode", "pi"]:
            assert name in agents, f"{name} not in {agents}"
