# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for harness foundation types (RFC 005)."""

import pytest
from pydantic import ValidationError


class TestHarnessTransport:
    """Tests for HarnessTransport enum."""

    def test_stdio_value(self):
        from openenv.core.harnesses import HarnessTransport

        assert HarnessTransport.STDIO == "stdio"

    def test_http_value(self):
        from openenv.core.harnesses import HarnessTransport

        assert HarnessTransport.STREAMABLE_HTTP == "http"

    def test_mcp_value(self):
        from openenv.core.harnesses import HarnessTransport

        assert HarnessTransport.MCP == "mcp"

    def test_is_string_enum(self):
        from openenv.core.harnesses import HarnessTransport

        assert isinstance(HarnessTransport.STDIO, str)


class TestHarnessEventType:
    """Tests for HarnessEventType enum."""

    def test_all_event_types_exist(self):
        from openenv.core.harnesses import HarnessEventType

        expected = [
            "llm_request",
            "llm_response",
            "llm_chunk",
            "tool_call",
            "tool_result",
            "text_output",
            "error",
            "turn_complete",
        ]
        for value in expected:
            assert HarnessEventType(value) is not None

    def test_is_string_enum(self):
        from openenv.core.harnesses import HarnessEventType

        assert isinstance(HarnessEventType.TOOL_CALL, str)


class TestHarnessEvent:
    """Tests for HarnessEvent Pydantic model."""

    def test_creation_with_required_fields(self):
        from openenv.core.harnesses import HarnessEvent, HarnessEventType

        event = HarnessEvent(
            type=HarnessEventType.TOOL_CALL,
            timestamp=1234567890.0,
        )
        assert event.type == HarnessEventType.TOOL_CALL
        assert event.timestamp == 1234567890.0
        assert event.data == {}

    def test_creation_with_data(self):
        from openenv.core.harnesses import HarnessEvent, HarnessEventType

        event = HarnessEvent(
            type=HarnessEventType.TOOL_CALL,
            timestamp=1234567890.0,
            data={"tool_name": "read_file", "arguments": {"path": "/tmp/test"}},
        )
        assert event.data["tool_name"] == "read_file"

    def test_serialization_roundtrip(self):
        from openenv.core.harnesses import HarnessEvent, HarnessEventType

        event = HarnessEvent(
            type=HarnessEventType.TEXT_OUTPUT,
            timestamp=100.0,
            data={"text": "hello"},
        )
        json_str = event.model_dump_json()
        restored = HarnessEvent.model_validate_json(json_str)
        assert restored.type == event.type
        assert restored.data == event.data

    def test_requires_type(self):
        from openenv.core.harnesses import HarnessEvent

        with pytest.raises(ValidationError):
            HarnessEvent(timestamp=100.0)

    def test_requires_timestamp(self):
        from openenv.core.harnesses import HarnessEvent, HarnessEventType

        with pytest.raises(ValidationError):
            HarnessEvent(type=HarnessEventType.ERROR)


class TestHarnessResponse:
    """Tests for HarnessResponse Pydantic model."""

    def test_creation_minimal(self):
        from openenv.core.harnesses import HarnessResponse

        resp = HarnessResponse(response="Task complete.")
        assert resp.response == "Task complete."
        assert resp.events == []
        assert resp.done is False

    def test_creation_with_done(self):
        from openenv.core.harnesses import HarnessResponse

        resp = HarnessResponse(response="Done.", done=True)
        assert resp.done is True

    def test_creation_with_events(self):
        from openenv.core.harnesses import (
            HarnessEvent,
            HarnessEventType,
            HarnessResponse,
        )

        events = [
            HarnessEvent(
                type=HarnessEventType.TOOL_CALL,
                timestamp=1.0,
                data={"tool_name": "bash"},
            ),
            HarnessEvent(
                type=HarnessEventType.TURN_COMPLETE,
                timestamp=2.0,
                data={"response": "Done."},
            ),
        ]
        resp = HarnessResponse(response="Done.", events=events, done=True)
        assert len(resp.events) == 2
        assert resp.events[0].type == HarnessEventType.TOOL_CALL

    def test_serialization_roundtrip(self):
        from openenv.core.harnesses import HarnessResponse

        resp = HarnessResponse(response="ok", done=True)
        json_str = resp.model_dump_json()
        restored = HarnessResponse.model_validate_json(json_str)
        assert restored.response == "ok"
        assert restored.done is True

    def test_requires_response(self):
        from openenv.core.harnesses import HarnessResponse

        with pytest.raises(ValidationError):
            HarnessResponse()


class TestHarnessConfig:
    """Tests for HarnessConfig Pydantic model."""

    def test_creation_minimal(self):
        from openenv.core.harnesses import HarnessConfig

        config = HarnessConfig(
            name="openclaw",
            command=["openclaw", "run"],
        )
        assert config.name == "openclaw"
        assert config.command == ["openclaw", "run"]

    def test_default_values(self):
        from openenv.core.harnesses import HarnessConfig, HarnessTransport

        config = HarnessConfig(name="test", command=["test"])
        assert config.working_directory == "/workspace"
        assert config.env_vars == {}
        assert config.transport == HarnessTransport.STDIO
        assert config.mcp_config_path is None
        assert config.startup_timeout_s == 30.0
        assert config.session_timeout_s == 600.0
        assert config.model is None
        assert config.api_key_env_var is None

    def test_custom_values(self):
        from openenv.core.harnesses import HarnessConfig, HarnessTransport

        config = HarnessConfig(
            name="claude-code",
            command=["claude", "--model", "opus"],
            working_directory="/home/user/project",
            env_vars={"ANTHROPIC_API_KEY": "sk-..."},
            transport=HarnessTransport.STREAMABLE_HTTP,
            mcp_config_path="/home/user/.claude/mcp.json",
            startup_timeout_s=60.0,
            session_timeout_s=1200.0,
            model="claude-opus-4-20250514",
            api_key_env_var="ANTHROPIC_API_KEY",
        )
        assert config.transport == HarnessTransport.STREAMABLE_HTTP
        assert config.model == "claude-opus-4-20250514"

    def test_requires_name(self):
        from openenv.core.harnesses import HarnessConfig

        with pytest.raises(ValidationError):
            HarnessConfig(command=["test"])

    def test_requires_command(self):
        from openenv.core.harnesses import HarnessConfig

        with pytest.raises(ValidationError):
            HarnessConfig(name="test")

    def test_serialization_roundtrip(self):
        from openenv.core.harnesses import HarnessConfig

        config = HarnessConfig(
            name="openclaw",
            command=["openclaw", "run"],
            model="claude-sonnet-4-20250514",
        )
        data = config.model_dump()
        restored = HarnessConfig.model_validate(data)
        assert restored.name == config.name
        assert restored.command == config.command
        assert restored.model == config.model


class TestHarnessAction:
    """Tests for HarnessAction (the action type for sending messages to harnesses)."""

    def test_creation(self):
        from openenv.core.harnesses import HarnessAction

        action = HarnessAction(message="Fix the bug in auth.py")
        assert action.message == "Fix the bug in auth.py"

    def test_inherits_from_action(self):
        from openenv.core.env_server.types import Action
        from openenv.core.harnesses import HarnessAction

        action = HarnessAction(message="hello")
        assert isinstance(action, Action)

    def test_has_metadata(self):
        from openenv.core.harnesses import HarnessAction

        action = HarnessAction(message="test", metadata={"source": "training_loop"})
        assert action.metadata["source"] == "training_loop"

    def test_requires_message(self):
        from openenv.core.harnesses import HarnessAction

        with pytest.raises(ValidationError):
            HarnessAction()

    def test_type_discriminator(self):
        from openenv.core.harnesses import HarnessAction

        action = HarnessAction(message="test")
        assert action.type == "harness_message"


class TestToolConflictResolution:
    """Tests for resolve_tool_conflicts utility."""

    def test_no_conflicts(self):
        from openenv.core.env_server.mcp_types import Tool
        from openenv.core.harnesses import resolve_tool_conflicts

        env_tools = [
            Tool(name="query_db", description="Query DB", input_schema={}),
        ]
        resolved = resolve_tool_conflicts(env_tools, ["bash", "read_file"])
        assert len(resolved) == 1
        assert resolved[0].name == "query_db"

    def test_conflict_adds_prefix(self):
        from openenv.core.env_server.mcp_types import Tool
        from openenv.core.harnesses import resolve_tool_conflicts

        env_tools = [
            Tool(name="read_file", description="Read a file", input_schema={}),
        ]
        resolved = resolve_tool_conflicts(env_tools, ["bash", "read_file"])
        assert len(resolved) == 1
        assert resolved[0].name == "env_read_file"

    def test_mixed_conflicts(self):
        from openenv.core.env_server.mcp_types import Tool
        from openenv.core.harnesses import resolve_tool_conflicts

        env_tools = [
            Tool(name="read_file", description="Read", input_schema={}),
            Tool(name="query_db", description="Query", input_schema={}),
            Tool(name="bash", description="Shell", input_schema={}),
        ]
        resolved = resolve_tool_conflicts(
            env_tools, ["bash", "read_file", "write_file"]
        )
        names = [t.name for t in resolved]
        assert "env_read_file" in names
        assert "query_db" in names
        assert "env_bash" in names

    def test_empty_env_tools(self):
        from openenv.core.harnesses import resolve_tool_conflicts

        resolved = resolve_tool_conflicts([], ["bash"])
        assert resolved == []

    def test_empty_builtin_tools(self):
        from openenv.core.env_server.mcp_types import Tool
        from openenv.core.harnesses import resolve_tool_conflicts

        env_tools = [
            Tool(name="my_tool", description="Test", input_schema={}),
        ]
        resolved = resolve_tool_conflicts(env_tools, [])
        assert len(resolved) == 1
        assert resolved[0].name == "my_tool"


class TestHarnessAdapterABC:
    """Tests for HarnessAdapter abstract base class."""

    def test_cannot_instantiate_directly(self):
        from openenv.core.harnesses import HarnessAdapter

        with pytest.raises(TypeError):
            HarnessAdapter(config=None)

    def test_concrete_subclass_must_implement_methods(self):
        from openenv.core.harnesses import HarnessAdapter, HarnessConfig

        class IncompleteAdapter(HarnessAdapter):
            pass

        config = HarnessConfig(name="test", command=["test"])
        with pytest.raises(TypeError):
            IncompleteAdapter(config=config)

    def test_concrete_subclass_works(self):
        from typing import AsyncIterator

        from openenv.core.harnesses import (
            HarnessAdapter,
            HarnessConfig,
            HarnessEvent,
            HarnessResponse,
        )

        class MockAdapter(HarnessAdapter):
            async def start(self, working_directory: str) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def inject_tools(self, tools) -> None:
                pass

            async def send_message(self, message: str) -> HarnessResponse:
                return HarnessResponse(response=f"Echo: {message}")

            async def send_message_streaming(
                self, message: str
            ) -> AsyncIterator[HarnessEvent]:
                yield  # type: ignore

            async def is_alive(self) -> bool:
                return False

        config = HarnessConfig(name="mock", command=["mock"])
        adapter = MockAdapter(config=config)
        assert adapter.config.name == "mock"


class TestModuleExports:
    """Tests that all expected types are importable from the harnesses module."""

    def test_import_harness_config(self):
        from openenv.core.harnesses import HarnessConfig

        assert HarnessConfig is not None

    def test_import_harness_transport(self):
        from openenv.core.harnesses import HarnessTransport

        assert HarnessTransport is not None

    def test_import_harness_adapter(self):
        from openenv.core.harnesses import HarnessAdapter

        assert HarnessAdapter is not None

    def test_import_harness_event(self):
        from openenv.core.harnesses import HarnessEvent

        assert HarnessEvent is not None

    def test_import_harness_event_type(self):
        from openenv.core.harnesses import HarnessEventType

        assert HarnessEventType is not None

    def test_import_harness_response(self):
        from openenv.core.harnesses import HarnessResponse

        assert HarnessResponse is not None

    def test_import_harness_action(self):
        from openenv.core.harnesses import HarnessAction

        assert HarnessAction is not None

    def test_import_resolve_tool_conflicts(self):
        from openenv.core.harnesses import resolve_tool_conflicts

        assert callable(resolve_tool_conflicts)
