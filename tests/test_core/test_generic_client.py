# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for GenericEnvClient and GenericAction
===================================================

Tests cover:
1. GenericEnvClient instantiation and basic operations
2. Dictionary-based action/observation handling
3. from_docker_image() inheritance
4. from_env() inheritance (HuggingFace registry)
5. AutoEnv integration with skip_install parameter
6. Comparison with typed clients
7. GenericAction class
8. AutoAction with skip_install parameter
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from openenv.core.client_types import StepResult
from openenv.core.generic_client import GenericAction, GenericEnvClient
from openenv.core.sync_client import SyncEnvClient


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.recv.return_value = '{"type": "response", "data": {"observation": {"output": "hello"}, "reward": 1.0, "done": false}}'
    return ws


@pytest.fixture
def mock_provider():
    """Create a mock container provider."""
    provider = Mock()
    provider.start_container.return_value = "http://localhost:8000"
    provider.wait_for_ready.return_value = None
    return provider


# ============================================================================
# GenericEnvClient Unit Tests
# ============================================================================


class TestGenericEnvClientInstantiation:
    """Test GenericEnvClient instantiation."""

    def test_instantiation_with_http_url(self):
        """Test that GenericEnvClient can be instantiated with HTTP URL."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        assert client._ws_url == "ws://localhost:8000/ws"

    def test_instantiation_with_https_url(self):
        """Test that GenericEnvClient can be instantiated with HTTPS URL."""
        client = GenericEnvClient(base_url="https://example.com")
        assert client._ws_url == "wss://example.com/ws"

    def test_instantiation_with_ws_url(self):
        """Test that GenericEnvClient can be instantiated with WS URL."""
        client = GenericEnvClient(base_url="ws://localhost:8000")
        assert client._ws_url == "ws://localhost:8000/ws"

    def test_instantiation_with_custom_timeouts(self):
        """Test custom timeout parameters."""
        client = GenericEnvClient(
            base_url="http://localhost:8000",
            connect_timeout_s=30.0,
            message_timeout_s=120.0,
        )
        assert client._connect_timeout == 30.0
        assert client._message_timeout == 120.0


class TestGenericEnvClientStepPayload:
    """Test _step_payload method."""

    def test_step_payload_passthrough(self):
        """Test that action dict passes through unchanged."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        action = {"code": "print('hello')", "timeout": 30}
        payload = client._step_payload(action)

        assert payload == action
        assert payload["code"] == "print('hello')"
        assert payload["timeout"] == 30

    def test_step_payload_empty_dict(self):
        """Test with empty action dict."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        action = {}
        payload = client._step_payload(action)

        assert payload == {}

    def test_step_payload_nested_dict(self):
        """Test with nested action dict."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        action = {
            "command": "execute",
            "params": {"file": "test.py", "args": ["--verbose"]},
        }
        payload = client._step_payload(action)

        assert payload == action
        assert payload["params"]["file"] == "test.py"


class TestGenericEnvClientParseResult:
    """Test _parse_result method."""

    def test_parse_result_full_payload(self):
        """Test parsing a complete result payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {
            "observation": {"stdout": "hello", "stderr": ""},
            "reward": 1.5,
            "done": True,
        }
        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert result.observation == {"stdout": "hello", "stderr": ""}
        assert result.reward == 1.5
        assert result.done is True

    def test_parse_result_minimal_payload(self):
        """Test parsing a minimal payload with defaults."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {}
        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert result.observation == {}
        assert result.reward is None
        assert result.done is False

    def test_parse_result_missing_reward(self):
        """Test parsing payload without reward."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {"observation": {"data": "test"}, "done": False}
        result = client._parse_result(payload)

        assert result.observation == {"data": "test"}
        assert result.reward is None
        assert result.done is False


class TestGenericEnvClientParseState:
    """Test _parse_state method."""

    def test_parse_state_full_payload(self):
        """Test parsing a complete state payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {
            "episode_id": "ep-123",
            "step_count": 5,
            "custom_field": "value",
        }
        state = client._parse_state(payload)

        assert state == payload
        assert state["episode_id"] == "ep-123"
        assert state["step_count"] == 5

    def test_parse_state_empty_payload(self):
        """Test parsing empty state payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {}
        state = client._parse_state(payload)

        assert state == {}


class TestGenericEnvClientFromDockerImage:
    """Test from_docker_image class method."""

    @pytest.mark.asyncio
    async def test_from_docker_image_creates_client(self, mock_provider):
        """Test that from_docker_image creates a connected client."""
        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = await GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            )

            assert isinstance(client, GenericEnvClient)
            mock_provider.start_container.assert_called_once_with("coding-env:latest")
            mock_provider.wait_for_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_from_docker_image_with_env_vars(self, mock_provider):
        """Test from_docker_image with environment variables."""
        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = await GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
                env_vars={"DEBUG": "1"},
            )

            assert isinstance(client, GenericEnvClient)
            mock_provider.start_container.assert_called_once_with(
                "coding-env:latest", env_vars={"DEBUG": "1"}
            )


class TestGenericEnvClientFromEnv:
    """Test from_env class method (HuggingFace registry)."""

    @pytest.mark.asyncio
    async def test_from_env_with_docker(self, mock_provider):
        """Test from_env with use_docker=True pulls from HF registry."""
        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = await GenericEnvClient.from_env(
                "user/my-env",
                use_docker=True,
                provider=mock_provider,
            )

            assert isinstance(client, GenericEnvClient)
            # Should construct HF registry URL
            mock_provider.start_container.assert_called_once()
            call_args = mock_provider.start_container.call_args
            assert "registry.hf.space/user-my-env" in call_args[0][0]


# ============================================================================
# AutoEnv skip_install Integration Tests
# ============================================================================


class TestAutoEnvSkipInstall:
    """Test AutoEnv.from_env() with skip_install parameter."""

    def test_skip_install_with_base_url(self):
        """Test skip_install=True with explicit base_url."""
        from openenv.auto.auto_env import AutoEnv

        with patch.object(AutoEnv, "_check_server_availability", return_value=True):
            client = AutoEnv.from_env(
                "echo",
                base_url="http://localhost:8000",
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)

    def test_skip_install_with_unavailable_server(self):
        """Test skip_install=True with unavailable server raises error."""
        from openenv.auto.auto_env import AutoEnv

        with patch.object(AutoEnv, "_check_server_availability", return_value=False):
            with pytest.raises(ConnectionError) as exc_info:
                AutoEnv.from_env(
                    "echo",
                    base_url="http://localhost:8000",
                    skip_install=True,
                )

            assert "Server not available" in str(exc_info.value)

    def test_skip_install_with_hub_url_and_running_space(self):
        """Test skip_install=True with HF Space that is running."""
        from openenv.auto.auto_env import AutoEnv

        with (
            patch.object(AutoEnv, "_check_space_availability", return_value=True),
            patch.object(
                AutoEnv,
                "_resolve_space_url",
                return_value="https://user-my-env.hf.space",
            ),
        ):
            client = AutoEnv.from_env(
                "user/my-env",
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)

    def test_skip_install_with_hub_url_and_docker(self, mock_provider):
        """Test skip_install=True with HF Space not running uses Docker."""
        from openenv.auto.auto_env import AutoEnv

        # Create an async mock for from_env (since GenericEnvClient.from_env is now async)
        async def mock_from_env_async(*args, **kwargs):
            return GenericEnvClient(base_url="http://localhost:8000")

        with (
            patch.object(AutoEnv, "_check_space_availability", return_value=False),
            patch.object(
                AutoEnv,
                "_resolve_space_url",
                return_value="https://user-my-env.hf.space",
            ),
            patch.object(
                GenericEnvClient,
                "from_env",
                side_effect=mock_from_env_async,
            ) as mock_from_env,
        ):
            AutoEnv.from_env(
                "user/my-env",
                skip_install=True,
            )

            mock_from_env.assert_called_once()
            call_kwargs = mock_from_env.call_args[1]
            assert call_kwargs.get("use_docker") is True

    def test_skip_install_local_env_without_docker_image_raises(self):
        """Test skip_install=True for local env without docker_image raises error."""
        from openenv.auto.auto_env import AutoEnv

        with pytest.raises(ValueError) as exc_info:
            AutoEnv.from_env(
                "echo",  # Local name, not Hub URL
                skip_install=True,
            )

        error_msg = str(exc_info.value)
        assert "skip_install=True" in error_msg
        assert "base_url" in error_msg or "docker_image" in error_msg

    def test_skip_install_local_env_with_docker_image(self, mock_provider):
        """Test skip_install=True for local env with docker_image."""
        from openenv.auto.auto_env import AutoEnv

        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = AutoEnv.from_env(
                "echo",
                docker_image="echo-env:latest",
                container_provider=mock_provider,
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)
            mock_provider.start_container.assert_called_once()

    def test_skip_install_false_still_works(self):
        """Test that skip_install=False (default) still works as before."""
        from openenv.auto._discovery import EnvironmentInfo, reset_discovery
        from openenv.auto.auto_env import AutoEnv

        reset_discovery()

        mock_env_info = EnvironmentInfo(
            env_key="echo",
            name="echo_env",
            package_name="openenv-echo-env",
            version="0.1.0",
            description="Echo environment",
            client_module_path="echo_env.client",
            client_class_name="EchoEnv",
            action_class_name="EchoAction",
            observation_class_name="EchoObservation",
            default_image="echo-env:latest",
            spec_version=1,
        )

        mock_discovery = Mock()
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_discovery.discover.return_value = {"echo": mock_env_info}

        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_env_info.get_client_class = Mock(return_value=mock_client_class)

        with (
            patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery),
            patch.object(AutoEnv, "_check_server_availability", return_value=True),
        ):
            result = AutoEnv.from_env(
                "echo",
                base_url="http://localhost:8000",
                skip_install=False,  # Explicit False
            )

            # Should return the typed client, not GenericEnvClient
            assert result is mock_client_instance
            assert not isinstance(result, GenericEnvClient)


# ============================================================================
# Comparison Tests: GenericEnvClient vs Typed Client
# ============================================================================


class TestGenericVsTypedComparison:
    """Compare behavior of GenericEnvClient vs typed clients."""

    def test_step_payload_generic_vs_typed(self):
        """Compare step payload generation."""
        # GenericEnvClient - pass through
        generic_client = GenericEnvClient(base_url="http://localhost:8000")
        generic_payload = generic_client._step_payload({"message": "hello"})

        # Should be identical to what a typed client would produce
        assert generic_payload == {"message": "hello"}

    def test_parse_result_generic_returns_dict(self):
        """GenericEnvClient returns dict observation."""
        generic_client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {
            "observation": {"echoed_message": "hello", "length": 5},
            "reward": 1.0,
            "done": False,
        }
        result = generic_client._parse_result(payload)

        # Observation is a dict, not a typed object
        assert isinstance(result.observation, dict)
        assert result.observation["echoed_message"] == "hello"
        # Access is via dict keys, not attributes
        assert result.observation.get("length") == 5


# ============================================================================
# Import Tests
# ============================================================================


class TestGenericEnvClientImports:
    """Test that GenericEnvClient can be imported from various locations."""

    def test_import_from_core(self):
        """Test import from openenv.core."""
        from openenv.core import GenericEnvClient as GC1

        assert GC1 is GenericEnvClient

    def test_import_from_openenv(self):
        """Test import from openenv package."""
        from openenv import GenericEnvClient as GC2

        assert GC2 is GenericEnvClient

    def test_import_from_generic_client_module(self):
        """Test direct import from module."""
        from openenv.core.generic_client import GenericEnvClient as GC3

        assert GC3 is GenericEnvClient


class TestSyncEnvClientImports:
    """Test that SyncEnvClient can be imported from various locations."""

    def test_import_from_core(self):
        """Test import from openenv.core."""
        from openenv.core import SyncEnvClient as SC1

        assert SC1 is SyncEnvClient

    def test_import_from_openenv(self):
        """Test import from openenv package."""
        from openenv import SyncEnvClient as SC2

        assert SC2 is SyncEnvClient

    def test_import_from_sync_client_module(self):
        """Test direct import from module."""
        from openenv.core.sync_client import SyncEnvClient as SC3

        assert SC3 is SyncEnvClient


class TestSyncEnvClientWrapper:
    """Test SyncEnvClient wrapper functionality."""

    def test_sync_method_returns_sync_client(self):
        """Test that .sync() returns a SyncEnvClient."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = client.sync()

        assert isinstance(sync_client, SyncEnvClient)
        assert sync_client._async is client

    def test_sync_client_has_async_client_property(self):
        """Test that SyncEnvClient exposes async_client property."""
        async_client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = async_client.sync()

        assert sync_client.async_client is async_client

    def test_sync_client_delegates_payload_methods(self):
        """Test that SyncEnvClient delegates _step_payload to async client."""
        async_client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = async_client.sync()

        action = {"code": "print('hello')"}
        payload = sync_client._step_payload(action)

        assert payload == action

    def test_sync_client_delegates_parse_result(self):
        """Test that SyncEnvClient delegates _parse_result to async client."""
        async_client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = async_client.sync()

        payload = {
            "observation": {"output": "hello"},
            "reward": 1.0,
            "done": False,
        }
        result = sync_client._parse_result(payload)

        assert result.observation == {"output": "hello"}
        assert result.reward == 1.0


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestGenericEnvClientContextManager:
    """Test context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager_enter_exit(self):
        """Test that async context manager works correctly."""
        with (
            patch.object(
                GenericEnvClient, "connect", new_callable=AsyncMock
            ) as mock_connect,
            patch.object(
                GenericEnvClient, "close", new_callable=AsyncMock
            ) as mock_close,
        ):
            async with GenericEnvClient(base_url="http://localhost:8000") as client:
                assert isinstance(client, GenericEnvClient)
                mock_connect.assert_called_once()

            mock_close.assert_called_once()

    def test_sync_context_manager_raises_error(self):
        """Test that sync context manager raises helpful error."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        with pytest.raises(TypeError) as exc_info:
            with client:
                pass

        assert "async by default" in str(exc_info.value)
        assert ".sync()" in str(exc_info.value)

    def test_sync_wrapper_context_manager(self):
        """Test SyncEnvClient context manager works correctly."""
        with (
            patch.object(
                GenericEnvClient, "connect", new_callable=AsyncMock
            ) as mock_connect,
            patch.object(
                GenericEnvClient, "close", new_callable=AsyncMock
            ) as mock_close,
        ):
            async_client = GenericEnvClient(base_url="http://localhost:8000")
            sync_client = async_client.sync()

            with sync_client as client:
                assert isinstance(client, SyncEnvClient)
                mock_connect.assert_called_once()

            mock_close.assert_called_once()


# ============================================================================
# Integration Tests (require running server)
# ============================================================================


@pytest.mark.integration
class TestGenericEnvClientIntegration:
    """
    Integration tests that require a running server.

    These tests require a server to be running on localhost.

    Start a server first:
        cd envs/echo_env/server && uvicorn app:app --host 0.0.0.0 --port 8000

    Run these tests with:
        pytest -m integration tests/test_core/test_generic_client.py -v
    """

    @pytest.fixture
    def local_echo_server(self):
        """Check if local echo server is running."""
        import requests

        base_url = "http://localhost:8000"
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Local echo server not healthy")
            return base_url
        except requests.RequestException:
            pytest.skip(
                "Local echo server not running. "
                "Start it with: cd envs/echo_env/server && uvicorn app:app"
            )

    def test_generic_client_with_local_server(self, local_echo_server):
        """Test GenericEnvClient with a real local server using sync wrapper."""
        with GenericEnvClient(base_url=local_echo_server).sync() as client:
            # Reset
            result = client.reset()
            assert result is not None
            assert isinstance(result.observation, dict)

            # Step with dict action
            action = {"message": "Hello from GenericEnvClient!"}
            step_result = client.step(action)

            assert step_result is not None
            assert isinstance(step_result.observation, dict)
            assert "Hello from GenericEnvClient!" in step_result.observation.get(
                "echoed_message", ""
            )

    def test_generic_client_multiple_steps(self, local_echo_server):
        """Test multiple steps with GenericEnvClient using sync wrapper."""
        with GenericEnvClient(base_url=local_echo_server).sync() as client:
            client.reset()

            messages = ["First", "Second", "Third"]
            for msg in messages:
                result = client.step({"message": msg})
                assert msg in result.observation.get("echoed_message", "")

    def test_generic_client_state(self, local_echo_server):
        """Test getting state with GenericEnvClient using sync wrapper."""
        with GenericEnvClient(base_url=local_echo_server).sync() as client:
            client.reset()

            # Execute some steps
            client.step({"message": "step 1"})
            client.step({"message": "step 2"})

            # Get state
            state = client.state()

            assert isinstance(state, dict)
            # State should have step_count
            assert "step_count" in state or len(state) > 0

    @pytest.mark.asyncio
    async def test_generic_client_async_with_local_server(self, local_echo_server):
        """Test GenericEnvClient with async API."""
        async with GenericEnvClient(base_url=local_echo_server) as client:
            # Reset
            result = await client.reset()
            assert result is not None
            assert isinstance(result.observation, dict)

            # Step with dict action
            action = {"message": "Hello from async GenericEnvClient!"}
            step_result = await client.step(action)

            assert step_result is not None
            assert isinstance(step_result.observation, dict)
            assert "Hello from async GenericEnvClient!" in step_result.observation.get(
                "echoed_message", ""
            )


@pytest.mark.integration
@pytest.mark.docker
class TestGenericEnvClientDocker:
    """
    Docker integration tests for GenericEnvClient.

    These tests require Docker to be running.

    Run these tests with:
        pytest -m "integration and docker" tests/test_core/test_generic_client.py -v
    """

    @pytest.fixture
    def check_docker_and_image(self):
        """Check if Docker is available and echo-env image exists."""
        import shutil
        import subprocess

        if not shutil.which("docker"):
            pytest.skip("Docker is not installed")

        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            if result.returncode != 0:
                pytest.skip("Docker daemon is not running")
        except Exception:
            pytest.skip("Cannot access Docker")

        # Check for echo-env image
        result = subprocess.run(
            ["docker", "images", "-q", "echo-env:latest"],
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            pytest.skip("Docker image 'echo-env:latest' not found")

    @pytest.mark.asyncio
    async def test_generic_client_from_docker_image(self, check_docker_and_image):
        """Test GenericEnvClient.from_docker_image() with real Docker."""
        client = await GenericEnvClient.from_docker_image("echo-env:latest")

        try:
            # Reset
            result = await client.reset()
            assert result is not None
            assert isinstance(result.observation, dict)

            # Step
            step_result = await client.step({"message": "Docker test!"})
            assert "Docker test!" in step_result.observation.get("echoed_message", "")

            print("GenericEnvClient.from_docker_image() works!")
        finally:
            await client.close()


# ============================================================================
# GenericAction Tests
# ============================================================================


class TestGenericAction:
    """Test GenericAction class."""

    def test_create_from_kwargs(self):
        """Test creating GenericAction from keyword arguments."""
        action = GenericAction(code="print('hello')", timeout=30)

        assert action["code"] == "print('hello')"
        assert action["timeout"] == 30

    def test_is_dict_subclass(self):
        """Test that GenericAction is a dict subclass."""
        action = GenericAction(message="test")

        assert isinstance(action, dict)
        assert isinstance(action, GenericAction)

    def test_dict_methods_work(self):
        """Test that dict methods work on GenericAction."""
        action = GenericAction(a=1, b=2)

        assert action.get("a") == 1
        assert action.get("c", "default") == "default"
        assert list(action.keys()) == ["a", "b"]
        assert list(action.values()) == [1, 2]

    def test_empty_action(self):
        """Test creating empty GenericAction."""
        action = GenericAction()

        assert len(action) == 0
        assert dict(action) == {}

    def test_nested_values(self):
        """Test GenericAction with nested values."""
        action = GenericAction(
            command="run",
            params={"file": "test.py", "args": ["--verbose"]},
        )

        assert action["command"] == "run"
        assert action["params"]["file"] == "test.py"
        assert action["params"]["args"] == ["--verbose"]

    def test_repr(self):
        """Test GenericAction repr."""
        action = GenericAction(code="x=1")

        repr_str = repr(action)
        assert "GenericAction" in repr_str
        assert "code=" in repr_str

    def test_can_be_used_with_generic_client(self):
        """Test that GenericAction works with GenericEnvClient._step_payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        action = GenericAction(message="hello")

        payload = client._step_payload(action)

        assert payload == {"message": "hello"}


class TestGenericActionImports:
    """Test GenericAction imports."""

    def test_import_from_core(self):
        """Test import from openenv.core."""
        from openenv.core import GenericAction as GA1

        assert GA1 is GenericAction

    def test_import_from_openenv(self):
        """Test import from openenv package."""
        from openenv import GenericAction as GA2

        assert GA2 is GenericAction

    def test_import_from_module(self):
        """Test direct import from module."""
        from openenv.core.generic_client import GenericAction as GA3

        assert GA3 is GenericAction


# ============================================================================
# AutoAction skip_install Tests
# ============================================================================


class TestAutoActionSkipInstall:
    """Test AutoAction.from_env() with skip_install parameter."""

    def test_skip_install_returns_generic_action(self):
        """Test skip_install=True returns GenericAction class."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_env("user/any-env", skip_install=True)

        assert ActionClass is GenericAction

    def test_skip_install_works_for_local_names(self):
        """Test skip_install=True works for local environment names."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_env("echo", skip_install=True)

        assert ActionClass is GenericAction

    def test_skip_install_from_hub_alias(self):
        """Test skip_install works with from_hub alias."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_hub("user/my-env", skip_install=True)

        assert ActionClass is GenericAction

    def test_skip_install_action_can_be_instantiated(self):
        """Test that returned GenericAction can be instantiated."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_env("user/repo", skip_install=True)

        # Create an action
        action = ActionClass(code="print('hello')", timeout=30)

        assert action["code"] == "print('hello')"
        assert action["timeout"] == 30

    def test_skip_install_false_still_works(self):
        """Test that skip_install=False (default) still works as before."""
        from openenv.auto._discovery import EnvironmentInfo, reset_discovery
        from openenv.auto.auto_action import AutoAction

        reset_discovery()

        mock_env_info = EnvironmentInfo(
            env_key="echo",
            name="echo_env",
            package_name="openenv-echo-env",
            version="0.1.0",
            description="Echo environment",
            client_module_path="echo_env.client",
            client_class_name="EchoEnv",
            action_class_name="EchoAction",
            observation_class_name="EchoObservation",
            default_image="echo-env:latest",
            spec_version=1,
        )

        mock_discovery = Mock()
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_discovery.discover.return_value = {"echo": mock_env_info}

        mock_action_class = Mock()
        mock_env_info.get_action_class = Mock(return_value=mock_action_class)

        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            result = AutoAction.from_env("echo", skip_install=False)

            # Should return the typed action class, not GenericAction
            assert result is mock_action_class
            assert result is not GenericAction


# ============================================================================
# End-to-End: AutoEnv + AutoAction with skip_install
# ============================================================================


class TestAutoEnvAutoActionSkipInstallIntegration:
    """Test AutoEnv and AutoAction work together with skip_install."""

    def test_both_skip_install_returns_generic_types(self):
        """Test that both AutoEnv and AutoAction with skip_install work together."""
        from openenv.auto.auto_action import AutoAction
        from openenv.auto.auto_env import AutoEnv

        with patch.object(AutoEnv, "_check_server_availability", return_value=True):
            # Get client without installing package
            client = AutoEnv.from_env(
                "user/my-env",
                base_url="http://localhost:8000",
                skip_install=True,
            )

            # Get action class without installing package
            ActionClass = AutoAction.from_env("user/my-env", skip_install=True)

            # Both should be generic types
            assert isinstance(client, GenericEnvClient)
            assert ActionClass is GenericAction

            # They should work together
            action = ActionClass(code="test")
            payload = client._step_payload(action)
            assert payload == {"code": "test"}

    def test_mixed_skip_install_raises_warning_scenario(self):
        """
        Test scenario where user forgets skip_install on AutoAction.

        This documents the expected behavior - if user uses skip_install
        on AutoEnv but not on AutoAction, AutoAction will try to install.
        """
        import os

        from openenv.auto._discovery import reset_discovery
        from openenv.auto.auto_action import AutoAction
        from openenv.auto.auto_env import AutoEnv

        reset_discovery()

        with patch.object(AutoEnv, "_check_server_availability", return_value=True):
            # Get client without installing package
            client = AutoEnv.from_env(
                "user/my-env",
                base_url="http://localhost:8000",
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)

            # Now if user forgets skip_install on AutoAction...
            # It will try to install from Hub and fail (no confirmation in tests)
            # Set env var to bypass confirmation, but mock installation to fail
            with (
                patch.dict(os.environ, {"OPENENV_TRUST_REMOTE_CODE": "1"}),
                patch(
                    "openenv.auto.auto_action.AutoEnv._ensure_package_from_hub"
                ) as mock_ensure,
            ):
                # Make _ensure_package_from_hub raise an error
                mock_ensure.side_effect = ValueError("Installation failed")

                # This should raise ValueError from installation attempt
                with pytest.raises(ValueError) as exc_info:
                    AutoAction.from_env("user/my-env")  # forgot skip_install=True!

                # Installation was attempted
                assert "Installation failed" in str(exc_info.value)
