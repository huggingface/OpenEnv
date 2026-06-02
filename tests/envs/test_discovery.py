# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Package-Based Environment Discovery
===================================================

Tests cover:
1. Package discovery using importlib.metadata
2. Manifest loading from package resources
3. Class name inference
4. Cache management
5. Helper functions (_normalize_env_name, _is_hub_url, etc.)
"""

from unittest.mock import Mock, patch

from openenv.auto._discovery import (
    _create_env_info_from_package,
    _infer_class_name,
    _is_hub_url,
    _normalize_env_name,
    EnvironmentDiscovery,
    EnvironmentInfo,
    get_discovery,
    reset_discovery,
)


class TestEnvironmentInfo:
    """Test EnvironmentInfo dataclass and methods."""

    def test_environment_info_creation(self):
        """Test creating EnvironmentInfo instance."""
        env_info = EnvironmentInfo(
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
        )

        assert env_info.env_key == "echo"
        assert env_info.name == "echo_env"
        assert env_info.package_name == "openenv-echo-env"
        assert env_info.client_class_name == "EchoEnv"
        assert env_info.default_image == "echo-env:latest"


class TestHelperFunctions:
    """Test helper functions."""

    def test_normalize_env_name_simple(self):
        """Test normalizing simple names."""
        assert _normalize_env_name("echo") == "echo_env"
        assert _normalize_env_name("coding") == "coding_env"

    def test_normalize_env_name_with_suffix(self):
        """Test normalizing names with -env suffix."""
        assert _normalize_env_name("echo-env") == "echo_env"
        assert _normalize_env_name("coding-env") == "coding_env"

    def test_normalize_env_name_with_underscore(self):
        """Test normalizing names with _env suffix."""
        assert _normalize_env_name("echo_env") == "echo_env"
        assert _normalize_env_name("coding_env") == "coding_env"

    def test_is_hub_url_with_slash(self):
        """Test Hub URL detection with org/repo pattern."""
        assert _is_hub_url("meta-pytorch/coding-env")
        assert _is_hub_url("myorg/myenv")

    def test_is_hub_url_with_domain(self):
        """Test Hub URL detection with full URL."""
        assert _is_hub_url("https://huggingface.co/meta-pytorch/coding-env")
        assert _is_hub_url("huggingface.co/spaces/myenv")

    def test_is_hub_url_local(self):
        """Test that local names are not detected as Hub URLs."""
        assert not _is_hub_url("echo")
        assert not _is_hub_url("coding-env")
        assert not _is_hub_url("echo_env")

    def test_infer_class_name_client(self):
        """Test inferring client class names."""
        assert _infer_class_name("echo_env", "client") == "EchoEnv"
        assert _infer_class_name("coding_env", "client") == "CodingEnv"
        assert _infer_class_name("browser_gym_env", "client") == "BrowserGymEnv"

    def test_infer_class_name_action(self):
        """Test inferring action class names."""
        assert _infer_class_name("echo_env", "action") == "EchoAction"
        assert _infer_class_name("coding_env", "action") == "CodingAction"

    def test_infer_class_name_observation(self):
        """Test inferring observation class names."""
        assert _infer_class_name("echo_env", "observation") == "EchoObservation"
        assert _infer_class_name("coding_env", "observation") == "CodingObservation"


class TestCreateEnvInfoFromPackage:
    """Test creating EnvironmentInfo from package data."""

    @patch("openenv.auto._discovery._load_manifest_from_package")
    def test_create_env_info_with_manifest(self, mock_load_manifest):
        """Test creating env info when manifest exists."""
        # Mock manifest data
        mock_load_manifest.return_value = {
            "name": "echo_env",
            "version": "0.1.0",
            "description": "Echo environment for OpenEnv",
            "spec_version": 1,
        }

        env_info = _create_env_info_from_package(
            package_name="openenv-echo-env", module_name="echo_env", version="0.1.0"
        )

        assert env_info is not None
        assert env_info.env_key == "echo"
        assert env_info.name == "echo_env"
        assert env_info.package_name == "openenv-echo-env"
        assert env_info.version == "0.1.0"
        assert env_info.client_class_name == "EchoEnv"
        assert env_info.action_class_name == "EchoAction"

    @patch("openenv.auto._discovery._load_manifest_from_package")
    def test_create_env_info_with_custom_class_names(self, mock_load_manifest):
        """Test creating env info with custom class names from manifest."""
        # Mock manifest with custom class names
        mock_load_manifest.return_value = {
            "name": "coding_env",
            "version": "0.1.0",
            "description": "Coding environment",
            "action": "CodeAction",  # Custom name
            "observation": "CodeObservation",  # Custom name
        }

        env_info = _create_env_info_from_package(
            package_name="openenv-coding_env", module_name="coding_env", version="0.1.0"
        )

        assert env_info.action_class_name == "CodeAction"
        assert env_info.observation_class_name == "CodeObservation"

    @patch("openenv.auto._discovery._load_manifest_from_package")
    def test_create_env_info_without_manifest(self, mock_load_manifest):
        """Test creating env info when no manifest exists (uses conventions)."""
        mock_load_manifest.return_value = None

        env_info = _create_env_info_from_package(
            package_name="openenv-test-env", module_name="test_env", version="1.0.0"
        )

        assert env_info is not None
        assert env_info.env_key == "test"
        assert env_info.name == "test_env"
        assert env_info.client_class_name == "TestEnv"
        assert env_info.action_class_name == "TestAction"


class TestEnvironmentDiscovery:
    """Test EnvironmentDiscovery class."""

    @patch("importlib.metadata.distributions")
    @patch("openenv.auto._discovery._create_env_info_from_package")
    def test_discover_installed_packages(self, mock_create_info, mock_distributions):
        """Test discovering installed packages."""
        # Mock distribution objects
        mock_dist1 = Mock()
        mock_dist1.metadata = {"Name": "openenv-echo-env"}
        mock_dist1.version = "0.1.0"

        mock_dist2 = Mock()
        mock_dist2.metadata = {"Name": "openenv-coding_env"}
        mock_dist2.version = "0.2.0"

        mock_dist3 = Mock()
        mock_dist3.metadata = {"Name": "openenv-core"}  # Legacy core wheel
        mock_dist3.version = "1.0.0"

        mock_distributions.return_value = [mock_dist1, mock_dist2, mock_dist3]

        # Mock env info creation
        def create_info_side_effect(package_name, module_name, version):
            return EnvironmentInfo(
                env_key=module_name.replace("_env", ""),
                name=f"{module_name}",
                package_name=package_name,
                version=version,
                description=f"{module_name} environment",
                client_module_path=f"{module_name}.client",
                client_class_name=f"{module_name.replace('_env', '').capitalize()}Env",
                action_class_name=f"{module_name.replace('_env', '').capitalize()}Action",
                observation_class_name=f"{module_name.replace('_env', '').capitalize()}Observation",
                default_image=f"{module_name.replace('_', '-')}:latest",
            )

        mock_create_info.side_effect = create_info_side_effect

        discovery = EnvironmentDiscovery()
        envs = discovery._discover_installed_packages()

        # Should discover 2 environments, not the legacy core wheel
        assert len(envs) == 2
        assert "echo" in envs
        assert "coding" in envs

    def test_get_environment(self):
        """Test getting a specific environment."""
        discovery = EnvironmentDiscovery()

        # Mock the discover method
        with patch.object(discovery, "discover") as mock_discover:
            mock_discover.return_value = {
                "echo": EnvironmentInfo(
                    env_key="echo",
                    name="echo_env",
                    package_name="openenv-echo-env",
                    version="0.1.0",
                    description="Echo",
                    client_module_path="echo_env.client",
                    client_class_name="EchoEnv",
                    action_class_name="EchoAction",
                    observation_class_name="EchoObservation",
                    default_image="echo-env:latest",
                )
            }

            env = discovery.get_environment("echo")
            assert env is not None
            assert env.env_key == "echo"

    def test_get_environment_not_found(self):
        """Test getting a non-existent environment."""
        discovery = EnvironmentDiscovery()

        with patch.object(discovery, "discover") as mock_discover:
            mock_discover.return_value = {}

            env = discovery.get_environment("nonexistent")
            assert env is None

    def test_get_environment_by_name_flexible(self):
        """Test getting environment with flexible name matching."""
        discovery = EnvironmentDiscovery()

        mock_env = EnvironmentInfo(
            env_key="echo",
            name="echo_env",
            package_name="openenv-echo-env",
            version="0.1.0",
            description="Echo",
            client_module_path="echo_env.client",
            client_class_name="EchoEnv",
            action_class_name="EchoAction",
            observation_class_name="EchoObservation",
            default_image="echo-env:latest",
        )

        with patch.object(discovery, "discover") as mock_discover:
            mock_discover.return_value = {"echo": mock_env}

            # All these should work
            assert discovery.get_environment_by_name("echo") is not None
            assert discovery.get_environment_by_name("echo-env") is not None
            assert discovery.get_environment_by_name("echo_env") is not None

    def test_cache_management(self):
        """Test cache loading and saving."""
        discovery = EnvironmentDiscovery()

        # Create mock environment
        mock_env = EnvironmentInfo(
            env_key="test",
            name="test_env",
            package_name="openenv-test",
            version="1.0.0",
            description="Test",
            client_module_path="test_env.client",
            client_class_name="TestEnv",
            action_class_name="TestAction",
            observation_class_name="TestObservation",
            default_image="test-env:latest",
        )

        envs = {"test": mock_env}

        # Test saving cache
        discovery._save_cache(envs)
        assert discovery._cache_file.exists()

        # Test loading cache
        loaded = discovery._load_cache()
        assert loaded is not None
        assert "test" in loaded

        # Clean up
        discovery.clear_cache()
        assert not discovery._cache_file.exists()


class TestGlobalDiscovery:
    """Test global discovery instance management."""

    def test_get_discovery_singleton(self):
        """Test that get_discovery returns singleton."""
        reset_discovery()

        discovery1 = get_discovery()
        discovery2 = get_discovery()

        assert discovery1 is discovery2

    def test_reset_discovery(self):
        """Test resetting global discovery instance."""
        discovery1 = get_discovery()

        reset_discovery()

        discovery2 = get_discovery()

        # Should be different instances after reset
        assert discovery1 is not discovery2


class TestListEnvironments:
    """Test list_environments output."""

    def test_list_environments_with_envs(self, capsys):
        """Test listing when environments are found."""
        discovery = EnvironmentDiscovery()

        mock_envs = {
            "echo": EnvironmentInfo(
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
            )
        }

        with patch.object(discovery, "discover", return_value=mock_envs):
            discovery.list_environments()

        captured = capsys.readouterr()
        assert "Available OpenEnv Environments" in captured.out
        assert "echo" in captured.out
        assert "Total: 1 environments" in captured.out

    def test_list_environments_empty(self, capsys):
        """Test listing when no environments are found."""
        discovery = EnvironmentDiscovery()

        with patch.object(discovery, "discover", return_value={}):
            discovery.list_environments()

        captured = capsys.readouterr()
        assert "No OpenEnv environments found" in captured.out
        assert "pip install openenv-" in captured.out
