# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Environment Auto-Discovery System
==================================

This module provides automatic discovery of OpenEnv environments by:
1. Discovering installed openenv-* packages using importlib.metadata
2. Loading manifests (openenv.yaml) from package resources
3. Caching results for performance
4. Supporting HuggingFace Hub downloads

This enables AutoEnv to work without coupling to src/envs/ directory.
"""

import importlib
import importlib.metadata
import importlib.resources
import json
import logging
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """
    Rich information about a discovered environment.

    Attributes:
        env_key: Environment key (e.g., "echo", "coding")
        name: Full environment name (e.g., "echo_env")
        package_name: Package name (e.g., "openenv-echo_env")
        version: Version string
        description: Human-readable description
        client_module_path: Full module path to client (e.g., "echo_env.client")
        client_class_name: Client class name (e.g., "EchoEnv")
        action_class_name: Action class name (e.g., "EchoAction")
        observation_class_name: Observation class name (e.g., "EchoObservation")
        default_image: Default Docker image name (e.g., "echo-env:latest")
        spec_version: OpenEnv spec version (from openenv.yaml)
        manifest: Original manifest data
    """

    env_key: str
    name: str
    package_name: str
    version: str
    description: str
    client_module_path: str
    client_class_name: str
    action_class_name: str
    observation_class_name: str
    default_image: str
    spec_version: Optional[int] = None
    manifest: Optional[Dict[str, Any]] = None

    def get_client_class(self) -> Type:
        """
        Dynamically import and return the client class.

        Returns:
            Client class (e.g., EchoEnv)

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.client_module_path)
            return getattr(module, self.client_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.client_class_name} from {self.client_module_path}: {e}\n"
                f"Make sure the package '{self.package_name}' is installed: "
                f"pip install {self.package_name}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class {self.client_class_name} not found in {self.client_module_path}: {e}"
            ) from e

    def get_action_class(self) -> Type:
        """
        Dynamically import and return the action class.

        Returns:
            Action class (e.g., EchoAction)

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.client_module_path)
            return getattr(module, self.action_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.action_class_name} from {self.client_module_path}: {e}\n"
                f"Make sure the package '{self.package_name}' is installed: "
                f"pip install {self.package_name}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class {self.action_class_name} not found in {self.client_module_path}: {e}"
            ) from e

    def get_observation_class(self) -> Type:
        """
        Dynamically import and return the observation class.

        Returns:
            Observation class (e.g., EchoObservation)

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.client_module_path)
            return getattr(module, self.observation_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.observation_class_name} from {self.client_module_path}: {e}\n"
                f"Make sure the package '{self.package_name}' is installed: "
                f"pip install {self.package_name}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class {self.observation_class_name} not found in {self.client_module_path}: {e}"
            ) from e


def _normalize_env_name(name: str) -> str:
    """
    Normalize environment name to standard format.

    Args:
        name: Input name (e.g., "echo", "echo-env", "echo_env")

    Returns:
        Normalized name (e.g., "echo_env")

    Examples:
        >>> _normalize_env_name("echo")
        'echo_env'
        >>> _normalize_env_name("echo-env")
        'echo_env'
        >>> _normalize_env_name("echo_env")
        'echo_env'
    """
    # Remove common suffixes
    name = re.sub(r"[-_]env$", "", name)
    # Convert hyphens to underscores
    name = name.replace("-", "_")
    # Add _env suffix if not present
    if not name.endswith("_env"):
        name = f"{name}_env"
    return name


def _is_hub_url(name: str) -> bool:
    """
    Check if name is a HuggingFace Hub URL or repo ID.

    Args:
        name: Input name

    Returns:
        True if it looks like a Hub URL

    Examples:
        >>> _is_hub_url("openenv/echo_env")
        True
        >>> _is_hub_url("https://huggingface.co/openenv/echo_env")
        True
        >>> _is_hub_url("echo")
        False
    """
    # Contains org/repo pattern or huggingface.co domain
    return "/" in name or "huggingface.co" in name


def _infer_class_name(env_name: str, class_type: str) -> str:
    """
    Infer class name from environment name using simple conventions.

    Args:
        env_name: Environment name (e.g., "echo_env")
        class_type: Type of class ("client", "action", "observation")

    Returns:
        Inferred class name

    Examples:
        >>> _infer_class_name("echo_env", "client")
        'EchoEnv'
        >>> _infer_class_name("echo_env", "action")
        'EchoAction'
    """
    # Remove _env suffix for base name
    base_name = env_name.replace("_env", "")

    # Convert to PascalCase
    pascal_name = "".join(word.capitalize() for word in base_name.split("_"))

    # Add suffix based on type
    if class_type == "client":
        return f"{pascal_name}Env"
    elif class_type == "action":
        return f"{pascal_name}Action"
    elif class_type == "observation":
        return f"{pascal_name}Observation"
    else:
        raise ValueError(f"Unknown class type: {class_type}")


def _load_manifest_from_package(
    package_name: str, module_name: str
) -> Optional[Dict[str, Any]]:
    """
    Load openenv.yaml manifest from an installed package.

    Args:
        package_name: Package name (e.g., "openenv-echo_env")
        module_name: Module name (e.g., "echo_env")

    Returns:
        Parsed manifest dictionary, or None if not found

    """
    try:
        # Try to read openenv.yaml from package
        if hasattr(importlib.resources, "files"):
            # Python 3.9+
            package_files = importlib.resources.files(module_name)
            if (package_files / "openenv.yaml").is_file():
                manifest_text = (package_files / "openenv.yaml").read_text()
                return yaml.safe_load(manifest_text)
        else:
            # Python 3.7-3.8 fallback
            with importlib.resources.open_text(module_name, "openenv.yaml") as f:
                return yaml.safe_load(f)
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        logger.debug(f"No openenv.yaml found in {module_name}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load openenv.yaml from {module_name}: {e}")
        return None


def _create_env_info_from_package(
    package_name: str, module_name: str, version: str
) -> Optional[EnvironmentInfo]:
    """
    Create EnvironmentInfo from an installed package.

    Args:
        package_name: Package name (e.g., "openenv-echo_env")
        module_name: Module name (e.g., "echo_env")
        version: Package version

    Returns:
        EnvironmentInfo instance, or None if invalid
    """
    # Load manifest
    manifest = _load_manifest_from_package(package_name, module_name)

    # Get environment name
    if manifest and "name" in manifest:
        env_name = manifest["name"]
    else:
        # Infer from module name
        env_name = module_name

    # Normalize to ensure _env suffix
    if not env_name.endswith("_env"):
        env_name = f"{env_name}_env"

    # Determine env_key (e.g., "echo_env" → "echo")
    env_key = env_name.replace("_env", "") if env_name.endswith("_env") else env_name

    # Get description
    description = (
        manifest.get("description", f"{env_name} environment")
        if manifest
        else f"{env_name} environment"
    )

    # Get spec version
    spec_version = manifest.get("spec_version") if manifest else None

    # Determine class names
    # Check if manifest has custom class names (custom format)
    if manifest and "action" in manifest and "observation" in manifest:
        # Custom format (like coding_env)
        client_class_name = _infer_class_name(env_name, "client")
        action_class_name = manifest.get(
            "action", _infer_class_name(env_name, "action")
        )
        observation_class_name = manifest.get(
            "observation", _infer_class_name(env_name, "observation")
        )
    else:
        # Use conventions
        client_class_name = _infer_class_name(env_name, "client")
        action_class_name = _infer_class_name(env_name, "action")
        observation_class_name = _infer_class_name(env_name, "observation")

    # Module path is just module_name.client
    client_module_path = f"{module_name}.client"

    # Determine default Docker image name
    image_name = env_name.replace("_", "-")
    default_image = f"{image_name}:latest"

    return EnvironmentInfo(
        env_key=env_key,
        name=env_name,
        package_name=package_name,
        version=version,
        description=description,
        client_module_path=client_module_path,
        client_class_name=client_class_name,
        action_class_name=action_class_name,
        observation_class_name=observation_class_name,
        default_image=default_image,
        spec_version=spec_version,
        manifest=manifest,
    )


class EnvironmentDiscovery:
    """
    Auto-discovery system for OpenEnv environments using installed packages.

    This class discovers installed openenv-* packages and loads their metadata.
    """

    def __init__(self):
        """Initialize discovery system."""
        self._cache: Optional[Dict[str, EnvironmentInfo]] = None
        self._cache_file = Path(tempfile.gettempdir()) / "openenv_discovery_cache.json"

    def _discover_installed_packages(self) -> Dict[str, EnvironmentInfo]:
        """
        Discover all installed openenv-* packages.

        Returns:
            Dictionary mapping env_key to EnvironmentInfo
        """
        environments = {}

        # Invalidate import caches to ensure we pick up newly installed packages
        importlib.invalidate_caches()

        # Get all installed packages
        try:
            distributions = importlib.metadata.distributions()
        except Exception as e:
            logger.warning(f"Failed to get installed packages: {e}")
            return environments

        # Filter for openenv-* environment packages; ignore the legacy core wheel.
        for dist in distributions:
            package_name = dist.metadata["Name"]

            if not package_name.startswith("openenv-"):
                continue

            if package_name == "openenv-core":
                continue

            # Get module name (e.g., "openenv-echo_env" → "echo_env")
            module_name = package_name.replace("openenv-", "").replace("-", "_")

            # Get version
            version = dist.version

            try:
                # Create environment info
                env_info = _create_env_info_from_package(
                    package_name, module_name, version
                )

                if env_info:
                    environments[env_info.env_key] = env_info
                    logger.debug(
                        f"Discovered environment: {env_info.env_key} ({package_name})"
                    )

            except Exception as e:
                logger.warning(f"Failed to load environment from {package_name}: {e}")
                continue

        return environments

    def _load_cache(self) -> Optional[Dict[str, EnvironmentInfo]]:
        """
        Load cached discovery results.

        Returns:
            Dictionary of env_key -> EnvironmentInfo, or None if cache invalid
        """
        if not self._cache_file.exists():
            return None

        try:
            with open(self._cache_file, "r") as f:
                cache_data = json.load(f)

            # Reconstruct EnvironmentInfo objects
            cache = {}
            for env_key, env_data in cache_data.items():
                cache[env_key] = EnvironmentInfo(**env_data)

            return cache
        except Exception as e:
            logger.warning(f"Failed to load discovery cache: {e}")
            return None

    def _save_cache(self, environments: Dict[str, EnvironmentInfo]) -> None:
        """
        Save discovery results to cache.

        Args:
            environments: Dictionary of env_key -> EnvironmentInfo
        """
        try:
            cache_data = {}
            for env_key, env_info in environments.items():
                cache_data[env_key] = asdict(env_info)

            with open(self._cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save discovery cache: {e}")

    def discover(self, use_cache: bool = True) -> Dict[str, EnvironmentInfo]:
        """
        Discover all installed OpenEnv environments.

        Args:
            use_cache: If True, try to load from cache first

        Returns:
            Dictionary mapping env_key to EnvironmentInfo

        Examples:
            >>> discovery = EnvironmentDiscovery()
            >>> envs = discovery.discover()
            >>> print(envs.keys())
            dict_keys(['echo', 'coding', ...])
        """
        # Try to load from memory cache first
        if use_cache and self._cache is not None:
            return self._cache

        # Try to load from file cache
        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                self._cache = cached
                return self._cache

        # Discover from installed packages
        environments = self._discover_installed_packages()

        # Save to cache
        self._save_cache(environments)
        self._cache = environments

        return environments

    def get_environment(self, env_key: str) -> Optional[EnvironmentInfo]:
        """
        Get information about a specific environment.

        Args:
            env_key: Environment key (e.g., "echo", "coding")

        Returns:
            EnvironmentInfo if found, None otherwise

        Examples:
            >>> discovery = EnvironmentDiscovery()
            >>> env = discovery.get_environment("echo")
            >>> print(env.client_class_name)
            'EchoEnv'
        """
        environments = self.discover()
        return environments.get(env_key)

    def get_environment_by_name(self, name: str) -> Optional[EnvironmentInfo]:
        """
        Get environment info by flexible name matching.

        Args:
            name: Environment name (e.g., "echo", "echo-env", "echo_env")

        Returns:
            EnvironmentInfo if found, None otherwise
        """
        # Normalize name to env_key
        normalized = _normalize_env_name(name)
        env_key = normalized.replace("_env", "")

        return self.get_environment(env_key)

    def list_environments(self) -> None:
        """
        Print a formatted list of all discovered environments.

        Examples:
            >>> discovery = EnvironmentDiscovery()
            >>> discovery.list_environments()
            Available OpenEnv Environments:
            ----------------------------------------------------------------------
              echo           : Echo Environment (v0.1.0) - openenv-echo_env
              coding         : Coding Environment (v0.1.0) - openenv-coding_env
              ...
        """
        environments = self.discover()

        print("Available OpenEnv Environments:")
        print("-" * 70)

        if not environments:
            print("  No OpenEnv environments found.")
            print("  Install environments with: pip install openenv-<env-name>")
        else:
            for env_key in sorted(environments.keys()):
                env = environments[env_key]
                print(f"  {env_key:<15}: {env.description} (v{env.version})")
                print(f"                   Package: {env.package_name}")

        print("-" * 70)
        print(f"Total: {len(environments)} environments")

    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        if self._cache_file.exists():
            self._cache_file.unlink()
        self._cache = None


# Global discovery instance
_global_discovery: Optional[EnvironmentDiscovery] = None


def get_discovery() -> EnvironmentDiscovery:
    """
    Get or create the global discovery instance.

    Returns:
        Global EnvironmentDiscovery instance

    Examples:
        >>> discovery = get_discovery()
        >>> envs = discovery.discover()
    """
    global _global_discovery

    if _global_discovery is None:
        _global_discovery = EnvironmentDiscovery()

    return _global_discovery


def reset_discovery() -> None:
    """Reset the global discovery instance (useful for testing)."""
    global _global_discovery
    if _global_discovery is not None:
        _global_discovery.clear_cache()
    _global_discovery = None
