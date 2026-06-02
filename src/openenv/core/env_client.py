# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Environment client for persistent sessions.

This module provides a WebSocket-based client that maintains a persistent connection
to an environment server, enabling efficient multi-step interactions without
the overhead of HTTP request/response cycles.

The client is async by default. For synchronous usage, use the `.sync()` method
to get a `SyncEnvClient` wrapper.

Example (async):
    >>> async with GenericEnvClient(base_url="ws://localhost:8000") as env:
    ...     result = await env.reset()
    ...     result = await env.step({"code": "print('hello')"})

Example (sync wrapper):
    >>> env = GenericEnvClient(base_url="ws://localhost:8000").sync()
    >>> with env:
    ...     result = env.reset()
    ...     result = env.step({"code": "print('hello')"})
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TYPE_CHECKING, TypeVar

from .client_types import StateT, StepResult
from .containers.runtime import LocalDockerProvider, UVProvider
from .utils import convert_to_ws_url

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    from .containers.runtime import ContainerProvider, RuntimeProvider
    from .sync_client import SyncEnvClient

from websockets.asyncio.client import connect as ws_connect

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")
EnvClientT = TypeVar("EnvClientT", bound="EnvClient")


class EnvClient(ABC, Generic[ActT, ObsT, StateT]):
    """
    Async environment client for persistent sessions.

    This client maintains a persistent WebSocket connection to an environment
    server, enabling efficient multi-step interactions. Each client instance
    corresponds to a dedicated environment session on the server.

    The client is async by default. For synchronous usage, use the `.sync()`
    method to get a `SyncEnvClient` wrapper.

    Features:
    - Lower latency for sequential interactions
    - Session state is maintained server-side
    - Better suited for long-running episodes
    - Async by default for modern Python async/await patterns

    Example (async):
        >>> from envs.coding_env.client import CodingEnv
        >>>
        >>> # Connect to a server using async context manager
        >>> async with CodingEnv(base_url="ws://localhost:8000") as env:
        ...     result = await env.reset(seed=42)
        ...     while not result.done:
        ...         action = agent.predict(result.observation)
        ...         result = await env.step(action)

    Example (sync wrapper):
        >>> env = CodingEnv(base_url="ws://localhost:8000").sync()
        >>> with env:
        ...     result = env.reset(seed=42)
        ...     result = env.step(action)
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        max_message_size_mb: float = 100.0,
        provider: Optional["ContainerProvider | RuntimeProvider"] = None,
        mode: Optional[str] = None,
    ):
        """
        Initialize environment client.

        Args:
            base_url: Base URL of the environment server (http:// or ws://).
                     Will be converted to ws:// if http:// is provided.
            connect_timeout_s: Timeout for establishing WebSocket connection
            message_timeout_s: Timeout for receiving responses to messages
            max_message_size_mb: Maximum WebSocket message size in megabytes.
                                Default 100MB to handle large observations (screenshots, DOM, etc.)
            provider: Optional container/runtime provider for lifecycle management.
                     Can be a ContainerProvider (Docker) or RuntimeProvider (UV).
            mode: Communication mode: 'simulation' for Gym-style API (default) or
                 'production' for MCP JSON-RPC protocol. Can also be set via the
                 OPENENV_CLIENT_MODE environment variable. Constructor parameter
                 takes precedence over environment variable. Case-insensitive.
        """
        # Determine mode (constructor > env var > default)
        if mode is None:
            mode = os.environ.get("OPENENV_CLIENT_MODE", "simulation")

        # Normalize and validate mode
        mode = mode.lower()
        if mode not in ("simulation", "production"):
            raise ValueError(
                f"Invalid mode: '{mode}'. Must be 'simulation' or 'production'. "
                f"Set via constructor parameter or OPENENV_CLIENT_MODE environment variable."
            )

        # Store mode (use object.__setattr__ to bypass immutability)
        object.__setattr__(self, "_mode", mode)

        # Convert HTTP URL to WebSocket URL
        ws_url = convert_to_ws_url(base_url)

        self._ws_url = f"{ws_url}/ws"
        self._connect_timeout = connect_timeout_s
        self._message_timeout = message_timeout_s
        self._max_message_size = int(
            max_message_size_mb * 1024 * 1024
        )  # Convert MB to bytes
        self._provider = provider
        self._ws: Optional[ClientConnection] = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification of _mode after initialization."""
        if name == "_mode" and hasattr(self, "_mode"):
            raise AttributeError("Cannot modify mode after initialization")
        super().__setattr__(name, value)

    async def connect(self) -> "EnvClient":
        """
        Establish WebSocket connection to the server.

        Returns:
            self for method chaining

        Raises:
            ConnectionError: If connection cannot be established
        """
        if self._ws is not None:
            return self

        # Bypass proxy for localhost connections
        ws_url_lower = self._ws_url.lower()
        is_localhost = "localhost" in ws_url_lower or "127.0.0.1" in ws_url_lower

        old_no_proxy = os.environ.get("NO_PROXY")
        if is_localhost:
            # Set NO_PROXY to bypass proxy for localhost
            current_no_proxy = old_no_proxy or ""
            if "localhost" not in current_no_proxy.lower():
                os.environ["NO_PROXY"] = (
                    f"{current_no_proxy},localhost,127.0.0.1"
                    if current_no_proxy
                    else "localhost,127.0.0.1"
                )

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=self._max_message_size,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {e}") from e
        finally:
            # Restore original NO_PROXY value
            if is_localhost:
                if old_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = old_no_proxy

        return self

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                # Send close message
                await self._send({"type": "close"})
            except Exception:
                pass  # Best effort
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _ensure_connected(self) -> None:
        """Ensure WebSocket connection is established."""
        if self._ws is None:
            await self.connect()

    async def _send(self, message: Dict[str, Any]) -> None:
        """Send a message over the WebSocket."""
        await self._ensure_connected()
        assert self._ws is not None
        await self._ws.send(json.dumps(message))

    async def _receive(self) -> Dict[str, Any]:
        """Receive and parse a message from the WebSocket."""
        assert self._ws is not None
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self._message_timeout)
        return json.loads(raw)

    async def _send_and_receive(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and wait for response."""
        await self._send(message)
        response = await self._receive()

        # Check for error response
        if response.get("type") == "error":
            error_data = response.get("data", {})
            raise RuntimeError(
                f"Server error: {error_data.get('message', 'Unknown error')} "
                f"(code: {error_data.get('code', 'UNKNOWN')})"
            )

        return response

    @classmethod
    async def from_docker_image(
        cls: Type[EnvClientT],
        image: str,
        provider: Optional["ContainerProvider"] = None,
        **kwargs: Any,
    ) -> EnvClientT:
        """
        Create an environment client by spinning up a Docker container.

        Args:
            image: Docker image name to run (e.g., "coding-env:latest")
            provider: Container provider to use (defaults to LocalDockerProvider)
            **kwargs: Additional arguments to pass to provider.start_container()

        Returns:
            Connected client instance
        """
        if provider is None:
            provider = LocalDockerProvider()

        # Start container
        base_url = provider.start_container(image, **kwargs)

        # Wait for server to be ready
        provider.wait_for_ready(base_url)

        # Create and connect client
        client = cls(base_url=base_url, provider=provider)
        await client.connect()

        return client

    @classmethod
    async def from_env(
        cls: Type[EnvClientT],
        repo_id: str,
        *,
        use_docker: bool = True,
        provider: Optional["ContainerProvider | RuntimeProvider"] = None,
        **provider_kwargs: Any,
    ) -> EnvClientT:
        """
        Create a client from a Hugging Face Space.

        Args:
            repo_id: Hugging Face space identifier ``{org}/{space}``.
            use_docker: When ``True`` (default) pull from the HF registry and
                launch via :class:`LocalDockerProvider`. When ``False`` run the
                space locally with :class:`UVProvider`.
            provider: Optional provider instance to reuse. Must be a
                :class:`ContainerProvider` when ``use_docker=True`` and a
                :class:`RuntimeProvider` otherwise.
            provider_kwargs: Additional keyword arguments forwarded to
                either the container provider's ``start_container`` (docker)
                or to the ``UVProvider`` constructor/start (uv). When
                ``use_docker=False``, the ``project_path`` argument can be
                used to override the default git URL
                (``git+https://huggingface.co/spaces/{repo_id}``).

        Returns:
            Connected client instance

        Examples:
            >>> # Pull and run from HF Docker registry
            >>> env = await MyEnv.from_env("openenv/echo-env")
            >>>
            >>> # Run locally with UV (clones the space)
            >>> env = await MyEnv.from_env("openenv/echo-env", use_docker=False)
            >>>
            >>> # Run from a local checkout
            >>> env = await MyEnv.from_env(
            ...     "openenv/echo-env",
            ...     use_docker=False,
            ...     project_path="/path/to/local/checkout"
            ... )
        """
        # Extract start args that apply to both providers
        start_args = {}
        for key in ("port", "env_vars", "workers"):
            if key in provider_kwargs:
                start_args[key] = provider_kwargs.pop(key)

        if use_docker:
            # Docker mode: pull from HF registry
            docker_provider = provider or LocalDockerProvider()
            tag = provider_kwargs.pop("tag", "latest")
            image = f"registry.hf.space/{repo_id.replace('/', '-')}:{tag}"
            base_url = docker_provider.start_container(
                image, **start_args, **provider_kwargs
            )
            docker_provider.wait_for_ready(base_url)

            client = cls(base_url=base_url, provider=docker_provider)
            await client.connect()
            return client
        else:
            # UV mode: clone and run with uv
            if provider is None:
                uv_kwargs = dict(provider_kwargs)
                project_path = uv_kwargs.pop("project_path", None)
                if project_path is None:
                    project_path = f"git+https://huggingface.co/spaces/{repo_id}"

                provider = UVProvider(project_path=project_path, **uv_kwargs)
            else:
                if provider_kwargs:
                    raise ValueError(
                        "provider_kwargs cannot be used when supplying a provider instance"
                    )

            base_url = provider.start(**start_args)
            provider.wait_for_ready()

            client = cls(base_url=base_url, provider=provider)
            await client.connect()
            return client

    @abstractmethod
    def _step_payload(self, action: ActT) -> Dict[str, Any]:
        """Convert an Action object to the JSON data expected by the env server."""
        raise NotImplementedError

    @abstractmethod
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ObsT]:
        """Convert a JSON response from the env server to StepResult[ObsT]."""
        raise NotImplementedError

    @abstractmethod
    def _parse_state(self, payload: Dict[str, Any]) -> StateT:
        """Convert a JSON response from the state endpoint to a State object."""
        raise NotImplementedError

    async def reset(self, **kwargs: Any) -> StepResult[ObsT]:
        """
        Reset the environment with optional parameters.

        Args:
            **kwargs: Optional parameters passed to the environment's reset method.
                     Common parameters include:
                     - seed: Random seed for reproducibility
                     - episode_id: Custom episode identifier

        Returns:
            StepResult containing initial observation
        """
        message = {
            "type": "reset",
            "data": kwargs,
        }
        response = await self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    async def step(self, action: ActT, **kwargs: Any) -> StepResult[ObsT]:
        """
        Execute an action in the environment.

        Args:
            action: The action to execute
            **kwargs: Optional parameters (currently ignored)

        Returns:
            StepResult containing observation, reward, and done status
        """
        message = {
            "type": "step",
            "data": self._step_payload(action),
        }
        response = await self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    async def state(self) -> StateT:
        """
        Get the current environment state from the server.

        Returns:
            State object with environment state information
        """
        message = {"type": "state"}
        response = await self._send_and_receive(message)
        return self._parse_state(response.get("data", {}))

    async def close(self) -> None:
        """
        Close the WebSocket connection and clean up resources.

        If this client was created via from_docker_image() or from_env(),
        this will also stop and remove the associated container/process.
        """
        await self.disconnect()

        if self._provider is not None:
            # Handle both ContainerProvider and RuntimeProvider
            if hasattr(self._provider, "stop_container"):
                self._provider.stop_container()
            elif hasattr(self._provider, "stop"):
                self._provider.stop()

    async def __aenter__(self) -> "EnvClient":
        """Enter async context manager, ensuring connection is established."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, closing connection."""
        await self.close()

    def __enter__(self) -> "EnvClient":
        """Sync context manager entry - raises error suggesting async usage."""
        raise TypeError(
            "EnvClient is async by default. Use 'async with' instead of 'with', "
            "or call .sync() to get a synchronous wrapper:\n"
            "  async with client:  # async usage\n"
            "  with client.sync():  # sync wrapper"
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sync context manager exit - should not be reached."""
        pass  # pragma: no cover

    def sync(self) -> "SyncEnvClient":
        """
        Return a synchronous wrapper around this async client.

        Use this method when you need synchronous access to the environment
        without async/await syntax. This is useful for:
        - Integration with synchronous codebases
        - Interactive/REPL usage
        - Stopping async from "infecting" the call stack

        Returns:
            SyncEnvClient wrapper that provides synchronous methods

        Example:
            >>> # Create async client and get sync wrapper
            >>> async_client = GenericEnvClient(base_url="http://localhost:8000")
            >>> sync_client = async_client.sync()
            >>>
            >>> # Use synchronous API
            >>> with sync_client:
            ...     result = sync_client.reset()
            ...     result = sync_client.step({"code": "print('hello')"})
        """
        from .sync_client import SyncEnvClient

        return SyncEnvClient(self)
