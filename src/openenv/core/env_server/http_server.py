# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HTTP server wrapper for Environment instances.

This module provides utilities to wrap any Environment subclass and expose it
over HTTP and WebSocket endpoints that EnvClient can consume.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from typing import Any, AsyncContextManager, Callable, cast, Dict, Optional, Type

_MISSING = object()

from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    Request,
    status,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import ValidationError

from .interfaces import Environment
from .mcp_environment import get_server_tools
from .mcp_types import (
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    McpMethod,
    WSMCPMessage,
    WSMCPResponse,
)
from .route_config import GetEndpointConfig, register_get_endpoints
from .serialization import deserialize_action, serialize_observation
from .types import (
    Action,
    ConcurrencyConfig,
    EnvironmentMetadata,
    HealthResponse,
    HealthStatus,
    Observation,
    ResetRequest,
    ResetResponse,
    SchemaResponse,
    ServerCapacityStatus,
    ServerMode,
    SessionInfo,
    State,
    StepRequest,
    StepResponse,
    WSCloseMessage,
    WSErrorCode,
    WSErrorResponse,
    WSObservationResponse,
    WSResetMessage,
    WSStateMessage,
    WSStateResponse,
    WSStepMessage,
)


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable form.

    Handles Pydantic models, dataclasses, and other common types.

    Args:
        obj: The object to convert

    Returns:
        A JSON-serializable representation of the object
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        # Pydantic model
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        # Object with __dict__
        return {k: _make_json_serializable(v) for k, v in obj.__dict__.items()}
    # Fallback to string representation
    return str(obj)


from .exceptions import (
    ConcurrencyConfigurationError,
    EnvironmentFactoryError,
    SessionCapacityError,
)


class HTTPEnvServer:
    """
    HTTP server wrapper for Environment instances.

    This class wraps an Environment and exposes its reset(), step(), and state
    methods as HTTP and WebSocket endpoints compatible with EnvClient.

    The server expects:
    - Action deserialization: Converts JSON dict to Action subclass
    - Observation serialization: Converts Observation subclass to JSON dict

    Example:
        >>> from core.env_server import HTTPEnvServer
        >>> from envs.coding_env.server import CodeExecutionEnvironment
        >>> from envs.coding_env.models import CodeAction, CodeObservation
        >>>
        >>> # Pass environment class (factory pattern)
        >>> server = HTTPEnvServer(
        ...     env=CodeExecutionEnvironment,
        ...     action_cls=CodeAction,
        ...     observation_cls=CodeObservation,
        ...     max_concurrent_envs=4,
        ... )
        >>>
        >>> # Register routes with FastAPI
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> server.register_routes(app)
    """

    def __init__(
        self,
        env: Callable[[], Environment],
        action_cls: Type[Action],
        observation_cls: Type[Observation],
        max_concurrent_envs: Optional[int] = None,
        concurrency_config: Optional[ConcurrencyConfig] = None,
    ):
        """
        Initialize HTTP server wrapper.

        Args:
            env: Environment factory (callable) that creates new instances.
                 Will be called to create a new environment for each WebSocket session.
            action_cls: The Action subclass this environment expects
            observation_cls: The Observation subclass this environment returns
            max_concurrent_envs: Maximum number of concurrent WebSocket sessions.
                                 Mutually exclusive with concurrency_config.
            concurrency_config: Optional ConcurrencyConfig for advanced concurrency settings.
                                Mutually exclusive with max_concurrent_envs.

        Raises:
            ValueError: If both max_concurrent_envs and concurrency_config are provided.
            ConcurrencyConfigurationError: If max_concurrent_envs > 1 for an
                environment that is not marked as SUPPORTS_CONCURRENT_SESSIONS.
        """
        # Validate that env is callable
        if not callable(env):
            raise TypeError(
                f"env must be a callable (class or factory function), got {type(env)}. "
                f"Pass the environment class (e.g., MyEnvironment) not an instance (e.g., MyEnvironment())."
            )

        self._env_factory: Callable[[], Environment] = env

        # Handle concurrency configuration
        if max_concurrent_envs is not None and concurrency_config is not None:
            raise ValueError(
                "Cannot specify both 'max_concurrent_envs' and 'concurrency_config'. "
                "Please use only one method to configure concurrency."
            )

        if concurrency_config is not None:
            self._concurrency_config = concurrency_config
        elif max_concurrent_envs is not None:
            self._concurrency_config = ConcurrencyConfig(
                max_concurrent_envs=max_concurrent_envs,
                session_timeout=None,
            )
        else:
            # Default configuration
            self._concurrency_config = ConcurrencyConfig(
                max_concurrent_envs=1,
                session_timeout=None,
            )

        self._max_concurrent_envs = self._concurrency_config.max_concurrent_envs

        # Validate concurrency configuration
        self._validate_concurrency_safety()

        self.action_cls = action_cls
        self.observation_cls = observation_cls

        # Session management for WebSocket connections
        self._sessions: Dict[str, Optional[Environment]] = {}
        self._session_executors: Dict[str, ThreadPoolExecutor] = {}
        self._session_stacks: Dict[str, AsyncExitStack] = {}
        self._session_info: Dict[str, SessionInfo] = {}
        self._session_lock = asyncio.Lock()

        # Create thread pool for running sync code in async context
        # This is needed for environments using sync libraries (e.g., Playwright)
        self._executor = ThreadPoolExecutor(max_workers=32)

        self._requires_single_thread_executor = self._detect_single_thread_requirement()
        self._shared_session_executor: Optional[ThreadPoolExecutor] = None
        if self._requires_single_thread_executor:
            self._shared_session_executor = ThreadPoolExecutor(max_workers=1)

        # Idle session reaper configuration.
        # Timeout is taken from ConcurrencyConfig.session_timeout;
        # None means no timeout (default — reaper is a no-op).
        self._session_idle_timeout_s: Optional[float] = (
            self._concurrency_config.session_timeout
        )
        self._reaper_task: Optional[asyncio.Task[None]] = None

    def _validate_concurrency_safety(self) -> None:
        """
        Validate that the environment supports the configured concurrency level.

        Raises:
            ConcurrencyConfigurationError: If max_concurrent_envs > 1 for an
                environment that is not marked as SUPPORTS_CONCURRENT_SESSIONS.
        """
        import functools

        if self._max_concurrent_envs <= 1:
            return

        factory = self._env_factory
        if isinstance(factory, functools.partial):
            factory = factory.func

        if inspect.isclass(factory):
            env_cls = factory
        else:
            _temp_env = self._env_factory()
            env_cls = type(_temp_env)
            _temp_env.close()
            del _temp_env

        if not getattr(env_cls, "SUPPORTS_CONCURRENT_SESSIONS", False):
            raise ConcurrencyConfigurationError(
                environment_name=env_cls.__name__,
                max_concurrent_envs=self._max_concurrent_envs,
            )

    def _detect_single_thread_requirement(self) -> bool:
        import functools

        factory = self._env_factory
        if isinstance(factory, functools.partial):
            factory = factory.func
        if inspect.isclass(factory):
            return getattr(factory, "REQUIRES_SINGLE_THREAD_EXECUTOR", False)
        return False

    def get_capacity_status(self) -> ServerCapacityStatus:
        """
        Get the current capacity status of the server.

        Returns:
            ServerCapacityStatus with current session counts and availability.
        """
        return ServerCapacityStatus.from_counts(
            active=len(self._sessions),
            max_sessions=self._max_concurrent_envs,
        )

    async def _run_sync_in_thread_pool(
        self, func: Callable[..., Observation], *args, **kwargs
    ) -> Observation:
        """Run a synchronous function in the thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    def _get_valid_kwargs(
        self,
        sig: inspect.Signature,
        kwargs: Dict[str, Any],
        skip_params: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """Filter kwargs to only include parameters accepted by the function signature."""
        if skip_params is None:
            skip_params = set()

        valid_kwargs = {}

        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        for k, v in kwargs.items():
            if k in sig.parameters or has_kwargs:
                if k not in skip_params:
                    valid_kwargs[k] = v

        return valid_kwargs

    async def _create_session(self) -> tuple[str, Environment]:
        """
        Create a new WebSocket session with its own environment instance.

        Returns:
            Tuple of (session_id, environment)

        Raises:
            SessionCapacityError: If max concurrent sessions reached
            EnvironmentFactoryError: If the factory fails to create an environment
        """
        async with self._session_lock:
            if len(self._sessions) >= self._max_concurrent_envs:
                raise SessionCapacityError(
                    active_sessions=len(self._sessions),
                    max_sessions=self._max_concurrent_envs,
                )

            session_id = str(uuid.uuid4())
            current_time = time.time()

            # Create executor and reserve slot so capacity is not exceeded while
            # we create the env outside the lock (avoids blocking other sessions)
            if self._shared_session_executor is not None:
                executor = self._shared_session_executor
            else:
                executor = ThreadPoolExecutor(max_workers=1)
            self._session_executors[session_id] = executor
            self._sessions[session_id] = None  # placeholder until env is ready

        try:
            # Create environment in the executor thread (outside lock)
            loop = asyncio.get_event_loop()
            env = await loop.run_in_executor(executor, self._env_factory)
        except Exception as e:
            async with self._session_lock:
                if executor is not self._shared_session_executor:
                    executor.shutdown(wait=False)
                self._session_executors.pop(session_id, None)
                self._sessions.pop(session_id, None)
            factory_name = getattr(
                self._env_factory, "__name__", str(self._env_factory)
            )
            raise EnvironmentFactoryError(factory_name) from e

        # Hold the MCP session open for the lifetime of this session,
        # matching the WebSocket path's AsyncExitStack pattern.  This
        # prevents per-request MCP transport teardown/reconnection and
        # preserves FastMCP session state (ctx.set_state / ctx.get_state)
        # across HTTP calls within the same OpenEnv session.
        stack = AsyncExitStack()
        try:
            mcp_session_factory = getattr(env, "mcp_session", None)
            if callable(mcp_session_factory):
                mcp_session_cm = cast(AsyncContextManager[Any], mcp_session_factory())
                await stack.enter_async_context(mcp_session_cm)
        except Exception:
            # MCP transport failed to start — clean up the reserved slot,
            # the env, and the executor so they don't leak permanently
            # against _max_concurrent_envs.
            await stack.aclose()  # best-effort
            async with self._session_lock:
                self._sessions.pop(session_id, None)
                self._session_executors.pop(session_id, None)
                self._session_info.pop(session_id, None)
            await self._cleanup_session_resources(env, executor)
            raise

        async with self._session_lock:
            self._sessions[session_id] = env
            self._session_stacks[session_id] = stack
            now = time.time()
            self._session_info[session_id] = SessionInfo(
                session_id=session_id,
                created_at=current_time,
                last_activity_at=now,
                step_count=0,
                environment_type=type(env).__name__,
            )

        return session_id, env

    async def _destroy_session(self, session_id: str) -> None:
        """
        Destroy a WebSocket session and cleanup resources.

        Args:
            session_id: The session ID to destroy
        """
        async with self._session_lock:
            env = self._sessions.pop(session_id, None)
            executor = self._session_executors.pop(session_id, None)
            stack = self._session_stacks.pop(session_id, None)
            self._session_info.pop(session_id, None)

        await self._cleanup_session_resources(env, executor, stack)

    async def _cleanup_session_resources(
        self,
        env: Optional[Environment],
        executor: Optional[ThreadPoolExecutor],
        stack: Optional[AsyncExitStack] = None,
    ) -> None:
        """Close an environment and shut down its executor (best-effort)."""
        # Close the MCP session stack first — this gracefully exits the
        # mcp_session() context (and the underlying FastMCP Client session)
        # before we tear down the environment references.
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                pass  # Best effort cleanup

        # Run close() in the same executor where the env was created
        # This is required for thread-sensitive libraries like Playwright/greenlet
        if env is not None:
            if executor is not None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(executor, env.close)
                except Exception:
                    # If executor close fails, try direct close as fallback
                    try:
                        env.close()
                    except Exception:
                        pass  # Best effort cleanup
            else:
                try:
                    env.close()
                except Exception:
                    pass  # Best effort cleanup

        if executor is not None and executor is not self._shared_session_executor:
            executor.shutdown(wait=False)

    def _update_session_activity(
        self, session_id: str, increment_step: bool = False
    ) -> None:
        """
        Update session activity timestamp and optionally increment step count.

        Args:
            session_id: The session ID to update
            increment_step: If True, increment the step count
        """
        if session_id in self._session_info:
            self._session_info[session_id].last_activity_at = time.time()
            if increment_step:
                self._session_info[session_id].step_count += 1

    async def _reap_idle_sessions(self) -> None:
        """Background task that periodically destroys sessions idle beyond the timeout."""
        timeout = self._session_idle_timeout_s
        if timeout is None:
            return  # no timeout configured — noop
        interval = max(timeout / 4, 5.0)  # check frequently enough
        while True:
            try:
                await asyncio.sleep(interval)
                now = time.time()
                stale_ids: list[str] = []
                async with self._session_lock:
                    for sid, info in self._session_info.items():
                        if now - info.last_activity_at > timeout:
                            stale_ids.append(sid)
                for sid in stale_ids:
                    # Re-check under lock: activity may have arrived since
                    # the snapshot was taken, making this session active again.
                    # Refresh `now` so slow _destroy_session calls don't cause
                    # subsequent entries to be validated against a stale clock.
                    now = time.time()
                    async with self._session_lock:
                        info = self._session_info.get(sid)
                        if info is None or (now - info.last_activity_at) <= timeout:
                            continue
                    await self._destroy_session(sid)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Idle-session reaper encountered an error (will retry): %s",
                    exc,
                )

    def _start_reaper(self) -> None:
        """Start the idle-session reaper if a timeout is configured."""
        if self._session_idle_timeout_s is not None and self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reap_idle_sessions())

    def _stop_reaper(self) -> None:
        """Cancel the reaper background task."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            self._reaper_task = None

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get information about a specific session.

        Args:
            session_id: The session ID to query

        Returns:
            SessionInfo if the session exists, None otherwise
        """
        return self._session_info.get(session_id)

    async def _run_in_session_executor(
        self, session_id: str, func: Callable[..., Observation], *args, **kwargs
    ) -> Observation:
        """Run a synchronous function in the session's thread pool executor."""
        executor = self._session_executors.get(session_id, self._executor)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    @property
    def active_sessions(self) -> int:
        """Return the number of active WebSocket sessions."""
        return len(self._sessions)

    @property
    def max_concurrent_envs(self) -> int:
        """Return the maximum number of concurrent environments."""
        return self._max_concurrent_envs

    @property
    def is_concurrency_safe(self) -> bool:
        """Return whether the environment is marked as concurrency safe."""
        import inspect

        if inspect.isclass(self._env_factory):
            return getattr(self._env_factory, "SUPPORTS_CONCURRENT_SESSIONS", False)
        else:
            _temp_env = self._env_factory()
            result = getattr(_temp_env, "SUPPORTS_CONCURRENT_SESSIONS", False)
            _temp_env.close()
            del _temp_env
            return result

    @property
    def concurrency_config(self) -> ConcurrencyConfig:
        """Return the concurrency configuration."""
        return self._concurrency_config

    def register_routes(
        self, app: FastAPI, mode: ServerMode | str = ServerMode.SIMULATION
    ) -> None:
        """
        Register HTTP routes on a FastAPI application.

        Args:
            app: FastAPI application instance
            mode: Server mode - either SIMULATION or PRODUCTION (or string equivalents).
                  In production mode, simulation control endpoints (/reset, /step, /state)
                  are NOT registered. Only safe endpoints (/health, /schema, /metadata, /ws)
                  are available. Defaults to SIMULATION for backwards compatibility.

        Raises:
            ValueError: If mode is not a valid ServerMode or string equivalent.
        """
        # Convert string to ServerMode enum for backwards compatibility
        if isinstance(mode, str):
            try:
                mode = ServerMode(mode.lower())
            except ValueError:
                valid_modes = [m.value for m in ServerMode]
                raise ValueError(
                    f"Invalid mode: '{mode}'. Must be one of: {valid_modes}"
                )

        # Wire up idle-session reaper lifecycle via app events
        server_ref = self

        async def _start_session_reaper() -> None:
            server_ref._start_reaper()

        async def _stop_session_reaper() -> None:
            server_ref._stop_reaper()
            if server_ref._shared_session_executor is not None:
                server_ref._shared_session_executor.shutdown(wait=True)

        if not getattr(app.router, "_openenv_reaper_registered", False):
            app.router.on_startup.append(_start_session_reaper)
            app.router.on_shutdown.append(_stop_session_reaper)
            app.router._openenv_reaper_registered = True  # type: ignore[attr-defined]

        # Helper function to handle reset endpoint
        async def reset_handler(
            request: ResetRequest = Body(default_factory=ResetRequest),
        ) -> ResetResponse:
            """Reset endpoint - returns initial observation."""
            _env = self._env_factory()

            try:
                kwargs = request.model_dump(exclude_unset=True)

                is_async = _env.reset_async.__func__ is not Environment.reset_async

                if is_async:
                    sig = inspect.signature(_env.reset_async)
                else:
                    sig = inspect.signature(_env.reset)
                valid_kwargs = self._get_valid_kwargs(sig, kwargs)

                if is_async:
                    observation = await _env.reset_async(**valid_kwargs)
                else:
                    observation = await self._run_sync_in_thread_pool(
                        _env.reset, **valid_kwargs
                    )
                return ResetResponse(**serialize_observation(observation))
            finally:
                _env.close()

        # Helper function to handle step endpoint
        async def step_handler(request: StepRequest) -> StepResponse:
            """Step endpoint - executes action and returns observation."""
            action_data = request.action

            try:
                action = deserialize_action(action_data, self.action_cls)
            except ValidationError as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=e.errors()
                )

            _env = self._env_factory()

            try:
                kwargs = request.model_dump(exclude_unset=True, exclude={"action"})

                is_async = _env.step_async.__func__ is not Environment.step_async

                if is_async:
                    sig = inspect.signature(_env.step_async)
                else:
                    sig = inspect.signature(_env.step)
                valid_kwargs = self._get_valid_kwargs(
                    sig, kwargs, skip_params={"action"}
                )

                if is_async:
                    observation = await _env.step_async(action, **valid_kwargs)
                else:
                    observation = await self._run_sync_in_thread_pool(
                        _env.step, action, **valid_kwargs
                    )

                return StepResponse(**serialize_observation(observation))
            finally:
                _env.close()

        # Helper function to handle MCP endpoint
        async def mcp_handler(
            request: JsonRpcRequest,
            session_env: Optional[Environment] = None,
            session_id: Optional[str] = None,
        ) -> JsonRpcResponse:
            """
            Handle MCP JSON-RPC requests.

            Supports tools/list and tools/call methods in JSON-RPC 2.0 format,
            plus OpenEnv session lifecycle methods for HTTP MCP:
            - openenv/session/create
            - openenv/session/close
            """
            method = request.method
            request_id = request.id
            params = request.params
            if not isinstance(params, dict):
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    "Params must be an object",
                    request_id=request_id,
                )

            # OpenEnv extension methods for explicit MCP session management.
            # This enables persistent MCP lifecycles over HTTP /mcp, matching WebSocket semantics.
            if method == "openenv/session/create":
                if session_env is not None and session_id is not None:
                    return JsonRpcResponse.success(
                        result={"session_id": session_id},
                        request_id=request_id,
                    )
                try:
                    created_session_id, _ = await self._create_session()
                except SessionCapacityError as e:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.SERVER_ERROR,
                        str(e),
                        request_id=request_id,
                        data={
                            "active_sessions": e.active_sessions,
                            "max_sessions": e.max_sessions,
                        },
                    )
                except EnvironmentFactoryError as e:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.SERVER_ERROR,
                        str(e),
                        request_id=request_id,
                        data={"factory_name": e.factory_name},
                    )
                return JsonRpcResponse.success(
                    result={"session_id": created_session_id},
                    request_id=request_id,
                )

            if method == "openenv/session/close":
                target_session_id = params.get("session_id")
                if not target_session_id:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_PARAMS,
                        "Invalid params - 'session_id' is required",
                        request_id=request_id,
                    )

                if session_id is not None and target_session_id == session_id:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_REQUEST,
                        "Cannot close active WebSocket-managed session via MCP method",
                        request_id=request_id,
                    )

                async with self._session_lock:
                    env = self._sessions.pop(target_session_id, _MISSING)
                    if env is not _MISSING:
                        executor = self._session_executors.pop(target_session_id, None)
                        stack = self._session_stacks.pop(target_session_id, None)
                        self._session_info.pop(target_session_id, None)
                    else:
                        executor = None
                        stack = None

                if env is _MISSING:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_PARAMS,
                        f"Unknown session_id: {target_session_id}",
                        request_id=request_id,
                    )

                if env is None:
                    # Session slot reserved but env factory still running;
                    # re-insert the placeholder AND the executor so
                    # _create_session can finish and the executor remains
                    # tracked for eventual shutdown.
                    async with self._session_lock:
                        self._sessions[target_session_id] = None
                        if executor is not None:
                            self._session_executors[target_session_id] = executor
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_REQUEST,
                        f"Session {target_session_id} is still initializing; retry shortly",
                        request_id=request_id,
                    )

                # env/executor/stack cleanup outside the lock
                await self._cleanup_session_resources(env, executor, stack)
                return JsonRpcResponse.success(
                    result={"session_id": target_session_id, "closed": True},
                    request_id=request_id,
                )

            requested_session_id = params.get("session_id")
            managed_session_id = session_id

            # Use provided session environment or create temporary one
            if session_env is not None:
                _env = session_env
                should_close = False
            elif requested_session_id:
                async with self._session_lock:
                    _env = self._sessions.get(requested_session_id, _MISSING)

                if _env is _MISSING:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_PARAMS,
                        f"Unknown session_id: {requested_session_id}",
                        request_id=request_id,
                    )

                if _env is None:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_REQUEST,
                        f"Session {requested_session_id} is still initializing; retry shortly",
                        request_id=request_id,
                    )

                should_close = False
                managed_session_id = requested_session_id
            else:
                _env = self._env_factory()
                should_close = True
            try:
                mcp_client = getattr(_env, "mcp_client", None)
                mcp_server = getattr(_env, "mcp_server", None)
                mcp_session_factory = getattr(_env, "mcp_session", None)

                if method == McpMethod.TOOLS_LIST:
                    # Check if environment is MCP-enabled
                    if mcp_client is None and mcp_server is None:
                        return JsonRpcResponse.error_response(
                            JsonRpcErrorCode.INTERNAL_ERROR,
                            "Environment does not support MCP",
                            request_id=request_id,
                        )

                    if mcp_client:
                        if managed_session_id and mcp_client.is_connected():
                            # Session-managed with live transport — call
                            # directly, no redundant re-entry.
                            tools = await mcp_client.list_tools()
                        elif callable(mcp_session_factory):
                            # Stateless request, or session-managed but the
                            # background transport was lost: (re-)open.
                            mcp_session_cm = cast(
                                AsyncContextManager[Any], mcp_session_factory()
                            )
                            async with mcp_session_cm:
                                tools = await mcp_client.list_tools()
                        else:
                            async with mcp_client:
                                tools = await mcp_client.list_tools()

                        return JsonRpcResponse.success(
                            result={
                                "tools": [
                                    t.model_dump()
                                    if hasattr(t, "model_dump")
                                    else dict(t)
                                    for t in tools
                                ]
                            },
                            request_id=request_id,
                        )

                    if mcp_server:
                        tools = []
                        for _tool_name, tool in get_server_tools(mcp_server).items():
                            tools.append(
                                {
                                    "name": tool.name,
                                    "description": tool.description or "",
                                    "inputSchema": tool.parameters or {},
                                }
                            )
                        return JsonRpcResponse.success(
                            result={"tools": tools},
                            request_id=request_id,
                        )

                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INTERNAL_ERROR,
                        "MCP server not available",
                        request_id=request_id,
                    )

                elif method == McpMethod.TOOLS_CALL:
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})

                    if mcp_client is None and mcp_server is None:
                        return JsonRpcResponse.error_response(
                            JsonRpcErrorCode.INTERNAL_ERROR,
                            "Environment does not support MCP",
                            request_id=request_id,
                        )

                    if not tool_name:
                        return JsonRpcResponse.error_response(
                            JsonRpcErrorCode.INVALID_PARAMS,
                            "Missing 'name' in params",
                            request_id=request_id,
                        )

                    if mcp_client:
                        if managed_session_id and mcp_client.is_connected():
                            # Session-managed with live transport.
                            result = await mcp_client.call_tool(
                                name=tool_name, arguments=arguments
                            )
                        elif callable(mcp_session_factory):
                            # Stateless request, or session-managed but the
                            # background transport was lost: (re-)open.
                            mcp_session_cm = cast(
                                AsyncContextManager[Any], mcp_session_factory()
                            )
                            async with mcp_session_cm:
                                result = await mcp_client.call_tool(
                                    name=tool_name, arguments=arguments
                                )
                        else:
                            async with mcp_client:
                                result = await mcp_client.call_tool(
                                    name=tool_name, arguments=arguments
                                )
                    elif mcp_server:
                        server_tools = get_server_tools(mcp_server)
                        if tool_name in server_tools:
                            tool = server_tools[tool_name]
                            if inspect.iscoroutinefunction(tool.fn):
                                result = await tool.fn(**arguments)
                            else:
                                result = tool.fn(**arguments)
                        else:
                            return JsonRpcResponse.error_response(
                                JsonRpcErrorCode.INVALID_PARAMS,
                                f"Tool not found: {tool_name}",
                                request_id=request_id,
                            )
                    else:
                        return JsonRpcResponse.error_response(
                            JsonRpcErrorCode.INTERNAL_ERROR,
                            "MCP server not available",
                            request_id=request_id,
                        )

                    # Ensure result is JSON serializable
                    serializable_result = _make_json_serializable(result)

                    return JsonRpcResponse.success(
                        result=serializable_result,
                        request_id=request_id,
                    )

                else:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.METHOD_NOT_FOUND,
                        f"Method not found: {method}",
                        request_id=request_id,
                    )

            except Exception as e:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INTERNAL_ERROR,
                    str(e),
                    request_id=request_id,
                )
            finally:
                if managed_session_id:
                    self._update_session_activity(
                        managed_session_id,
                        increment_step=(method == McpMethod.TOOLS_CALL),
                    )
                if should_close:
                    _env.close()

        # Register MCP WebSocket endpoint (available in both production and simulation modes)
        @app.websocket("/mcp")
        async def mcp_websocket_endpoint(websocket: WebSocket):
            """
            WebSocket endpoint for MCP JSON-RPC requests.

            Each WebSocket connection gets its own environment instance for MCP operations.

            Message Protocol:
            - Client sends: JSON-RPC 2.0 request (tools/list, tools/call)
            - Server responds: JSON-RPC 2.0 response (result or error)
            """
            await websocket.accept()

            session_id = None
            session_env = None

            try:
                # Create session with dedicated environment
                session_id, session_env = await self._create_session()
                if session_env is None:
                    raise RuntimeError(
                        "Session environment not initialized for MCP websocket"
                    )

                # If environment has an mcp_session context manager, hold it open
                # for the lifetime of the websocket connection

                async with AsyncExitStack() as stack:
                    mcp_session_factory = getattr(session_env, "mcp_session", None)
                    if callable(mcp_session_factory):
                        mcp_session_cm = cast(
                            AsyncContextManager[Any], mcp_session_factory()
                        )
                        await stack.enter_async_context(mcp_session_cm)

                    while True:
                        # Receive message from client
                        raw_message = await websocket.receive_text()

                        try:
                            jsonrpc_dict = json.loads(raw_message)
                            jsonrpc_request = JsonRpcRequest(**jsonrpc_dict)
                        except json.JSONDecodeError as e:
                            error_resp = JsonRpcResponse.error_response(
                                JsonRpcErrorCode.PARSE_ERROR,
                                f"Parse error: {e}",
                            )
                            await websocket.send_text(error_resp.model_dump_json())
                            continue
                        except ValidationError as e:
                            error_resp = JsonRpcResponse.error_response(
                                JsonRpcErrorCode.INVALID_REQUEST,
                                f"Invalid request: {e}",
                            )
                            await websocket.send_text(error_resp.model_dump_json())
                            continue

                        try:
                            # Call mcp_handler with session environment
                            response = await mcp_handler(
                                jsonrpc_request,
                                session_env=session_env,
                                session_id=session_id,
                            )
                            await websocket.send_text(response.model_dump_json())
                        except Exception as e:
                            error_resp = JsonRpcResponse.error_response(
                                JsonRpcErrorCode.INTERNAL_ERROR,
                                str(e),
                                request_id=jsonrpc_request.id,
                            )
                            await websocket.send_text(error_resp.model_dump_json())

            except WebSocketDisconnect:
                pass
            except SessionCapacityError as e:
                error_resp = JsonRpcResponse.error_response(
                    JsonRpcErrorCode.SERVER_ERROR,
                    str(e),
                    data={
                        "active_sessions": e.active_sessions,
                        "max_sessions": e.max_sessions,
                    },
                )
                await websocket.send_text(error_resp.model_dump_json())
            except EnvironmentFactoryError as e:
                error_resp = JsonRpcResponse.error_response(
                    JsonRpcErrorCode.SERVER_ERROR,
                    str(e),
                    data={"factory_name": e.factory_name},
                )
                await websocket.send_text(error_resp.model_dump_json())
            except Exception as e:
                error_resp = JsonRpcResponse.error_response(
                    JsonRpcErrorCode.SERVER_ERROR,
                    str(e),
                )
                await websocket.send_text(error_resp.model_dump_json())
            finally:
                if session_id:
                    await self._destroy_session(session_id)
                try:
                    await websocket.close()
                except RuntimeError:
                    pass

        # Register simulation control routes only in simulation mode
        if mode == ServerMode.SIMULATION:

            @app.post(
                "/reset",
                response_model=ResetResponse,
                tags=["Environment Control"],
                summary="Reset the environment",
                description="""
Reset the environment to its initial state and return the first observation.

You can optionally provide a seed for reproducibility and an episode_id for tracking.
                """,
                responses={
                    200: {
                        "description": "Environment reset successfully",
                        "content": {
                            "application/json": {
                                "example": {
                                    "observation": {"status": "ready", "data": {}},
                                    "reward": None,
                                    "done": False,
                                }
                            }
                        },
                    }
                },
            )
            async def reset(
                request: ResetRequest = Body(default_factory=ResetRequest),
            ) -> ResetResponse:
                return await reset_handler(request)

            @app.post(
                "/step",
                response_model=StepResponse,
                tags=["Environment Control"],
                summary="Execute an action in the environment",
                description="""
Execute an action in the environment and receive the resulting observation.

The action must conform to the environment's action schema, which can be
retrieved from the `/schema` endpoint. If the action is invalid,
the endpoint will return HTTP 422 with detailed validation errors.

The response includes:
- **observation**: The environment's response to the action
- **reward**: Optional reward signal (float or None)
- **done**: Boolean indicating if the episode has terminated
                """,
                responses={
                    200: {
                        "description": "Action executed successfully",
                        "content": {
                            "application/json": {
                                "example": {
                                    "observation": {"status": "success", "data": {}},
                                    "reward": 1.0,
                                    "done": False,
                                }
                            }
                        },
                    },
                    422: {
                        "description": "Validation error - invalid action format or values",
                        "content": {
                            "application/json": {
                                "example": {
                                    "detail": [
                                        {
                                            "type": "string_too_short",
                                            "loc": ["body", "action", "message"],
                                            "msg": "String should have at least 1 character",
                                            "input": "",
                                        }
                                    ]
                                }
                            }
                        },
                    },
                    500: {
                        "description": "Internal server error during action execution"
                    },
                },
            )
            async def step(request: StepRequest) -> StepResponse:
                return await step_handler(request)

        def get_state_handler() -> State:
            _env = self._env_factory()
            try:
                return _env.state
            finally:
                _env.close()

        def get_metadata_handler() -> EnvironmentMetadata:
            _env = self._env_factory()
            try:
                return _env.get_metadata()
            finally:
                _env.close()

        # Build list of GET endpoints based on mode
        get_endpoints = [
            GetEndpointConfig(
                path="/metadata",
                handler=get_metadata_handler,
                response_model=EnvironmentMetadata,
                tag="Environment Info",
                summary="Get environment metadata",
                description="""
Get metadata about this environment.

Returns information about the environment including name, description,
version, author, and documentation links.
                """,
            ),
            GetEndpointConfig(
                path="/health",
                handler=lambda: HealthResponse(status=HealthStatus.HEALTHY),
                response_model=HealthResponse,
                tag="Health",
                summary="Health check",
                description="Check if the environment server is running and healthy.",
            ),
        ]

        # Only register /state endpoint in simulation mode
        if mode == ServerMode.SIMULATION:
            get_endpoints.insert(
                0,
                GetEndpointConfig(
                    path="/state",
                    handler=get_state_handler,
                    response_model=State,
                    tag="State Management",
                    summary="Get current environment state",
                    description="""
Retrieve the current internal state of the environment.

The structure of the state object is defined by the environment's State model.
                    """,
                ),
            )

        register_get_endpoints(app, get_endpoints)

        # Register combined schema endpoint
        @app.get(
            "/schema",
            response_model=SchemaResponse,
            tags=["Schema"],
            summary="Get all JSON schemas",
            description="""
Get JSON schemas for actions, observations, and state in a single response.

Returns a combined schema object containing:
- **action**: JSON schema for actions accepted by this environment
- **observation**: JSON schema for observations returned by this environment
- **state**: JSON schema for environment state objects

This is more efficient than calling individual schema endpoints and provides
all schema information needed to interact with the environment.
            """,
            responses={
                200: {
                    "description": "Combined schemas retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "action": {
                                    "type": "object",
                                    "properties": {"message": {"type": "string"}},
                                },
                                "observation": {
                                    "type": "object",
                                    "properties": {"response": {"type": "string"}},
                                },
                                "state": {
                                    "type": "object",
                                    "properties": {"step_count": {"type": "integer"}},
                                },
                            }
                        }
                    },
                }
            },
        )
        async def get_schemas() -> SchemaResponse:
            """Return all schemas in one response."""
            return SchemaResponse(
                action=self.action_cls.model_json_schema(),
                observation=self.observation_cls.model_json_schema(),
                state=State.model_json_schema(),
            )

        # Register MCP endpoint for production mode (direct MCP access)
        @app.post("/mcp")
        async def mcp_endpoint(request_raw: Request) -> Dict[str, Any]:
            """
            MCP JSON-RPC endpoint for production mode.

            Bypasses step() overhead and provides direct access to MCP tools.
            Supports tools/list and tools/call methods.
            """
            # Parse JSON manually to handle parse errors gracefully
            try:
                body = await request_raw.body()
                request_dict = json.loads(body)
                request = JsonRpcRequest(**request_dict)
            except json.JSONDecodeError:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.PARSE_ERROR
                ).model_dump()
            except ValidationError as e:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INVALID_REQUEST,
                    f"Invalid request: {e}",
                ).model_dump()
            except Exception:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.PARSE_ERROR
                ).model_dump()

            response = await mcp_handler(request)
            return response.model_dump()

        # Register WebSocket endpoint for persistent sessions
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """
            WebSocket endpoint for persistent environment sessions.

            Each WebSocket connection gets its own environment instance.

            Message Protocol:
            - Client sends: WSResetMessage | WSStepMessage | WSStateMessage | WSCloseMessage
            - Server responds: WSObservationResponse | WSStateResponse | WSErrorResponse
            """
            await websocket.accept()

            session_id = None
            session_env = None

            try:
                # Create session with dedicated environment
                session_id, session_env = await self._create_session()
                if session_env is None:
                    raise RuntimeError(
                        "Session environment not initialized for websocket"
                    )

                # Keep MCP session open for entire websocket lifetime
                # (avoids reconnect overhead on every message)

                async with AsyncExitStack() as stack:
                    mcp_session_factory = getattr(session_env, "mcp_session", None)
                    if callable(mcp_session_factory):
                        mcp_session_cm = cast(
                            AsyncContextManager[Any], mcp_session_factory()
                        )
                        await stack.enter_async_context(mcp_session_cm)

                    while True:
                        # Receive message from client
                        raw_message = await websocket.receive_text()

                        try:
                            message_dict = json.loads(raw_message)
                        except json.JSONDecodeError as e:
                            error_resp = WSErrorResponse(
                                data={
                                    "message": f"Invalid JSON: {e}",
                                    "code": WSErrorCode.INVALID_JSON,
                                }
                            )
                            await websocket.send_text(error_resp.model_dump_json())
                            continue

                        msg_type = message_dict.get("type", "")

                        try:
                            match msg_type:
                                case "reset":
                                    msg = WSResetMessage(**message_dict)

                                    is_async = (
                                        session_env.reset_async.__func__
                                        is not Environment.reset_async
                                    )

                                    if is_async:
                                        sig = inspect.signature(session_env.reset_async)
                                        valid_kwargs = self._get_valid_kwargs(
                                            sig, msg.data
                                        )
                                        observation = await session_env.reset_async(
                                            **valid_kwargs
                                        )
                                    else:
                                        sig = inspect.signature(session_env.reset)
                                        valid_kwargs = self._get_valid_kwargs(
                                            sig, msg.data
                                        )
                                        observation = (
                                            await self._run_in_session_executor(
                                                session_id,
                                                session_env.reset,
                                                **valid_kwargs,
                                            )
                                        )

                                    self._update_session_activity(session_id)

                                    response = WSObservationResponse(
                                        data=serialize_observation(observation),
                                    )

                                case "step":
                                    msg = WSStepMessage(**message_dict)
                                    action = deserialize_action(
                                        msg.data, self.action_cls
                                    )

                                    is_async = (
                                        session_env.step_async.__func__
                                        is not Environment.step_async
                                    )

                                    if is_async:
                                        observation = await session_env.step_async(
                                            action
                                        )
                                    else:
                                        observation = (
                                            await self._run_in_session_executor(
                                                session_id, session_env.step, action
                                            )
                                        )

                                    self._update_session_activity(
                                        session_id, increment_step=True
                                    )

                                    response = WSObservationResponse(
                                        data=serialize_observation(observation)
                                    )

                                case "state":
                                    msg = WSStateMessage(**message_dict)
                                    state = session_env.state
                                    if hasattr(state, "model_dump"):
                                        state_data = state.model_dump()
                                    else:
                                        state_data = dict(state) if state else {}

                                    response = WSStateResponse(data=state_data)

                                case "close":
                                    msg = WSCloseMessage(**message_dict)
                                    break

                                case "mcp":
                                    msg = WSMCPMessage(**message_dict)
                                    try:
                                        rpc_request = JsonRpcRequest(**msg.data)
                                    except (ValidationError, Exception) as e:
                                        rpc_response = JsonRpcResponse.error_response(
                                            JsonRpcErrorCode.INVALID_REQUEST,
                                            f"Invalid request: {e}",
                                        )
                                    else:
                                        rpc_response = await mcp_handler(
                                            rpc_request,
                                            session_env=session_env,
                                            session_id=session_id,
                                        )
                                    response = WSMCPResponse(
                                        data=rpc_response.model_dump()
                                    )

                                case _:
                                    response = WSErrorResponse(
                                        data={
                                            "message": f"Unknown message type: {msg_type}",
                                            "code": WSErrorCode.UNKNOWN_TYPE,
                                        }
                                    )

                            await websocket.send_text(response.model_dump_json())

                        except ValidationError as e:
                            error_resp = WSErrorResponse(
                                data={
                                    "message": "Invalid message",
                                    "code": WSErrorCode.VALIDATION_ERROR,
                                    "errors": e.errors(),
                                }
                            )
                            await websocket.send_text(error_resp.model_dump_json())
                        except Exception as e:
                            error_resp = WSErrorResponse(
                                data={
                                    "message": str(e),
                                    "code": WSErrorCode.EXECUTION_ERROR,
                                }
                            )
                            await websocket.send_text(error_resp.model_dump_json())

            except WebSocketDisconnect:
                pass
            except SessionCapacityError as e:
                error_resp = WSErrorResponse(
                    data={
                        "message": str(e),
                        "code": WSErrorCode.CAPACITY_REACHED,
                        "active_sessions": e.active_sessions,
                        "max_sessions": e.max_sessions,
                    }
                )
                await websocket.send_text(error_resp.model_dump_json())
            except EnvironmentFactoryError as e:
                error_resp = WSErrorResponse(
                    data={
                        "message": str(e),
                        "code": WSErrorCode.FACTORY_ERROR,
                        "factory_name": e.factory_name,
                    }
                )
                await websocket.send_text(error_resp.model_dump_json())
            except Exception as e:
                error_resp = WSErrorResponse(
                    data={"message": str(e), "code": WSErrorCode.SESSION_ERROR}
                )
                await websocket.send_text(error_resp.model_dump_json())
            finally:
                if session_id:
                    await self._destroy_session(session_id)
                try:
                    await websocket.close()
                except RuntimeError:
                    pass


def create_app(
    env: Callable[[], Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[ConcurrencyConfig] = None,
    gradio_builder: Optional[Callable[..., Any]] = None,
) -> FastAPI:
    """
    Create a FastAPI application with or without web interface.

    This function creates a FastAPI app with the web interface enabled by default,
    including README integration for better user experience.

    Args:
        env: Environment factory (callable) that creates new instances
        action_cls: The Action subclass this environment expects
        observation_cls: The Observation subclass this environment returns
        env_name: Optional environment name for README loading
        max_concurrent_envs: Maximum concurrent WebSocket sessions.
                             Mutually exclusive with concurrency_config.
        concurrency_config: Optional ConcurrencyConfig for advanced concurrency settings.
                            Mutually exclusive with max_concurrent_envs.
        gradio_builder: Optional callable to build a custom Gradio UI at /web.
            Signature: (web_manager, action_fields, metadata, is_chat_env, title,
            quick_start_md) -> gr.Blocks. When None, the default Gradio app is used.
            See docs/customizing-web-ui.md.

    Returns:
        FastAPI application instance with or without web interface and README integration
    """
    # Check if web interface should be enabled
    # This can be controlled via environment variable or build argument
    enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    if enable_web:
        # Gradio-based web UI (gradio is a core dependency)
        from .web_interface import create_web_interface_app

        return create_web_interface_app(
            cast(Any, env),
            action_cls,
            observation_cls,
            env_name,
            max_concurrent_envs,
            concurrency_config,
            gradio_builder=gradio_builder,
        )
    else:
        # Use standard FastAPI app without web interface
        return create_fastapi_app(
            env, action_cls, observation_cls, max_concurrent_envs, concurrency_config
        )


def create_fastapi_app(
    env: Callable[[], Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[ConcurrencyConfig] = None,
) -> FastAPI:
    """
    Create a FastAPI application with comprehensive documentation.

    Args:
        env: Environment factory (callable) that creates new instances
        action_cls: The Action subclass this environment expects
        observation_cls: The Observation subclass this environment returns
        max_concurrent_envs: Maximum concurrent WebSocket sessions.
                             Mutually exclusive with concurrency_config.
        concurrency_config: Optional ConcurrencyConfig for advanced concurrency settings.
                            Mutually exclusive with max_concurrent_envs.

    Returns:
        FastAPI application instance
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError(
            "FastAPI is required. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="OpenEnv Environment HTTP API",
        version="1.0.0",
        description="""
# OpenEnv Environment HTTP API

HTTP API for interacting with OpenEnv environments through a standardized interface.

## Features

* **Environment Reset**: Initialize or restart episodes
* **Action Execution**: Send actions and receive observations
* **State Inspection**: Query current environment state
* **Schema Access**: Retrieve JSON schemas for actions and observations

## Workflow

1. Call `/reset` to start a new episode and get initial observation
2. Call `/step` repeatedly with actions to interact with environment
3. Episode ends when observation returns `done: true`
4. Call `/state` anytime to inspect current environment state

## Documentation

* **Swagger UI**: Available at `/docs`
* **ReDoc**: Available at `/redoc`
* **OpenAPI Schema**: Available at `/openapi.json`
        """,
        openapi_tags=[
            {
                "name": "Environment Control",
                "description": "Core operations for environment interaction (reset, step)",
            },
            {
                "name": "State Management",
                "description": "Operations for inspecting environment state",
            },
            {
                "name": "Environment Info",
                "description": "Information about the environment",
            },
            {
                "name": "Schema",
                "description": "JSON Schema endpoints for actions, observations, and state",
            },
            {"name": "Health", "description": "Service health and status checks"},
        ],
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "OpenEnv Team",
            "url": "https://github.com/meta-pytorch/OpenEnv",
        },
        license_info={
            "name": "BSD-3-Clause",
            "url": "https://github.com/meta-pytorch/OpenEnv/blob/main/LICENSE",
        },
    )

    server = HTTPEnvServer(
        env,
        action_cls,
        observation_cls,
        max_concurrent_envs,
        concurrency_config=concurrency_config,
    )
    server.register_routes(app)
    return app
