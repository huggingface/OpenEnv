# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Web interface for OpenEnv environments.

When ENABLE_WEB_INTERFACE is set, the server exposes a Gradio UI at /web for
reset, step, and state observation. Controlled by the CLI enable_interface
option (e.g. openenv push --enable-interface) or ENABLE_WEB_INTERFACE env var.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

import gradio as gr
from fastapi import Body, FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from .gradio_theme import OPENENV_GRADIO_CSS, OPENENV_GRADIO_THEME
from .gradio_ui import build_gradio_app, get_gradio_display_title
from .interfaces import Environment
from .serialization import deserialize_action_with_preprocessing, serialize_observation
from .types import Action, EnvironmentMetadata, Observation, State

# Quick Start markdown template; placeholders match init suffixes (__ENV_NAME__, __ENV_CLASS_NAME__*).
DEFAULT_QUICK_START_MARKDOWN = """
### Connect to this environment

Connect from Python using `__ENV_CLASS_NAME__Env`:

```python
from __ENV_NAME__ import __ENV_CLASS_NAME__Action, __ENV_CLASS_NAME__Env

with __ENV_CLASS_NAME__Env.from_env("<SPACE_ID>") as env:
    result = await env.step(__ENV_CLASS_NAME__Action(message="..."))
```

Or connect directly to a running server:

```python
env = __ENV_CLASS_NAME__Env(base_url="http://localhost:8000")
```

### Contribute to this environment

Submit improvements via pull request on the Hugging Face Hub.

```bash
openenv fork <SPACE_ID> --repo-id <your-username>/<your-repo-name>
```

Then make your changes and submit a pull request:

```bash
cd <forked-repo>
openenv push <SPACE_ID> --create-pr
```

For more information, see the [OpenEnv documentation](https://meta-pytorch.org/OpenEnv/).
"""


def get_quick_start_markdown(
    metadata: Optional[EnvironmentMetadata],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
) -> str:
    """
    Build Quick Start markdown with class names replaced from current env (init-style suffixes).

    Uses the same placeholder names as the init template so that __ENV_CLASS_NAME__Env,
    __ENV_CLASS_NAME__Action, __ENV_CLASS_NAME__Observation and __ENV_NAME__ are
    replaced with the actual class/package names.
    """
    import os

    # Prefix from action class (e.g. EchoAction -> Echo)
    action_name = getattr(action_cls, "__name__", "Action")
    if action_name.endswith("Action"):
        prefix = action_name[: -len("Action")]
    else:
        prefix = action_name.replace("Action", "").strip() or "Env"

    env_client_name = f"{prefix}Env"
    obs_name = getattr(observation_cls, "__name__", "Observation")
    pkg_name = (metadata.name if metadata else "env").replace(" ", "_").lower()

    space_id = os.environ.get("SPACE_ID", "<hf-username>/<hf-repo-name>")

    content = DEFAULT_QUICK_START_MARKDOWN
    content = content.replace("__ENV_CLASS_NAME__Env", env_client_name)
    content = content.replace("__ENV_CLASS_NAME__Action", action_name)
    content = content.replace("__ENV_CLASS_NAME__Observation", obs_name)
    content = content.replace("__ENV_CLASS_NAME__", prefix)
    content = content.replace("__ENV_NAME__", pkg_name)
    content = content.replace("<SPACE_ID>", space_id)
    return content.strip()


def load_environment_metadata(
    env: Environment, env_name: Optional[str] = None
) -> EnvironmentMetadata:
    """
    Load environment metadata including README content.

    Args:
        env: The environment instance, class, or factory function.
             - If a class: used as a factory, won't call instance methods
             - If a function: used as a factory, won't call instance methods
             - If an instance: may call get_metadata() if available
        env_name: Optional environment name for README file lookup

    Returns:
        EnvironmentMetadata with loaded information
    """
    import inspect

    # Determine what type of env we received:
    # 1. A class (used as factory) - e.g., PythonCodeActEnv
    # 2. A function (factory function) - e.g., create_chat_environment
    # 3. An actual instance - e.g., SnakeEnvironment()
    is_class = inspect.isclass(env)
    is_function = inspect.isfunction(env) or inspect.ismethod(env)
    is_factory = is_class or is_function

    # Try to get metadata from environment if it's an instance with get_metadata
    if not is_factory and hasattr(env, "get_metadata"):
        return env.get_metadata()

    # Determine the class name for default metadata
    if is_class:
        # env is the class itself
        class_name = env.__name__
    elif is_function:
        # env is a factory function - use its name or derive from env_name
        class_name = env_name or env.__name__
    else:
        # env is an instance
        class_name = env.__class__.__name__

    # Default metadata
    metadata = EnvironmentMetadata(
        name=env_name or class_name,
        description=f"{class_name} environment",
        version="1.0.0",
    )

    # Try to load README from file system
    readme_content = _load_readme_from_filesystem(env_name)
    if readme_content:
        metadata.readme_content = readme_content

    return metadata


def _load_readme_from_filesystem(env_name: Optional[str]) -> Optional[str]:
    """
    Load README content from the filesystem.

    Tries multiple locations:
    1. Container filesystem: /app/README.md
    2. Local development: src/envs/{env_name}/README.md
    3. Environment variable: ENV_README_PATH
    """
    import os
    from pathlib import Path

    # Try container filesystem first
    container_readme = Path("/app/README.md")
    if container_readme.exists():
        try:
            return container_readme.read_text(encoding="utf-8")
        except Exception:
            pass

    # Try environment variable path
    custom_path = os.environ.get("ENV_README_PATH")
    if custom_path and Path(custom_path).exists():
        try:
            return Path(custom_path).read_text(encoding="utf-8")
        except Exception:
            pass

    # Try local development path
    if env_name:
        local_readme = Path(f"src/envs/{env_name}/README.md")
        if local_readme.exists():
            try:
                return local_readme.read_text(encoding="utf-8")
            except Exception:
                pass

    return None


class ActionLog(BaseModel):
    """Log entry for an action taken."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    timestamp: str = Field(description="Timestamp when action was taken")
    action: Dict[str, Any] = Field(description="Action that was taken")
    observation: Dict[str, Any] = Field(description="Observation returned from action")
    reward: Optional[float] = Field(
        default=None, description="Reward received from action"
    )
    done: bool = Field(description="Whether the episode is done after this action")
    step_count: int = Field(description="Step count when this action was taken")


class EpisodeState(BaseModel):
    """Current episode state for the web interface."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: Optional[str] = Field(default=None, description="Current episode ID")
    step_count: int = Field(description="Current step count in episode")
    current_observation: Optional[Dict[str, Any]] = Field(
        default=None, description="Current observation"
    )
    action_logs: List[ActionLog] = Field(
        default_factory=list, description="List of action logs"
    )
    is_reset: bool = Field(
        default=True, description="Whether the episode has been reset"
    )


class WebInterfaceManager:
    """Manages the web interface for an environment."""

    MAX_ACTION_LOGS = 1000

    def __init__(
        self,
        env: Environment,
        action_cls: Type[Action],
        observation_cls: Type[Observation],
        metadata: Optional[EnvironmentMetadata] = None,
    ):
        import inspect

        # If env is a class or factory function, instantiate it
        if inspect.isclass(env) or inspect.isfunction(env):
            self.env = env()
        else:
            self.env = env
        self.action_cls = action_cls
        self.observation_cls = observation_cls
        self.metadata = metadata or EnvironmentMetadata(
            name=env.__class__.__name__,
            description=f"{env.__class__.__name__} environment",
        )
        self.episode_state = EpisodeState(
            episode_id=None,
            step_count=0,
            current_observation=None,
            action_logs=[],
        )
        self.connected_clients: List[WebSocket] = []
        # Thread pool for running sync code (e.g., Playwright sync API) in async context
        self._executor = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _get_valid_kwargs(
        sig: inspect.Signature,
        kwargs: Dict[str, Any],
        skip_params: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """Filter kwargs to only those accepted by the target function."""
        skip_params = skip_params or set()
        valid_kwargs: Dict[str, Any] = {}
        has_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )

        for key, value in kwargs.items():
            if key in skip_params:
                continue
            if key in sig.parameters or has_var_kwargs:
                valid_kwargs[key] = value

        return valid_kwargs

    async def _run_sync_in_thread_pool(self, func, *args, **kwargs):
        """Run a synchronous function in the thread pool executor.

        This is needed for environments using sync libraries (e.g., Playwright sync API)
        that cannot be called directly from an async context.
        """
        loop = asyncio.get_event_loop()
        # Use default arguments to capture values at lambda definition time
        # to avoid closure issues with late binding
        return await loop.run_in_executor(
            self._executor, lambda f=func, a=args, kw=kwargs: f(*a, **kw)
        )

    async def connect_websocket(self, websocket: WebSocket):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.connected_clients.append(websocket)

        # Send current state to the new client
        await self._send_state_update()

    async def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect a WebSocket client."""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)

    async def _send_state_update(self):
        """Send current state to all connected clients."""
        if not self.connected_clients:
            return

        state_data = {
            "type": "state_update",
            "episode_state": self.episode_state.model_dump(),
        }

        # Send to all connected clients
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(state_data))
            except Exception:
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.remove(client)

    async def reset_environment(
        self, reset_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Reset the environment and update state."""
        reset_kwargs = reset_kwargs or {}

        is_async = self.env.reset_async.__func__ is not Environment.reset_async
        sig = inspect.signature(self.env.reset_async if is_async else self.env.reset)
        valid_kwargs = self._get_valid_kwargs(sig, reset_kwargs)

        if is_async:
            observation = await self.env.reset_async(**valid_kwargs)
        else:
            # Run sync reset in thread pool to avoid blocking event loop
            # and to support environments using sync libraries (e.g., Playwright)
            observation = await self._run_sync_in_thread_pool(
                self.env.reset, **valid_kwargs
            )
        state: State = self.env.state

        # Serialize observation once using shared utility
        serialized = serialize_observation(observation)

        # Update episode state
        self.episode_state.episode_id = state.episode_id
        self.episode_state.step_count = 0
        self.episode_state.current_observation = serialized["observation"]
        self.episode_state.action_logs = []
        self.episode_state.is_reset = True

        # Send state update
        await self._send_state_update()

        return serialized

    async def step_environment(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step in the environment and update state."""
        # Deserialize action with preprocessing for web interface special cases
        action: Action = deserialize_action_with_preprocessing(
            action_data, self.action_cls
        )

        # Run sync step in thread pool to avoid blocking event loop
        # and to support environments using sync libraries (e.g., Playwright)
        observation: Observation = await self._run_sync_in_thread_pool(
            self.env.step, action
        )
        state: State = self.env.state

        # Serialize observation once using shared utility
        serialized = serialize_observation(observation)

        # Create action log
        action_log = ActionLog(
            timestamp=datetime.now().isoformat(),
            action=action.model_dump(exclude={"metadata"}),
            observation=serialized["observation"],
            reward=observation.reward,
            done=observation.done,
            step_count=state.step_count,
        )

        # Update episode state
        self.episode_state.episode_id = state.episode_id
        self.episode_state.step_count = state.step_count
        self.episode_state.current_observation = serialized["observation"]
        self.episode_state.action_logs.append(action_log)
        if len(self.episode_state.action_logs) > self.MAX_ACTION_LOGS:
            self.episode_state.action_logs = self.episode_state.action_logs[
                -self.MAX_ACTION_LOGS :
            ]
        self.episode_state.is_reset = False

        # Send state update
        await self._send_state_update()

        return serialized

    def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        state: State = self.env.state
        return state.model_dump()


def create_web_interface_app(
    env: Environment,
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[Any] = None,
    gradio_builder: Optional[Callable[..., Any]] = None,
) -> FastAPI:
    """
    Create a FastAPI application with web interface for the given environment.

    Args:
        env: The Environment instance to serve
        action_cls: The Action subclass this environment expects
        observation_cls: The Observation subclass this environment returns
        env_name: Optional environment name for README loading
        max_concurrent_envs: Maximum concurrent WebSocket sessions
        concurrency_config: Optional ConcurrencyConfig for advanced concurrency settings
        gradio_builder: Optional callable (web_manager, action_fields, metadata,
            is_chat_env, title, quick_start_md) -> gr.Blocks to use instead of the
            default Gradio UI. Lets envs replace or customize the /web interface.

    Returns:
        FastAPI application instance with web interface
    """
    from .http_server import create_fastapi_app

    # Create the base environment app
    app = create_fastapi_app(
        env, action_cls, observation_cls, max_concurrent_envs, concurrency_config
    )

    # Load environment metadata
    metadata = load_environment_metadata(env, env_name)

    # Create web interface manager
    web_manager = WebInterfaceManager(env, action_cls, observation_cls, metadata)

    # Web API routes first (so they take precedence over Gradio mount at /web)
    @app.get("/", include_in_schema=False)
    async def web_root():
        """Redirect the app root to the Gradio interface."""
        return RedirectResponse(url="/web/")

    @app.get("/web", include_in_schema=False)
    async def web_root_no_slash():
        """Redirect /web to /web/ for mounted Gradio deployments behind proxies."""
        return RedirectResponse(url="/web/")

    @app.get("/web/metadata")
    async def web_metadata():
        """Get environment metadata."""
        return web_manager.metadata.model_dump()

    @app.websocket("/ws/ui")
    async def websocket_ui_endpoint(websocket: WebSocket):
        """WebSocket endpoint for web UI real-time updates.

        Note: Uses /ws/ui to avoid conflict with /ws in http_server.py
        which is used for concurrent environment sessions.
        """
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    @app.post("/web/reset")
    async def web_reset(request: Optional[Dict[str, Any]] = Body(default=None)):
        """Reset endpoint for web interface."""
        return await web_manager.reset_environment(request)

    @app.post("/web/step")
    async def web_step(request: Dict[str, Any]):
        """Step endpoint for web interface."""
        # Check if this is a message-based request (chat environment)
        if "message" in request:
            message = request["message"]
            if hasattr(web_manager.env, "message_to_action"):
                action = web_manager.env.message_to_action(message)
                if hasattr(action, "tokens"):
                    action_data = {"tokens": action.tokens.tolist()}
                else:
                    action_data = action.model_dump(exclude={"metadata"})
            else:
                action_data = {"message": message}
        else:
            action_data = request.get("action", {})

        return await web_manager.step_environment(action_data)

    @app.get("/web/state")
    async def web_state():
        """State endpoint for web interface."""
        try:
            return web_manager.get_state()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    action_fields = _extract_action_fields(action_cls)
    is_chat_env = _is_chat_env(action_cls)
    quick_start_md = get_quick_start_markdown(metadata, action_cls, observation_cls)

    default_blocks = build_gradio_app(
        web_manager,
        action_fields,
        metadata,
        is_chat_env,
        title=metadata.name,
        quick_start_md=quick_start_md,
    )
    if gradio_builder is not None:
        custom_blocks = gradio_builder(
            web_manager,
            action_fields,
            metadata,
            is_chat_env,
            metadata.name,
            quick_start_md,
        )
        if not isinstance(custom_blocks, gr.Blocks):
            raise TypeError(
                f"gradio_builder must return a gr.Blocks instance, "
                f"got {type(custom_blocks).__name__}"
            )
        gradio_blocks = gr.TabbedInterface(
            [default_blocks, custom_blocks],
            tab_names=["Playground", "Custom"],
            title=get_gradio_display_title(metadata),
        )
    else:
        gradio_blocks = default_blocks
    app = gr.mount_gradio_app(
        app,
        gradio_blocks,
        path="/web",
        theme=OPENENV_GRADIO_THEME,
        css=OPENENV_GRADIO_CSS,
    )

    return app


def _is_chat_env(action_cls: Type[Action]) -> bool:
    """Return True if the action class is a chat-style env (tokens field)."""
    if hasattr(action_cls, "model_fields"):
        for field_name, field_info in action_cls.model_fields.items():
            if (
                field_name == "tokens"
                and hasattr(field_info.annotation, "__name__")
                and "Tensor" in str(field_info.annotation)
            ):
                return True
    return False


def _extract_action_fields(action_cls: Type[Action]) -> List[Dict[str, Any]]:
    """Extract enhanced field metadata from Action class for form generation."""
    # Use Pydantic's JSON schema generation for robust metadata extraction
    try:
        schema = action_cls.model_json_schema()
    except AttributeError:
        # Fallback for non-Pydantic v2 models or if something goes wrong
        return []

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    action_fields = []

    for field_name, field_info in properties.items():
        if field_name == "metadata":
            continue

        # JSON schema "type" can be a string or list/undefined
        # Determine our internal input type
        input_type = _determine_input_type_from_schema(field_info, field_name)

        is_required = field_name in required_fields

        action_fields.append(
            {
                "name": field_name,
                "type": input_type,
                "required": is_required,
                "description": field_info.get("description", ""),
                "default_value": field_info.get("default"),
                "choices": field_info.get("enum"),
                "min_value": field_info.get("minimum"),
                "max_value": field_info.get("maximum"),
                "min_length": field_info.get("minLength"),
                "max_length": field_info.get("maxLength"),
                "pattern": field_info.get("pattern"),
                "placeholder": _generate_placeholder(field_name, field_info),
                "help_text": _generate_help_text(field_name, field_info),
            }
        )

    return action_fields


def _determine_input_type_from_schema(
    field_info: Dict[str, Any], field_name: str
) -> str:
    """Determine input type from JSON schema for form generation (Gradio UI)."""
    schema_type = field_info.get("type")

    # Check for specific tensor field convention
    if "tokens" in field_name.lower():
        return "tensor"

    if "enum" in field_info:
        return "select"

    if schema_type == "boolean":
        return "checkbox"

    if schema_type == "integer" or schema_type == "number":
        return "number"

    if schema_type == "string":
        # Check if it should be a textarea
        if (
            field_info.get("maxLength", 0) > 100
            or "message" in field_name.lower()
            or "code" in field_name.lower()
        ):
            return "textarea"
        return "text"

    # Default fallback
    return "text"


def _generate_placeholder(field_name: str, field_info: Dict[str, Any]) -> str:
    """Generate placeholder text."""
    if "message" in field_name.lower():
        return f"Enter {field_name.replace('_', ' ')}..."
    elif "code" in field_name.lower():
        return "Enter Python code here..."
    elif "tokens" in field_name.lower():
        return "Enter comma-separated token IDs (e.g., 1,2,3,4,5)"
    else:
        return f"Enter {field_name.replace('_', ' ')}..."


def _generate_help_text(field_name: str, field_info: Dict[str, Any]) -> str:
    """Generate help text."""
    description = field_info.get("description", "")
    if description:
        return description

    if "action_id" in field_name.lower():
        return "The action ID to execute in environment"
    elif "game_name" in field_name.lower():
        return "Name of game or environment"
    elif "tokens" in field_name.lower():
        return "Token IDs as a comma-separated list of integers"
    elif "code" in field_name.lower():
        return "Python code to execute in environment"
    elif "message" in field_name.lower():
        return "Text message to send"

    return ""
