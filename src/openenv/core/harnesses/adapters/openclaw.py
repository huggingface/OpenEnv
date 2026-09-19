# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenClaw harness adapter (RFC 005).

OpenClaw is an open-source agentic platform that supports MCP tools,
multi-provider LLM backends, and multi-channel communication.

This adapter manages the OpenClaw process lifecycle, injects MCP tool
configurations, and communicates via stdin/stdout.

OpenClaw stores its config at ~/.openclaw/openclaw.json and supports
MCP server configuration via mcpServers entries in the config file.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from openenv.core.harnesses.adapter import HarnessAdapter
from openenv.core.harnesses.types import (
    HarnessConfig,
    HarnessEvent,
    HarnessEventType,
    HarnessResponse,
)


class OpenClawAdapter(HarnessAdapter):
    """Adapter for the OpenClaw agentic harness.

    Manages an OpenClaw process, injects environment MCP tools into
    OpenClaw's config, and communicates via stdin/stdout.

    OpenClaw config is written to the working directory at
    ``.openclaw/openclaw.json`` (or the path specified in
    ``config.mcp_config_path``).
    """

    def __init__(self, config: HarnessConfig) -> None:
        super().__init__(config)
        self._process: Optional[asyncio.subprocess.Process] = None
        self._injected_tools: List[Dict[str, Any]] = []

    async def start(self, working_directory: str) -> None:
        """Start the OpenClaw process."""
        # Start with parent env so PATH, PYTHONPATH, etc. are preserved,
        # then layer on configured and API key env vars.
        env: Optional[Dict[str, str]] = None
        if self.config.env_vars or self.config.api_key_env_var:
            env = dict(os.environ)
            env.update(self.config.env_vars)
            if self.config.api_key_env_var:
                key = os.environ.get(self.config.api_key_env_var, "")
                if key:
                    env[self.config.api_key_env_var] = key

        cmd = list(self.config.command)
        if self.config.model:
            cmd.extend(["--model", self.config.model])

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory,
            env=env,
        )

    async def stop(self) -> None:
        """Stop the OpenClaw process."""
        if self._process is not None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._process.kill()
            finally:
                self._process = None

    async def inject_tools(self, tools: List) -> None:
        """Write MCP server config for OpenClaw to discover.

        Creates an ``openclaw.json`` config file that registers an
        ``openenv`` MCP server pointing to the environment's MCP bridge.

        Args:
            tools: List of MCP tool definitions from the environment.
        """
        self._injected_tools = [{"name": getattr(t, "name", str(t))} for t in tools]

        if not tools:
            return

        config_path = self.config.mcp_config_path or str(
            Path(self.config.working_directory) / ".openclaw" / "openclaw.json"
        )
        mcp_config = {
            "mcpServers": {
                "openenv": {
                    "command": "openenv-mcp-bridge",
                    "args": ["--tools", json.dumps([t for t in self._injected_tools])],
                }
            }
        }

        path = Path(config_path)
        # Merge with existing config if present
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        if "mcpServers" not in existing:
            existing["mcpServers"] = {}
        existing["mcpServers"]["openenv"] = mcp_config["mcpServers"]["openenv"]

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2))

    async def send_message(self, message: str) -> HarnessResponse:
        """Send a message to OpenClaw via stdin and read the response.

        The message is sent as a JSON line and the response is read
        as a JSON line from stdout.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("OpenClaw process is not running")

        events: List[HarnessEvent] = []
        ts = time.time()

        # Send message as JSON line
        payload = json.dumps({"message": message}) + "\n"
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()

        events.append(
            HarnessEvent(
                type=HarnessEventType.LLM_REQUEST,
                timestamp=ts,
                data={"message": message},
            )
        )

        # Read response line
        if self._process.stdout is None:
            raise RuntimeError("OpenClaw stdout is not available")

        try:
            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self.config.session_timeout_s,
            )
        except asyncio.TimeoutError:
            events.append(
                HarnessEvent(
                    type=HarnessEventType.ERROR,
                    timestamp=time.time(),
                    data={"message": "Session timeout", "recoverable": False},
                )
            )
            return HarnessResponse(
                response="Error: session timeout",
                events=events,
                done=True,
            )

        response_text = line.decode().strip()
        done = False

        # Try to parse as JSON
        try:
            data = json.loads(response_text)
            response_text = data.get("response", response_text)
            done = data.get("done", False)

            # Extract tool call events if present
            for tool_event in data.get("tool_calls", []):
                events.append(
                    HarnessEvent(
                        type=HarnessEventType.TOOL_CALL,
                        timestamp=time.time(),
                        data=tool_event,
                    )
                )
        except json.JSONDecodeError:
            pass

        events.append(
            HarnessEvent(
                type=HarnessEventType.TURN_COMPLETE,
                timestamp=time.time(),
                data={"response": response_text},
            )
        )

        return HarnessResponse(
            response=response_text,
            events=events,
            done=done,
        )

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        """Send a message and stream events as they arrive."""
        response = await self.send_message(message)
        for event in response.events:
            yield event

    async def is_alive(self) -> bool:
        """Check if the OpenClaw process is still running."""
        if self._process is None:
            return False
        return self._process.returncode is None
