# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pi CLI harness adapter."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import as_file, files
from typing import Any, Callable

from . import (
    _serialize_for_message,
    CLIHarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    Message,
    ResourceSession,
    RolloutEvent,
    SessionMCPBridge,
    ToolResult,
    ToolTraceEntry,
)


def _messages_to_prompt(messages: list[Message]) -> str:
    if len(messages) == 1 and isinstance(messages[0].get("content"), str):
        return str(messages[0]["content"])
    parts = []
    for message in messages:
        role = str(message.get("role", "message"))
        content = message.get("content", "")
        parts.append(f"{role}:\n{_serialize_for_message(content)}")
    return "\n\n".join(parts)


def _json_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"type": "raw", "text": line}
        if isinstance(event, dict):
            events.append(event)
    return events


def _messages_from_events(events: list[dict[str, Any]], stdout: str) -> list[Message]:
    for event in reversed(events):
        messages = event.get("messages")
        if event.get("type") == "agent_end" and isinstance(messages, list):
            return [message for message in messages if isinstance(message, dict)]

    messages = [
        event["message"]
        for event in events
        if event.get("type") == "message_end" and isinstance(event.get("message"), dict)
    ]
    if messages:
        return messages
    return [{"role": "assistant", "content": stdout}]


def _response_body(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _bridge_handler(
    bridge: SessionMCPBridge,
    tool_trace: list[ToolTraceEntry],
) -> type[BaseHTTPRequestHandler]:
    class BridgeHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return None

        def do_POST(self) -> None:
            try:
                content_length = int(self.headers.get("content-length", "0"))
                request = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except Exception as exc:
                _response_body(
                    self,
                    400,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": str(exc)},
                    },
                )
                return

            response = bridge.handle_request(request)
            if request.get("method") == "tools/call":
                params = request.get("params", {}) or {}
                result = response.get("result")
                if isinstance(result, dict):
                    tool_trace.append(
                        ToolTraceEntry(
                            tool_name=str(params.get("name", "")),
                            arguments=dict(params.get("arguments", {}) or {}),
                            result=ToolResult(
                                data=result.get("data"),
                                done=bool(result.get("done")),
                                metadata=dict(result.get("metadata", {}) or {}),
                                error=result.get("error"),
                            ),
                        )
                    )
            _response_body(self, 200, response)

    return BridgeHandler


class PiCLIHarnessAdapter(CLIHarnessAdapter):
    """Black-box harness adapter that drives an OpenEnv session with the Pi CLI."""

    def __init__(
        self,
        *,
        pi_command: str | list[str] | tuple[str, ...] = "pi",
        provider: str | None = None,
        model: str | None = None,
        cwd: str | None = None,
        timeout_s: float = 600.0,
        extra_args: list[str] | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ):
        self._pi_command = (
            [pi_command] if isinstance(pi_command, str) else list(pi_command)
        )
        self._provider = provider
        self._model = model
        self._cwd = cwd
        self._timeout_s = timeout_s
        self._extra_args = list(extra_args or [])
        self._command_runner = command_runner or subprocess.run
        super().__init__(runner=self._run_pi)

    def _run_pi(
        self,
        bridge: SessionMCPBridge,
        session: ResourceSession,
        limits: HarnessRunLimits,
    ) -> HarnessRolloutResult:
        del limits
        tools = session.list_tools()
        tool_trace: list[ToolTraceEntry] = []
        bridge_resource = files("openenv.core.harness").joinpath("pi_bridge.mjs")

        with as_file(bridge_resource) as extension_path:
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                _bridge_handler(bridge, tool_trace),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            bridge_url = f"http://127.0.0.1:{server.server_port}"

            env = dict(os.environ)
            env["OPENENV_PI_BRIDGE_URL"] = bridge_url

            command = [
                *self._pi_command,
                "--mode",
                "json",
                "--print",
                "--no-session",
                "--no-builtin-tools",
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-context-files",
                "--extension",
                str(extension_path),
                "--tools",
                ",".join(tool.name for tool in tools),
            ]
            if self._provider:
                command.extend(["--provider", self._provider])
            if self._model:
                command.extend(["--model", self._model])
            command.extend(self._extra_args)
            command.append(_messages_to_prompt(session.initial_messages()))

            try:
                completed = self._command_runner(
                    command,
                    cwd=self._cwd,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=self._timeout_s,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=1.0)

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode != 0:
            raise RuntimeError(
                "pi CLI failed with exit code "
                f"{completed.returncode}: {(stderr or stdout).strip()}"
            )

        events = _json_events(stdout)
        return HarnessRolloutResult(
            messages=_messages_from_events(events, stdout),
            tool_trace=tool_trace,
            events=[RolloutEvent(type="pi_event", payload=event) for event in events],
            done=bool(tool_trace and tool_trace[-1].result.done),
            metrics={
                "harness": "pi_cli",
                "pi_events": len(events),
                "tool_calls": len(tool_trace),
                "stderr": stderr,
            },
        )


__all__ = ["PiCLIHarnessAdapter"]
