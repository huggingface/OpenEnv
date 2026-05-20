#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight MCP tool server deployed INSIDE the sandbox.

This script is written into the sandbox by the SWE environment before the
agent launches.  It exposes a single ``terminal`` tool over HTTP (streamable
HTTP MCP transport) that the agent discovers via its MCP config.

The ``terminal`` tool supports two call shapes:

    terminal(command="git diff HEAD")
        Execute a shell command in the workspace and return output.

    terminal(final_answer="I fixed the bug by ...")
        Signal that the agent is done.  The server runs the pre-configured
        verify commands, computes a reward, writes it to the reward file,
        and returns the verification outcome.

Configuration is read from a JSON file whose path is given by the
``SWE_MCP_CONFIG`` environment variable (default:
``/home/user/.swe_mcp_config.json``).

This file is stdlib-only so it works in bare sandbox images (no pip deps).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# ── Config ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = "/home/user/.swe_mcp_config.json"
DEFAULT_PORT = 8765
DEFAULT_WORKSPACE = "/home/user/workdir"
DEFAULT_COMMAND_TIMEOUT = 300
DEFAULT_OUTPUT_LIMIT = 16_000  # chars per command output
REWARD_FILE = "/home/user/logs/verifier/reward.txt"
FINAL_ANSWER_FILE = "/home/user/logs/agent/final_answer.txt"
DONE_MARKER = "/home/user/logs/agent/.done"

_config: dict[str, Any] = {}
_done = False


def load_config() -> dict[str, Any]:
    """Load config from the JSON file."""
    path = os.environ.get("SWE_MCP_CONFIG", DEFAULT_CONFIG_PATH)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def execute_command(
    command: str,
    workspace: str,
    timeout: int,
    output_limit: int,
) -> dict[str, Any]:
    """Run a shell command and return structured result."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-output_limit:] if proc.stdout else "",
            "stderr": proc.stderr[-output_limit:] if proc.stderr else "",
            "duration_s": round(time.time() - t0, 3),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "duration_s": round(time.time() - t0, 3),
            "timed_out": True,
        }
    except Exception as exc:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.time() - t0, 3),
            "timed_out": False,
        }


def run_verify(
    verify_commands: list[str],
    workspace: str,
    timeout: int,
    output_limit: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Run verify commands and compute reward.

    Returns (reward, results_list).
    Reward = passed / total.  An explicit reward.txt overrides.
    """
    results = []
    passed = 0
    for cmd in verify_commands:
        r = execute_command(cmd, workspace, timeout, output_limit)
        results.append({"cmd": cmd, **r})
        if r["exit_code"] == 0:
            passed += 1

    # Check for explicit reward override
    reward = passed / max(len(verify_commands), 1)
    if os.path.exists(REWARD_FILE):
        try:
            with open(REWARD_FILE) as f:
                reward = float(f.read().strip())
        except (ValueError, OSError):
            pass

    return reward, results


def handle_terminal(arguments: dict[str, Any]) -> dict[str, Any]:
    """Core terminal tool logic."""
    global _done

    command = arguments.get("command")
    final_answer = arguments.get("final_answer")

    if _done:
        return {
            "error": "Session is already complete. No further commands accepted.",
            "done": True,
        }

    if command and final_answer:
        return {
            "error": "Provide exactly one of 'command' or 'final_answer', not both.",
            "done": False,
        }

    if not command and not final_answer:
        return {
            "error": "Provide either 'command' or 'final_answer'.",
            "done": False,
        }

    workspace = _config.get("workspace", DEFAULT_WORKSPACE)
    timeout = _config.get("timeout_per_command_s", DEFAULT_COMMAND_TIMEOUT)
    output_limit = _config.get("output_limit", DEFAULT_OUTPUT_LIMIT)

    if command:
        result = execute_command(command, workspace, timeout, output_limit)
        return {
            "output": result["stdout"],
            "stderr": result["stderr"],
            "exit_code": result["exit_code"],
            "timed_out": result["timed_out"],
            "done": False,
        }

    # final_answer path
    _done = True

    # Write final answer
    os.makedirs(os.path.dirname(FINAL_ANSWER_FILE), exist_ok=True)
    with open(FINAL_ANSWER_FILE, "w") as f:
        f.write(final_answer)

    # Write done marker
    os.makedirs(os.path.dirname(DONE_MARKER), exist_ok=True)
    with open(DONE_MARKER, "w") as f:
        f.write("1")

    # Run verify commands
    verify_commands = _config.get("verify_commands", [])
    if verify_commands:
        reward, verify_results = run_verify(
            verify_commands, workspace, timeout, output_limit
        )
        # Write reward
        os.makedirs(os.path.dirname(REWARD_FILE), exist_ok=True)
        with open(REWARD_FILE, "w") as f:
            f.write(str(reward))

        return {
            "message": "Submission received. Verification complete.",
            "reward": reward,
            "verify_results": verify_results,
            "done": True,
        }

    return {
        "message": "Submission received. No verify commands configured.",
        "reward": None,
        "done": True,
    }


# ── MCP JSON-RPC ──────────────────────────────────────────────────────────

TERMINAL_TOOL_SCHEMA = {
    "name": "terminal",
    "description": (
        "Execute a shell command in the repository workspace, or submit "
        "your final answer.\n\n"
        "Use terminal(command='...') to run any shell command (git, "
        "python, cat, grep, etc.).\n\n"
        "When you are confident the issue is fixed, call "
        "terminal(final_answer='...') to submit. This runs the "
        "verification tests and reports the result."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute in the workspace.",
            },
            "final_answer": {
                "type": "string",
                "description": (
                    "Submit your final answer. Explain what you changed "
                    "and why. This triggers verification."
                ),
            },
        },
    },
}


def handle_jsonrpc(request: dict[str, Any]) -> dict[str, Any]:
    """Handle one MCP JSON-RPC request."""
    request_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {}) or {}

    if method == "initialize":
        return _success(
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "swe-terminal", "version": "1.0.0"},
            },
            request_id,
        )

    if method == "notifications/initialized":
        # Client acknowledgement, no response needed for notifications
        return _success({}, request_id)

    if method == "tools/list":
        return _success({"tools": [TERMINAL_TOOL_SCHEMA]}, request_id)

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name != "terminal":
            return _error(-32601, f"Unknown tool: {tool_name}", request_id)

        result = handle_terminal(arguments)
        return _success(
            {
                "content": [
                    {"type": "text", "text": json.dumps(result, indent=2)},
                ],
                "isError": "error" in result,
            },
            request_id,
        )

    if method == "ping":
        return _success({}, request_id)

    return _error(-32601, f"Method not found: {method}", request_id)


def _success(result: Any, request_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(code: int, message: str, request_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


# ── HTTP Server ────────────────────────────────────────────────────────────


class MCPHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for MCP streamable HTTP transport."""

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(
                400,
                _error(-32700, "Parse error", None),
            )
            return

        response = handle_jsonrpc(request)
        self._send_json(200, response)

    def do_GET(self):
        """Health check endpoint."""
        if self.path in ("/health", "/"):
            self._send_json(200, {"status": "ok", "tool": "terminal"})
        else:
            self.send_error(404)

    def _send_json(self, status: int, data: Any) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        """Suppress noisy per-request logs; write to stderr only."""
        sys.stderr.write(f"[swe-mcp] {fmt % args}\n")


# ── Stdio transport ────────────────────────────────────────────────────────


def run_stdio() -> None:
    """Run as an MCP stdio server (read JSON-RPC from stdin, write to stdout).

    This is the preferred transport for agent MCP discovery (.mcp.json):
    the agent launches this script as a subprocess and communicates over
    stdin/stdout.  Logs go to stderr.
    """
    sys.stderr.write("[swe-mcp] terminal tool server (stdio mode)\n")
    sys.stderr.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response = _error(-32700, "Parse error", None)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue

        # Notifications (no id) don't need a response.
        if request.get("id") is None and request.get("method", "").startswith(
            "notifications/"
        ):
            continue

        response = handle_jsonrpc(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


def run_http() -> None:
    """Run as an HTTP server (MCP streamable HTTP transport)."""
    port = _config.get("port", DEFAULT_PORT)
    port_env = os.environ.get("SWE_MCP_PORT")
    if port_env:
        port = int(port_env)

    server = HTTPServer(("127.0.0.1", port), MCPHandler)
    sys.stderr.write(f"[swe-mcp] terminal tool server on http://127.0.0.1:{port}\n")
    sys.stderr.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    """Entry point: load config and start in stdio or HTTP mode."""
    global _config
    _config = load_config()

    # Ensure log directories exist (best-effort; may fail outside sandbox).
    for d in [
        "/home/user/logs/verifier",
        "/home/user/logs/agent",
    ]:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass

    if "--stdio" in sys.argv:
        run_stdio()
    else:
        run_http()


if __name__ == "__main__":
    main()
