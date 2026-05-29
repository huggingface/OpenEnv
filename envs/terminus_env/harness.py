# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Harness-oriented Terminus session adapter."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable

from openenv.core.env_server.mcp_types import CallToolAction, Tool
from openenv.core.harness import (
    ResourceSessionFactory,
    StepEnvSessionAdapter,
    ToolResult,
    VerifyResult,
)

from .client import TerminusEnv

_TERMINUS_TOOLS: list[Tool] = [
    Tool(
        name="terminal",
        description=(
            "Run a shell command in the Terminus sandbox, or submit final_answer "
            "to trigger verification."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run in the sandbox.",
                },
                "final_answer": {
                    "type": "string",
                    "description": "Final answer to submit when the task is complete.",
                },
            },
            "additionalProperties": False,
        },
    )
]


def _task_field(task: Any, *names: str, default: Any = None) -> Any:
    if not isinstance(task, dict):
        return default
    for name in names:
        value = task.get(name)
        if value is not None:
            return value
    return default


def _coerce_commands(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def _format_initial_prompt(result: Any, task: Any) -> str:
    if isinstance(task, str):
        instruction = task
        setup_commands: list[str] = []
        verify_commands: list[str] = []
    elif isinstance(task, list):
        user_messages = [
            item.get("content")
            for item in task
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        instruction = str(user_messages[-1] if user_messages else task)
        setup_commands = []
        verify_commands = []
    elif isinstance(task, dict):
        instruction = str(
            _task_field(task, "instruction", "prompt", "question", "task", default="")
        )
        setup_commands = _coerce_commands(_task_field(task, "setup", "setup_scripts"))
        verify_commands = _coerce_commands(_task_field(task, "verify", "verify_scripts"))
    else:
        instruction = str(task or "")
        setup_commands = []
        verify_commands = []

    metadata = getattr(result.observation, "metadata", {}) or {}
    verify_commands = _coerce_commands(
        metadata.get("verify_commands") or verify_commands
    )

    parts = []
    if instruction:
        parts.append(f"Task:\n{instruction}")
    else:
        parts.append("Task:\nUse the terminal tool to solve the current task.")

    reset_message = metadata.get("message")
    if reset_message:
        parts.append(f"Environment:\n{reset_message}")

    if setup_commands:
        parts.append(
            "Setup commands have already run:\n"
            + "\n".join(f"- {command}" for command in setup_commands)
        )
    if verify_commands:
        parts.append(
            "Verification commands will run after final_answer:\n"
            + "\n".join(f"- {command}" for command in verify_commands)
        )

    parts.append(
        'Use {"command": "..."} to inspect and modify the sandbox. '
        'When finished, use {"final_answer": "..."} exactly once so '
        "verification runs and emits the environment reward."
    )
    return "\n\n".join(parts)


def _extract_tool_output(observation: Any) -> Any:
    result = getattr(observation, "result", None)
    if result is None:
        return None
    if hasattr(result, "data"):
        return result.data
    if isinstance(result, dict):
        if "data" in result:
            return result["data"]
        content = result.get("content")
        if isinstance(content, list):
            texts = [
                str(item.get("text"))
                for item in content
                if isinstance(item, dict) and item.get("text") is not None
            ]
            if texts:
                return "\n".join(texts)
        return result
    content = getattr(result, "content", None)
    if isinstance(content, list):
        texts = [
            getattr(item, "text", None)
            for item in content
            if getattr(item, "text", None) is not None
        ]
        if texts:
            return "\n".join(texts)
    return result


def _tool_error_message(observation: Any) -> str | None:
    error = getattr(observation, "error", None)
    if error is None:
        return None
    message = getattr(error, "message", None)
    if message is not None:
        return str(message)
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error)


def _state_to_data(state: Any) -> Any:
    if state is None:
        return None
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return state


def _build_tool_result(
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    state: Any,
) -> ToolResult:
    output = _extract_tool_output(result.observation)
    error = _tool_error_message(result.observation)
    data = {
        "tool_name": tool_name,
        "arguments": dict(arguments),
        "output": output,
        "reward": result.reward,
        "done": result.done,
    }
    if error:
        data["error"] = error

    return ToolResult(
        data=data,
        done=bool(result.done),
        error=error,
        metadata={
            "reward": result.reward,
            "state": _state_to_data(state),
        },
    )


def _build_verify(
    transcript: list[dict[str, Any]],
    final_state: Any | None,
    last_result: Any | None,
    state: Any,
) -> VerifyResult:
    reward = None if last_result is None else last_result.reward
    done = False if last_result is None else bool(last_result.done)
    state_data = _state_to_data(state)
    metrics = {
        "done": done,
        "step_count": getattr(state, "step_count", 0),
        "commands": len(getattr(state, "commands", []) or []),
        "verify_commands": len(getattr(state, "verify_commands", []) or []),
        "setup_commands": len(getattr(state, "setup_results", []) or []),
        "submitted_answer": getattr(state, "submitted_answer", None) is not None,
        "sandbox_id": getattr(state, "sandbox_id", None),
    }
    if state is None and last_result is not None:
        metrics["step_count"] = len(transcript)
    return VerifyResult(
        env_reward=reward,
        done=done,
        metrics=metrics,
        artifacts={
            "final_state": state_data,
            "final_rollout": final_state,
            "transcript_length": len(transcript),
        },
    )


def _build_reset_kwargs(
    task: Any,
    default_setup: list[str],
    default_verify: list[str],
    default_sandbox: dict[str, Any],
) -> dict[str, Any]:
    reset_kwargs: dict[str, Any] = dict(default_sandbox)
    setup = list(default_setup)
    verify = list(default_verify)
    if isinstance(task, dict):
        setup = _coerce_commands(_task_field(task, "setup", "setup_scripts", default=setup))
        verify = _coerce_commands(
            _task_field(task, "verify", "verify_scripts", default=verify)
        )
        for key in (
            "sandbox_image",
            "sandbox_flavor",
            "sandbox_timeout",
            "hf_sandbox_image",
            "hf_sandbox_flavor",
            "hf_sandbox_timeout",
            "forward_hf_token",
            "sandbox_backend",
            "sandbox_root",
        ):
            if key in task:
                reset_kwargs[key] = task[key]

    if setup:
        reset_kwargs["setup"] = setup
    if verify:
        reset_kwargs["verify"] = verify
    return reset_kwargs


class TerminusSessionFactory(ResourceSessionFactory):
    """Create Terminus-backed resource sessions for harness rollouts."""

    def __init__(
        self,
        client_factory: Callable[[], TerminusEnv],
        *,
        default_setup: list[str] | None = None,
        default_verify: list[str] | None = None,
        sandbox: dict[str, Any] | None = None,
    ):
        self._client_factory = client_factory
        self._default_setup = list(default_setup or [])
        self._default_verify = list(default_verify or [])
        self._sandbox = dict(sandbox or {})

    def create(
        self,
        task: Any = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> StepEnvSessionAdapter:
        reset_kwargs = _build_reset_kwargs(
            task,
            self._default_setup,
            self._default_verify,
            self._sandbox,
        )

        return StepEnvSessionAdapter(
            client=self._client_factory(),
            task=task,
            seed=seed,
            episode_id=episode_id,
            tool_specs=list(_TERMINUS_TOOLS),
            action_builder=lambda name, arguments: CallToolAction(
                tool_name=name,
                arguments=dict(arguments),
            ),
            initial_messages_builder=lambda result, current_task: [
                {
                    "role": "user",
                    "content": _format_initial_prompt(result, current_task),
                }
            ],
            tool_result_builder=_build_tool_result,
            verify_builder=_build_verify,
            reset_kwargs=reset_kwargs,
        )


_TERMINAL_CALL_RE = re.compile(r"terminal\s*\((?P<body>.*?)\)", re.DOTALL)


def build_terminal_tool_call(response_text: str, *, call_id: str = "terminal-0"):
    """Parse a terminal call from model text.

    The preferred format is one JSON object containing ``command`` or
    ``final_answer``. The parser also accepts Pi-style ``terminal(...)`` text
    because small policy models often imitate that syntax before they learn
    structured tool calls. Invalid text falls back to a shell command so the
    environment, not this parser, decides whether a rollout earns reward.
    """

    from openenv.core.llm_client import ToolCall

    text = _strip_code_fence(response_text.strip())
    payload = _parse_terminal_json(text)
    if payload is None:
        payload = _parse_terminal_expression(text)
    if payload is None:
        payload = {"command": response_text}

    arguments = {
        key: str(payload[key])
        for key in ("command", "final_answer")
        if payload.get(key) is not None
    }
    if not arguments:
        arguments = {"command": ""}
    if arguments.get("command") and arguments.get("final_answer"):
        arguments = {"command": arguments["command"]}
    return ToolCall(id=call_id, name="terminal", args=arguments)


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    stripped = text.strip("`").strip()
    if stripped.startswith("json"):
        return stripped[4:].strip()
    return stripped


def _parse_terminal_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        normalized = _normalize_terminal_payload(payload)
        if normalized is not None:
            return normalized
    return None


def _normalize_terminal_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if "arguments" in payload:
        arguments = payload["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        return _normalize_terminal_payload(arguments)
    if any(payload.get(key) is not None for key in ("command", "final_answer")):
        return payload
    return None


def _parse_terminal_expression(text: str) -> dict[str, Any] | None:
    match = _TERMINAL_CALL_RE.search(text)
    if not match:
        return None
    try:
        expression = ast.parse(f"terminal({match.group('body')})", mode="eval")
    except SyntaxError:
        return None
    if not isinstance(expression.body, ast.Call):
        return None
    payload: dict[str, Any] = {}
    for keyword in expression.body.keywords:
        if keyword.arg not in {"command", "final_answer"}:
            continue
        value = keyword.value
        try:
            payload[keyword.arg] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            if isinstance(value, ast.Name):
                payload[keyword.arg] = value.id
    return payload or None


__all__ = [
    "TerminusSessionFactory",
    "build_terminal_tool_call",
]
