"""Harness-oriented BrowserGym session adapters."""

from __future__ import annotations

import re
from typing import Any

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    ResourceSessionFactory,
    StepEnvSessionAdapter,
    ToolResult,
    VerifyResult,
)
from openenv.core.llm_client import ToolCall

from .models import BrowserGymAction

_BROWSERGYM_TOOLS = [
    Tool(
        name="click",
        description="Click an element by BrowserGym bid.",
        input_schema={
            "type": "object",
            "properties": {
                "bid": {"type": "string"},
            },
            "required": ["bid"],
        },
    ),
    Tool(
        name="fill",
        description="Fill an input field by bid.",
        input_schema={
            "type": "object",
            "properties": {
                "bid": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["bid", "text"],
        },
    ),
    Tool(
        name="send_keys",
        description="Send keyboard input to the page.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="scroll",
        description="Scroll the page up or down.",
        input_schema={
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down"]},
            },
            "required": ["direction"],
        },
    ),
    Tool(
        name="noop",
        description="Take no action on the current page.",
        input_schema={"type": "object", "properties": {}},
    ),
]


def _quote(value: str) -> str:
    return repr(value)


def build_browsergym_action_str(tool_name: str, arguments: dict[str, Any]) -> str:
    """Convert a BrowserGym tool call into the action string the env expects."""

    if tool_name == "click":
        return f"click({_quote(str(arguments['bid']))})"
    if tool_name == "fill":
        return (
            f"fill({_quote(str(arguments['bid']))}, {_quote(str(arguments['text']))})"
        )
    if tool_name == "send_keys":
        return f"send_keys({_quote(str(arguments['text']))})"
    if tool_name == "scroll":
        return f"scroll({_quote(str(arguments['direction']))})"
    if tool_name == "noop":
        return "noop()"

    raise KeyError(f"Unsupported BrowserGym tool: {tool_name}")


def _format_browsergym_prompt(observation: Any, task: Any) -> str:
    goal = getattr(observation, "goal", "") or (task or "")
    page_text = getattr(observation, "axtree_txt", "") or getattr(
        observation, "text", ""
    )
    error = getattr(observation, "error", "")

    parts = []
    if goal:
        parts.append(f"Goal: {goal}")
    if error:
        parts.append(f"Previous action error: {error}")
    if page_text:
        parts.append(f"Page structure:\n{page_text}")
    parts.append("Choose the next browser action.")
    return "\n\n".join(parts)


def _build_browsergym_tool_result(
    task: Any,
):
    def builder(
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        state: Any,
    ) -> ToolResult:
        observation = result.observation
        data = {
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "goal": getattr(observation, "goal", "") or (task or ""),
            "observation_text": getattr(observation, "axtree_txt", "")
            or getattr(observation, "text", ""),
            "url": getattr(observation, "url", ""),
            "error": getattr(observation, "error", ""),
            "last_action_error": getattr(observation, "last_action_error", False),
            "reward": result.reward,
            "done": result.done,
        }
        metadata = {
            "reward": result.reward,
            "state": state.model_dump() if hasattr(state, "model_dump") else state,
        }
        return ToolResult(data=data, done=bool(result.done), metadata=metadata)

    return builder


def _build_browsergym_verify(
    transcript: list[dict[str, Any]],
    final_state: Any | None,
    last_result: Any | None,
    state: Any,
) -> VerifyResult:
    reward = None if last_result is None else last_result.reward
    done = False if last_result is None else bool(last_result.done)
    metrics = {
        "step_count": getattr(state, "step_count", 0),
        "cum_reward": getattr(state, "cum_reward", reward or 0.0),
        "benchmark": getattr(state, "benchmark", ""),
        "task_name": getattr(state, "task_name", ""),
    }
    artifacts = {
        "final_state": state.model_dump() if hasattr(state, "model_dump") else state,
        "final_rollout": final_state,
        "transcript_length": len(transcript),
    }
    return VerifyResult(
        env_reward=reward,
        done=done,
        metrics=metrics,
        artifacts=artifacts,
    )


class BrowserGymSessionFactory(ResourceSessionFactory):
    """Create BrowserGym-backed resource sessions from client factories."""

    def __init__(self, client_factory, *, default_task: str | None = None):
        self._client_factory = client_factory
        self._default_task = default_task

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> StepEnvSessionAdapter:
        session_task = task if task is not None else self._default_task
        client = self._client_factory()

        reset_kwargs = {}
        if session_task is not None:
            reset_kwargs["task_name"] = session_task

        return StepEnvSessionAdapter(
            client=client,
            task=session_task,
            seed=seed,
            episode_id=episode_id,
            tool_specs=list(_BROWSERGYM_TOOLS),
            action_builder=lambda name, arguments: BrowserGymAction(
                action_str=build_browsergym_action_str(name, arguments)
            ),
            initial_messages_builder=lambda result, current_task: [
                {
                    "role": "user",
                    "content": _format_browsergym_prompt(
                        result.observation,
                        current_task,
                    ),
                }
            ],
            tool_result_builder=_build_browsergym_tool_result(session_task),
            verify_builder=_build_browsergym_verify,
            reset_kwargs=reset_kwargs,
        )


_CLICK_RE = re.compile(r"^click\(\s*(?P<q>['\"])(?P<bid>.+?)(?P=q)\s*\)$")
_FILL_RE = re.compile(
    r"^fill\(\s*(?P<bq>['\"])(?P<bid>.+?)(?P=bq)\s*,\s*"
    r"(?P<tq>['\"])(?P<text>.*?)(?P=tq)\s*\)$"
)
_SEND_KEYS_RE = re.compile(r"^send_keys\(\s*(?P<q>['\"])(?P<text>.*?)(?P=q)\s*\)$")
_SCROLL_RE = re.compile(r"^scroll\(\s*['\"]?(?P<direction>up|down)['\"]?\s*\)$")
_NOOP_RE = re.compile(r"^noop\(\s*\)$")


def build_browsergym_action_tool_call(action_text: str) -> ToolCall:
    """Parse a text BrowserGym action into a structured tool call."""

    text = action_text.strip()

    if match := _CLICK_RE.match(text):
        return ToolCall(
            id="browsergym-click",
            name="click",
            args={"bid": match.group("bid")},
        )
    if match := _FILL_RE.match(text):
        return ToolCall(
            id="browsergym-fill",
            name="fill",
            args={
                "bid": match.group("bid"),
                "text": match.group("text"),
            },
        )
    if match := _SEND_KEYS_RE.match(text):
        return ToolCall(
            id="browsergym-send_keys",
            name="send_keys",
            args={"text": match.group("text")},
        )
    if match := _SCROLL_RE.match(text):
        return ToolCall(
            id="browsergym-scroll",
            name="scroll",
            args={"direction": match.group("direction")},
        )
    if _NOOP_RE.match(text):
        return ToolCall(id="browsergym-noop", name="noop", args={})

    raise ValueError(f"Unsupported BrowserGym action: {action_text}")


__all__ = [
    "BrowserGymSessionFactory",
    "build_browsergym_action_str",
    "build_browsergym_action_tool_call",
]
