# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental harness helpers for training and evaluation.

These helpers live outside the stable ``openenv.core`` package surface while
RFC 005 is still under review. Import them from ``openenv.core.harness``.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..client_types import StepResult
from ..env_server.mcp_types import JsonRpcErrorCode, JsonRpcResponse, Tool
from ..env_server.types import State
from ..llm_client import LLMResponse

Message = dict[str, Any]
RESERVED_TOOL_NAMES = frozenset({"reset", "step", "state", "close"})


@dataclass
class ToolResult:
    """Normalized result from a resource session tool invocation."""

    data: Any = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class VerifyResult:
    """Final rollout data produced after a rollout completes.

    ``env_reward`` must forward reward already produced inside the environment.
    It must not synthesize a new reward in the orchestration layer.
    """

    env_reward: float | None = None
    done: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutEvent:
    """Event emitted while a harness drives a rollout."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolTraceEntry:
    """Record of a tool call issued by the harness."""

    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult


@dataclass
class ModelStepResult:
    """Structured output from one white-box model sampling step."""

    response: LLMResponse
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


@dataclass
class HarnessRolloutResult:
    """Complete result of a harness-driven rollout."""

    messages: list[Message] = field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = field(default_factory=list)
    events: list[RolloutEvent] = field(default_factory=list)
    done: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


@dataclass
class HarnessRunLimits:
    """Execution limits for harness-driven rollouts."""

    max_turns: int = 10
    max_tool_calls_per_turn: int | None = None
    max_total_tool_calls: int | None = None
    sampling: dict[str, Any] = field(default_factory=dict)


class ModelStep(Protocol):
    """Callable used by white-box harnesses to sample the next model turn."""

    def __call__(
        self,
        messages: list[Message],
        tools: list[Tool],
        sampling: dict[str, Any],
    ) -> ModelStepResult: ...


class ResourceSession(ABC):
    """Per-rollout environment/resource session exposed to harnesses."""

    @abstractmethod
    def initial_messages(self) -> list[Message]:
        """Return initial prompt messages for the harness."""

    @abstractmethod
    def list_tools(self) -> list[Tool]:
        """Return the MCP-style tool manifest exposed by the session."""

    @abstractmethod
    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke a session tool."""

    @abstractmethod
    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        """Finalize a rollout after the harness has stopped.

        This hook may add metrics or artifacts and may forward the final reward
        already produced by the environment.
        """

    @abstractmethod
    def close(self) -> None:
        """Release session resources."""


class ResourceSessionFactory(ABC):
    """Factory for producing isolated per-rollout sessions."""

    @abstractmethod
    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> ResourceSession:
        """Create one isolated resource session for a rollout."""


class HarnessAdapter(ABC):
    """Interface implemented by harness drivers."""

    @abstractmethod
    def run_white_box(
        self,
        model_step: ModelStep,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        """Run a rollout while the trainer owns model sampling."""

    @abstractmethod
    def run_black_box(
        self,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        """Run a rollout with an opaque harness (evaluation-only path)."""


def _serialize_for_message(value: Any) -> str:
    """Convert structured tool data to a stable text payload."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _state_to_data(state: Any) -> Any:
    """Convert state objects to plain data for metrics and artifacts."""

    if state is None:
        return None
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return state


def _tool_result_reward(tool_result: ToolResult) -> float | None:
    """Extract the environment reward already emitted by a tool result."""

    reward = tool_result.metadata.get("reward")
    if reward is None and isinstance(tool_result.data, dict):
        reward = tool_result.data.get("reward")
    if reward is None:
        return None
    return float(reward)


def _resolve_env_reward(
    rollout: HarnessRolloutResult,
    verify: VerifyResult,
) -> float:
    """Resolve the final environment reward without allowing external synthesis."""

    trace_reward: float | None = None
    for entry in reversed(rollout.tool_trace):
        trace_reward = _tool_result_reward(entry.result)
        if trace_reward is not None:
            break

    verify_reward = None if verify.env_reward is None else float(verify.env_reward)
    if (
        trace_reward is not None
        and verify_reward is not None
        and not math.isclose(
            verify_reward,
            trace_reward,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    ):
        raise ValueError(
            "verify.env_reward must forward the environment reward from the rollout"
        )

    if trace_reward is not None:
        return trace_reward
    if verify_reward is not None:
        return verify_reward
    raise ValueError("rollout did not produce an environment reward")


class StepEnvSessionAdapter(ResourceSession):
    """Expose an existing step/reset/state client as a resource session."""

    def __init__(
        self,
        client: Any,
        *,
        task: Any = None,
        seed: int | None = None,
        episode_id: str | None = None,
        tool_specs: list[Tool],
        action_builder: Callable[[str, dict[str, Any]], Any],
        initial_messages_builder: Callable[[StepResult[Any], Any], list[Message]],
        tool_result_builder: Callable[
            [str, dict[str, Any], StepResult[Any], Any],
            ToolResult,
        ]
        | None = None,
        verify_builder: Callable[
            [list[Message], Any | None, StepResult[Any] | None, Any],
            VerifyResult,
        ]
        | None = None,
        reset_kwargs: dict[str, Any] | None = None,
    ):
        if hasattr(client, "sync") and callable(client.sync):
            self._client = client.sync()
        else:
            self._client = client

        self._task = task
        self._tool_specs = list(tool_specs)
        self._action_builder = action_builder
        self._initial_messages_builder = initial_messages_builder
        self._tool_result_builder = tool_result_builder or self._default_tool_result
        self._verify_builder = verify_builder or self._default_verify
        self._closed = False
        self._tools_by_name = {tool.name: tool for tool in self._tool_specs}
        reserved = sorted(set(self._tools_by_name) & RESERVED_TOOL_NAMES)
        if reserved:
            raise ValueError(
                "Tool names are reserved for orchestration controls: "
                + ", ".join(reserved)
            )

        reset_payload = dict(reset_kwargs or {})
        if seed is not None:
            reset_payload.setdefault("seed", seed)
        if episode_id is not None:
            reset_payload.setdefault("episode_id", episode_id)

        self._last_result: StepResult[Any] | None = None
        self._last_state = None
        try:
            self._initial_result: StepResult[Any] = self._client.reset(**reset_payload)
            self._last_state = self._read_state()
        except Exception:
            self.close()
            raise

    def _read_state(self) -> Any:
        if hasattr(self._client, "state") and callable(self._client.state):
            return self._client.state()
        return None

    def _default_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: StepResult[Any],
        state: Any,
    ) -> ToolResult:
        return ToolResult(
            data={
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "observation": result.observation,
                "reward": result.reward,
                "done": result.done,
            },
            done=bool(result.done),
            metadata={
                "reward": result.reward,
                "state": _state_to_data(state),
            },
        )

    def _default_verify(
        self,
        transcript: list[Message],
        final_state: Any | None,
        last_result: StepResult[Any] | None,
        state: Any,
    ) -> VerifyResult:
        reward = None if last_result is None else last_result.reward
        done = False if last_result is None else bool(last_result.done)
        metrics = {}
        if isinstance(state, State):
            metrics["step_count"] = state.step_count
        elif isinstance(state, dict) and "step_count" in state:
            metrics["step_count"] = state["step_count"]

        return VerifyResult(
            env_reward=reward,
            done=done,
            metrics=metrics,
            artifacts={
                "final_state": _state_to_data(state),
                "transcript_length": len(transcript),
            },
        )

    def initial_messages(self) -> list[Message]:
        return list(self._initial_messages_builder(self._initial_result, self._task))

    def list_tools(self) -> list[Tool]:
        return list(self._tool_specs)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name in RESERVED_TOOL_NAMES:
            raise KeyError(f"Reserved orchestration tool: {name}")
        if name not in self._tools_by_name:
            raise KeyError(f"Unknown tool: {name}")

        action = self._action_builder(name, dict(arguments))
        result = self._client.step(action)
        state = self._read_state()
        self._last_result = result
        self._last_state = state
        return self._tool_result_builder(name, dict(arguments), result, state)

    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        return self._verify_builder(
            list(transcript),
            final_state,
            self._last_result,
            self._last_state,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if hasattr(self._client, "close") and callable(self._client.close):
            self._client.close()


class SessionMCPBridge:
    """Expose a resource session through an in-process MCP JSON-RPC bridge."""

    def __init__(self, session: ResourceSession):
        self.session = session

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {}) or {}

        if request.get("jsonrpc") != "2.0":
            return JsonRpcResponse.error_response(
                JsonRpcErrorCode.INVALID_REQUEST,
                message="Invalid Request",
                request_id=request_id,
            ).model_dump()

        if method == "tools/list":
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in self.session.list_tools()
            ]
            return JsonRpcResponse.success(
                {"tools": tools},
                request_id=request_id,
            ).model_dump()

        if method == "tools/call":
            if "name" not in params:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    message="Missing tool name",
                    request_id=request_id,
                ).model_dump()

            tool_name = params["name"]
            if tool_name in RESERVED_TOOL_NAMES:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    message=f"Reserved orchestration tool: {tool_name}",
                    request_id=request_id,
                ).model_dump()

            try:
                result = self.session.call_tool(
                    tool_name,
                    dict(params.get("arguments", {})),
                )
            except KeyError as exc:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.METHOD_NOT_FOUND,
                    message=str(exc.args[0]) if exc.args else "Method not found",
                    request_id=request_id,
                ).model_dump()
            except ValueError as exc:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    message=str(exc),
                    request_id=request_id,
                ).model_dump()
            except Exception as exc:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INTERNAL_ERROR,
                    message=str(exc),
                    request_id=request_id,
                ).model_dump()
            return JsonRpcResponse.success(
                {
                    "data": result.data,
                    "done": result.done,
                    "metadata": dict(result.metadata),
                    "error": result.error,
                },
                request_id=request_id,
            ).model_dump()

        return JsonRpcResponse.error_response(
            JsonRpcErrorCode.METHOD_NOT_FOUND,
            message="Method not found",
            request_id=request_id,
        ).model_dump()


class MCPHarnessAdapter(HarnessAdapter):
    """White-box harness that follows an MCP tool-calling loop."""

    def run_white_box(
        self,
        model_step: ModelStep,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        run_limits = limits or HarnessRunLimits()
        messages = list(session.initial_messages())
        tools = session.list_tools()
        result = HarnessRolloutResult(messages=list(messages))
        total_tool_calls = 0

        for turn_index in range(run_limits.max_turns):
            step_result = model_step(messages, tools, dict(run_limits.sampling))
            assistant_message = step_result.response.to_message_dict()
            messages.append(assistant_message)

            result.prompt_ids.extend(step_result.prompt_ids)
            result.completion_ids.extend(step_result.completion_ids)
            result.logprobs.extend(step_result.logprobs)
            result.events.append(
                RolloutEvent(
                    type="model_response",
                    payload={
                        "turn": turn_index,
                        "content": step_result.response.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": dict(tool_call.args),
                            }
                            for tool_call in step_result.response.tool_calls
                        ],
                    },
                )
            )

            if not step_result.response.tool_calls:
                result.done = True
                break

            tool_calls = step_result.response.tool_calls
            if run_limits.max_tool_calls_per_turn is not None:
                tool_calls = tool_calls[: run_limits.max_tool_calls_per_turn]

            for tool_call in tool_calls:
                if (
                    run_limits.max_total_tool_calls is not None
                    and total_tool_calls >= run_limits.max_total_tool_calls
                ):
                    result.metrics["truncated"] = True
                    result.messages = list(messages)
                    return result

                tool_result = session.call_tool(tool_call.name, dict(tool_call.args))
                total_tool_calls += 1

                trace_entry = ToolTraceEntry(
                    tool_name=tool_call.name,
                    arguments=dict(tool_call.args),
                    result=tool_result,
                )
                result.tool_trace.append(trace_entry)
                result.events.append(
                    RolloutEvent(
                        type="tool_call",
                        payload={
                            "tool_name": tool_call.name,
                            "arguments": dict(tool_call.args),
                            "done": tool_result.done,
                        },
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": _serialize_for_message(tool_result.data),
                    }
                )

                if tool_result.done:
                    result.done = True
                    break

            if result.done:
                break

        result.messages = list(messages)
        result.metrics.setdefault(
            "turns",
            len([event for event in result.events if event.type == "model_response"]),
        )
        result.metrics.setdefault("tool_calls", len(result.tool_trace))
        return result

    def run_black_box(
        self,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        raise NotImplementedError(
            "MCPHarnessAdapter is a white-box harness. "
            "Use CLIHarnessAdapter for opaque evaluation harnesses."
        )


class CLIHarnessAdapter(HarnessAdapter):
    """Thin black-box adapter for opaque CLI-style harnesses."""

    def __init__(
        self,
        runner: Callable[
            [SessionMCPBridge, ResourceSession, HarnessRunLimits],
            HarnessRolloutResult,
        ],
    ):
        self._runner = runner

    def run_white_box(
        self,
        model_step: ModelStep,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        raise NotImplementedError(
            "CLIHarnessAdapter only supports black-box evaluation rollouts."
        )

    def run_black_box(
        self,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        run_limits = limits or HarnessRunLimits()
        bridge = SessionMCPBridge(session)
        return self._runner(bridge, session, run_limits)


def build_harness_rollout_func(
    *,
    session_factory: Any,
    harness_adapter: HarnessAdapter,
    model_step_builder: Callable[[Any, ResourceSession], ModelStep],
    limits: HarnessRunLimits | None = None,
    reward_key: str = "env_reward",
) -> Callable[[list[Any], Any], dict[str, list[Any]]]:
    """Build a TRL-compatible rollout function from sessions and harnesses."""

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, list[Any]]:
        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        rewards: list[float] = []
        verify_metrics: list[dict[str, Any]] = []

        for prompt in prompts:
            session = session_factory.create(task=prompt)
            try:
                model_step = model_step_builder(trainer, session)
                rollout = harness_adapter.run_white_box(
                    model_step=model_step,
                    session=session,
                    limits=limits,
                )
                verify = session.verify(
                    transcript=rollout.messages,
                    final_state={
                        "done": rollout.done,
                        "metrics": dict(rollout.metrics),
                        "events": [
                            {
                                "type": event.type,
                                "payload": dict(event.payload),
                            }
                            for event in rollout.events
                        ],
                        "tool_trace": [
                            {
                                "tool_name": entry.tool_name,
                                "arguments": dict(entry.arguments),
                                "result": {
                                    "data": entry.result.data,
                                    "done": entry.result.done,
                                    "metadata": dict(entry.result.metadata),
                                    "error": entry.result.error,
                                },
                            }
                            for entry in rollout.tool_trace
                        ],
                    },
                )

                all_prompt_ids.append(list(rollout.prompt_ids))
                all_completion_ids.append(list(rollout.completion_ids))
                all_logprobs.append(list(rollout.logprobs))
                rewards.append(_resolve_env_reward(rollout, verify))
                verify_metrics.append(dict(verify.metrics))
            finally:
                session.close()

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            reward_key: rewards,
            "verify_metrics": verify_metrics,
        }

    return rollout_func


from .pi_cli import PiCLIHarnessAdapter

__all__ = [
    "CLIHarnessAdapter",
    "HarnessAdapter",
    "HarnessRolloutResult",
    "HarnessRunLimits",
    "MCPHarnessAdapter",
    "Message",
    "ModelStep",
    "ModelStepResult",
    "PiCLIHarnessAdapter",
    "RESERVED_TOOL_NAMES",
    "ResourceSession",
    "ResourceSessionFactory",
    "RolloutEvent",
    "SessionMCPBridge",
    "StepEnvSessionAdapter",
    "ToolResult",
    "ToolTraceEntry",
    "VerifyResult",
    "build_harness_rollout_func",
]
