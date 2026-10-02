# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""τ²-bench as an OpenEnv environment.

Each episode is one τ²-bench task: a customer-service conversation between the
agent and an LLM-simulated user, in a domain with a database, a policy, and
tools. The agent acts through MCP tools:

- the domain's own tools (e.g. `get_user_details`, `book_reservation`), which
  read and write the task's database;
- `respond_to_user(message)`, which sends a message to the simulated user and
  returns their reply;
- `done()`, which ends the conversation from the agent's side.

The episode ends when the user or the agent stops, and τ²-bench's own
evaluator turns the final database state and conversation into the reward.

The simulated user (and, in domains that need one, the evaluator's judge) is an
LLM called through litellm. It defaults to Hugging Face Inference Providers.
"""

import asyncio
import json
import os
import random
import threading
import uuid
from typing import Any, Optional

import tau2.evaluator.evaluator_nl_assertions as nl_assertions
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool import ToolResult
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation
from tau2.gym.gym_agent import AgentGymEnv
from tau2.registry import registry
from tau2.user.user_simulator_base import OUT_OF_SCOPE, STOP, TRANSFER
from tau2.utils.llm_utils import generate
from tau2.utils.utils import DATA_DIR

from ..models import Tau2State

HF_ROUTER = "https://router.huggingface.co/v1"
# What the simulated customer appends to its last message, and why it leaves.
END_TOKENS = (STOP, TRANSFER, OUT_OF_SCOPE)


def without_end_tokens(text: str) -> str:
    """The customer's message without τ²-bench's end-of-conversation tokens."""
    for token in END_TOKENS:
        text = text.replace(token, "")
    return text.strip()


DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_MODELS = {
    "hf": DEFAULT_MODEL,
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-5",
}


def llm_config(
    provider: str, model: Optional[str] = None, hf_token: Optional[str] = None
) -> tuple[str, dict]:
    """
    The litellm model name and arguments for a provider.

    Args:
        provider (`str`):
            `"hf"` (Hugging Face Inference Providers, with `HF_TOKEN`), `"openai"`
            (with `OPENAI_API_KEY`) or `"anthropic"` (with `ANTHROPIC_API_KEY`).
        model (`str`, *optional*):
            Model id. Defaults to the provider's entry in `DEFAULT_MODELS`.
        hf_token (`str`, *optional*):
            Token for `"hf"`. Defaults to `HF_TOKEN`.

    Returns:
        `tuple[str, dict]`: the litellm model name and the call arguments.
    """
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"unknown provider {provider!r}; use one of {list(DEFAULT_MODELS)}"
        )
    model = model or DEFAULT_MODELS[provider]
    if provider == "hf":
        # The router is OpenAI-compatible, so litellm talks to it as an OpenAI endpoint.
        return f"openai/{model}", {
            "api_base": HF_ROUTER,
            "api_key": hf_token or os.environ.get("HF_TOKEN"),
            "temperature": 0.0,
            "num_retries": 5,
        }
    return f"{provider}/{model}", {"temperature": 0.0, "num_retries": 5}


# τ²-bench's retail judge reads its model from module constants, shared by every
# conversation in the process. It runs in the thread that steps the simulation, so
# each conversation sets its own judge right before stepping.
_judge = threading.local()


def _judge_generate(**kwargs):
    return generate(**{**kwargs, "model": _judge.model, **_judge.args})


nl_assertions.generate = _judge_generate


# `respond_to_user` and `done` wait on the simulated customer (and on retail, the judge),
# which are LLM calls with retries, so tool calls get longer than MCP's 30 s default.
TOOL_TIMEOUT_S = 300.0


# What FastMCP gives a function tool returning `str`, so clients get the text as `.data`.
TEXT_OUTPUT = {
    "type": "object",
    "properties": {"result": {"type": "string"}},
    "required": ["result"],
    "x-fastmcp-wrap-result": True,
}


class Tau2Tool(Tool):
    """An MCP tool that forwards its call to the running τ²-bench simulation."""

    env: Any = None
    output_schema: dict[str, Any] = TEXT_OUTPUT

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        if self.name == "respond_to_user":
            action = arguments["message"]
        else:
            action = json.dumps({"name": self.name, "arguments": arguments})
        text = await asyncio.to_thread(self.env.act, action)
        return ToolResult(content=text, structured_content={"result": text})


RESPOND_TO_USER = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "What to say to the user."}
    },
    "required": ["message"],
}


class Tau2Environment(MCPEnvironment):
    """
    One τ²-bench domain as an MCP environment.

    Args:
        domain (`str`, *optional*, defaults to `"airline"`):
            τ²-bench domain: `airline`, `retail`, `telecom`, `telecom-workflow`,
            `banking_knowledge` or `mock`.
        split (`str`, *optional*, defaults to `"test"`):
            Task split, e.g. `train`, `test` or `base` (all tasks).
        user_provider (`str`, *optional*, defaults to `"hf"`):
            Provider for the simulated user and the evaluator's judge. See
            [`llm_config`].
        user_model (`str`, *optional*):
            Model for the simulated user and the judge. Defaults to the
            provider's default.
        max_steps (`int`, *optional*, defaults to `100`):
            τ²-bench's limit on messages per conversation.
        hf_token (`str`, *optional*):
            Token for the `"hf"` provider, e.g. a visitor's. Defaults to `HF_TOKEN`.
    """

    def __init__(
        self,
        domain: str = "airline",
        split: str = "test",
        user_provider: str = "hf",
        user_model: Optional[str] = None,
        max_steps: int = 100,
        hf_token: Optional[str] = None,
    ):
        if not (DATA_DIR / "tau2" / "domains").is_dir():
            raise FileNotFoundError(
                f"τ²-bench data not found in {DATA_DIR}. Point TAU2_DATA_DIR at the `data` "
                "folder of a tau2-bench checkout (see the README)."
            )
        self.domain = domain
        self.split = split
        self.max_steps = max_steps
        self.user_provider = user_provider
        self.user_llm, self.user_llm_args = llm_config(
            user_provider, user_model, hf_token
        )

        self.task_ids = [t.id for t in registry.get_tasks_loader(domain)(split)]

        mcp = FastMCP(f"tau2_{domain}")
        for tool in registry.get_env_constructor(domain)().get_tools():
            schema = tool.openai_schema["function"]
            mcp.add_tool(
                Tau2Tool(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=schema["parameters"],
                    env=self,
                )
            )
        mcp.add_tool(
            Tau2Tool(
                name="respond_to_user",
                description="Send a message to the user and get their reply.",
                parameters=RESPOND_TO_USER,
                env=self,
            )
        )
        mcp.add_tool(
            Tau2Tool(
                name="done",
                description="End the conversation, once the user's request is handled.",
                parameters={"type": "object", "properties": {}},
                env=self,
            )
        )
        super().__init__(mcp)

        self._gym: Optional[AgentGymEnv] = None
        self._state = Tau2State(domain=domain, split=split)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Start a τ²-bench task: the simulated user opens the conversation.

        Args:
            seed (`int`, *optional*):
                Picks the task when `task_id` is not given.
            episode_id (`str`, *optional*):
                Episode identifier. A UUID is generated when `None`.
            task_id (`str`, *optional*):
                The τ²-bench task to run. A random task of the split otherwise.

        Returns:
            `Observation` whose metadata has the user's first message, the
            domain policy the agent must follow, and the task id.
        """
        # Only conversations need the token, so the server and its task explorer run without one.
        if self.user_provider == "hf" and not self.user_llm_args["api_key"]:
            raise ValueError(
                "HF_TOKEN is not set. The simulated user runs on Hugging Face Inference "
                "Providers by default: set HF_TOKEN, or TAU2_USER_PROVIDER=openai|anthropic."
            )
        task_id = task_id or random.Random(seed).choice(self.task_ids)
        self._gym = AgentGymEnv(
            domain=self.domain,
            task_id=task_id,
            max_steps=self.max_steps,
            user_llm=self.user_llm,
            user_llm_args=self.user_llm_args,
        )
        first_message, info = self._gym.reset(seed=seed)
        self._state = Tau2State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            domain=self.domain,
            split=self.split,
            task_id=task_id,
        )
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "user_message": first_message.removeprefix("user: "),
                "policy": info["policy"],
                "task_id": task_id,
            },
        )

    def act(self, action: str) -> str:
        """Advance the simulation with one agent action and return what came back."""
        if self._gym is None:
            raise RuntimeError("call reset() before calling tools")
        _judge.model, _judge.args = self.user_llm, self.user_llm_args
        observation, reward, terminated, _, info = self._gym.step(action)
        if terminated:
            self._state.done = True
            self._state.reward = reward
            self._state.reward_info = json.loads(info["reward_info"])
        return observation.removeprefix("user: ").removeprefix("tool: ")

    def _finish(self, observation: Observation) -> Observation:
        if not self._state.done:
            return observation
        return observation.model_copy(
            update={
                "done": True,
                "reward": self._state.reward,
                "metadata": {
                    **observation.metadata,
                    "reward_info": self._state.reward_info,
                },
            }
        )

    def step(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Observation:
        self._state.step_count += 1
        return self._finish(
            super().step(action, timeout_s=timeout_s or TOOL_TIMEOUT_S, **kwargs)
        )

    async def step_async(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Observation:
        self._state.step_count += 1
        return self._finish(
            await super().step_async(
                action, timeout_s=timeout_s or TOOL_TIMEOUT_S, **kwargs
            )
        )

    def _step_impl(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction."
            },
        )

    @property
    def state(self) -> Tau2State:
        return self._state
