# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl[vllm]",
#     "peft",
#     "trackio",
#     "kernels",
#     "openenv-browsergym @ git+https://huggingface.co/spaces/openenv/browsergym_env",
# ]
# ///

"""Harness-oriented BrowserGym rollout example for TRL.

This mirrors the usual BrowserGym + TRL setup, but moves environment interaction
behind the ResourceSession / HarnessAdapter abstractions so the same session
surface can be reused by white-box training and black-box evaluation harnesses.
"""

from __future__ import annotations

import re

from browsergym_env import BrowserGymEnv
from browsergym_env.harness import (
    BrowserGymSessionFactory,
    build_browsergym_action_tool_call,
)
from transformers import AutoTokenizer

from openenv.core.harness import (
    HarnessRunLimits,
    MCPHarnessAdapter,
    ModelStepResult,
    build_harness_rollout_func,
)
from openenv.core.llm_client import LLMResponse
from trl.experimental.openenv import generate_rollout_completions

SYSTEM_PROMPT = """You control a web browser through BrowserGym actions.

Reply with exactly one BrowserGym action such as:
- click('13')
- fill('42', 'hello world')
- send_keys('Enter')
- scroll('down')
- noop()
"""

_ACTION_PREFIX_RE = re.compile(r"^(action|next action)\s*[:-]\s*", re.IGNORECASE)
_ACTION_PATTERN = re.compile(
    r"(click|fill|send_keys|scroll|noop)\s*\(.*?\)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_browsergym_action(response_text: str) -> str:
    if not response_text:
        return "noop()"

    for raw_line in response_text.splitlines():
        line = _ACTION_PREFIX_RE.sub("", raw_line.strip().strip("`"))
        if not line:
            continue
        if match := _ACTION_PATTERN.search(line):
            action = re.sub(r"\s+", " ", match.group(0).strip()).rstrip(";")
            open_paren = action.find("(")
            if open_paren > 0:
                return f"{action[:open_paren].lower()}{action[open_paren:]}"
            return action

    if match := _ACTION_PATTERN.search(response_text):
        action = re.sub(r"\s+", " ", match.group(0).strip()).rstrip(";")
        open_paren = action.find("(")
        if open_paren > 0:
            return f"{action[:open_paren].lower()}{action[open_paren:]}"
        return action
    return "noop()"


def _build_safe_browsergym_tool_call(response_text: str):
    try:
        return build_browsergym_action_tool_call(
            _extract_browsergym_action(response_text),
        )
    except ValueError:
        return build_browsergym_action_tool_call("noop()")


def build_trl_browsergym_model_step(trainer, tokenizer: AutoTokenizer):
    """Wrap TRL generation so the harness can drive BrowserGym via tool calls."""

    def model_step(messages, tools, sampling):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT.strip()}, *messages],
            add_generation_prompt=True,
            tokenize=False,
        )
        rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
        completion_text = rollout_output.get("text") or tokenizer.decode(
            rollout_output["completion_ids"],
            skip_special_tokens=True,
        )
        tool_call = _build_safe_browsergym_tool_call(completion_text)
        return ModelStepResult(
            response=LLMResponse(
                content=completion_text,
                tool_calls=[tool_call],
            ),
            prompt_ids=list(rollout_output["prompt_ids"]),
            completion_ids=list(rollout_output["completion_ids"]),
            logprobs=list(rollout_output["logprobs"]),
        )

    return model_step


def build_browsergym_rollout_func(
    space_url: str, tokenizer: AutoTokenizer, max_steps: int
):
    """Create a TRL rollout function backed by BrowserGym sessions."""

    session_factory = BrowserGymSessionFactory(
        client_factory=lambda: BrowserGymEnv(base_url=space_url),
    )

    return build_harness_rollout_func(
        session_factory=session_factory,
        harness_adapter=MCPHarnessAdapter(),
        model_step_builder=lambda trainer, session: build_trl_browsergym_model_step(
            trainer,
            tokenizer,
        ),
        limits=HarnessRunLimits(max_turns=max_steps),
    )


__all__ = [
    "SYSTEM_PROMPT",
    "build_browsergym_rollout_func",
    "build_trl_browsergym_model_step",
]
