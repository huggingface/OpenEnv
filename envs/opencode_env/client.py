# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the deployed opencode_env server.

The server exposes a single MCP tool ``run_rollout`` that runs one OpenCode
rollout in an E2B sandbox and returns a JSON-serialized :class:`RolloutResult`.

Example::

    from opencode_env import OpenCodeEnv

    with OpenCodeEnv(base_url="https://adithya-sk-opencode-env.hf.space") as env:
        env.reset()
        result = env.run_rollout(
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini",
            instruction="Create binary_search.py exposing def binary_search(arr, target) -> int...",
            setup=[],
            verify=["python /home/user/test.py"],
            task_id="binary_search_v1",
        )
        print(result.reward, len(result.proxy_turns))
"""

from __future__ import annotations

import json
from typing import Any

from openenv.core.mcp_client import MCPToolClient

try:
    from .models import RolloutResult
except ImportError:  # pragma: no cover
    from models import RolloutResult  # type: ignore


class OpenCodeEnv(MCPToolClient):
    """Typed client for the opencode_env MCP server.

    Inherits ``reset`` / ``call_tool`` / ``list_tools`` / ``from_docker_image``
    / context-manager semantics from :class:`MCPToolClient`.
    """

    def run_rollout(
        self,
        *,
        # Endpoint — pass either the shorthand selector OR explicit fields.
        endpoint: str = "",                # "vllm" | "openai" | "hf_router"
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        # Task — the "list of bash commands" shape
        instruction: str,
        setup: list[str] | None = None,
        verify: list[str] | None = None,
        # Bookkeeping / tunables
        task_id: str = "",
        mode: str = "transparent_proxy",
        disable_thinking: bool | None = None,
        max_tokens_cap: int = 4096,
        top_logprobs: int = 5,
        agent_timeout_s: float = 600.0,
        template: str = "",
    ) -> RolloutResult:
        """Run one OpenCode rollout and return the typed result.

        Args:
            base_url: OpenAI-compatible LLM endpoint (with trailing /v1).
            api_key: Bearer token for the LLM. Use ``"intercepted"`` for vLLM
                if it doesn't enforce auth.
            model: Model id understood by the LLM endpoint
                (e.g. ``"gpt-4o-mini"``, ``"Qwen/Qwen3.5-4B"``,
                ``"Qwen/Qwen3-4B-Instruct-2507:nscale"``).
            instruction: Prompt passed to ``opencode run``.
            setup: Bash commands run sequentially **before** the agent starts.
                Each command runs in the sandbox; non-zero exit aborts setup.
            verify: Bash commands run sequentially **after** the agent exits.
                Reward = ``passed_count / total`` unless any command writes a
                float to ``/home/user/logs/verifier/reward.txt`` (override).
            task_id: Echoed back in the result for traceability.
            mode: ``"transparent_proxy"`` (captures per-token logprobs via
                an in-sandbox FastAPI proxy) or ``"black_box"`` (no proxy).
            disable_thinking: Inject
                ``chat_template_kwargs.enable_thinking=false`` on forwarded
                requests. Needed for Qwen3.5 vLLM; harmless on Instruct
                variants; rejected by OpenAI direct.
            max_tokens_cap: Clamp on per-turn ``max_tokens``. OpenCode asks
                for ~32k by default; gpt-4o-mini caps at 16k.
            top_logprobs: Top-k logprobs requested upstream. HF Router caps
                at 5; OpenAI accepts up to 20; vLLM is unbounded.
            agent_timeout_s: Hard wall-clock budget for one ``opencode run``.
            template: E2B template name (e.g. ``"opencode-rl"``). Empty
                string uses the default (slow) base image.

        Returns:
            A :class:`RolloutResult` with reward, per-turn logprobs, file
            outputs, setup/verify results, and diagnostic tails.
        """
        raw = self.call_tool(
            "run_rollout",
            endpoint=endpoint,
            base_url=base_url,
            api_key=api_key,
            model=model,
            instruction=instruction,
            setup=list(setup or []),
            verify=list(verify or []),
            task_id=task_id,
            mode=mode,
            disable_thinking=disable_thinking,
            max_tokens_cap=max_tokens_cap,
            top_logprobs=top_logprobs,
            agent_timeout_s=agent_timeout_s,
            template=template,
        )
        return RolloutResult.model_validate_json(_extract_text(raw))


def _extract_text(result: Any) -> str:
    """Pull the JSON text out of whatever shape the MCP layer returns.

    Handles the three shapes :meth:`MCPToolClient.call_tool` may surface:
    a raw string, a ``CallToolObservation``-like object with
    ``.result.content[0].text``, or a dict with ``content[0]["text"]``.
    """
    if isinstance(result, str):
        return result

    inner = getattr(result, "result", None)
    if inner is not None:
        content = getattr(inner, "content", None)
        if content:
            first = content[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                return text
            if isinstance(first, dict) and "text" in first:
                return first["text"]

    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
        nested = result.get("result")
        if isinstance(nested, dict):
            content = nested.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
        return json.dumps(result, default=str)

    content = getattr(result, "content", None)
    if content:
        first = content[0]
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text

    return str(result)
