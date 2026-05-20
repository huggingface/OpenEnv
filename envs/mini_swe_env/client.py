# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the Mini SWE Environment.

Provides a typed interface for running SWE-bench rollouts through the
deployed MCP server.

Example::

    from mini_swe_env import MiniSWEEnv

    with MiniSWEEnv(base_url="http://localhost:8000") as env:
        env.reset()
        result = env.run_swe_rollout(
            instance_id="requests__requests-12345",
            repo="psf/requests",
            base_commit="abc123...",
            instruction="Fix the redirect edge case...",
            verify=["python -m pytest tests/test_redirect.py -q"],
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini",
            agent="pi",
        )
        print(f"Reward: {result.reward}")
"""

from __future__ import annotations

import json
from typing import Any

from openenv.core.mcp_client import MCPToolClient

try:
    from .models import SWERolloutResult
except ImportError:  # pragma: no cover
    from models import SWERolloutResult  # type: ignore


class MiniSWEEnv(MCPToolClient):
    """Typed client for the mini_swe_env MCP server.

    Inherits ``reset`` / ``call_tool`` / ``list_tools`` / ``from_docker_image``
    / context-manager semantics from :class:`MCPToolClient`.
    """

    def run_swe_rollout(
        self,
        *,
        # Task fields (SWETask shape).
        instance_id: str = "",
        repo: str = "",
        base_commit: str = "",
        instruction: str = "",
        setup: list[str] | None = None,
        verify: list[str] | None = None,
        timeout_s: int = 1800,
        # Agent config.
        agent: str = "pi",
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        agent_timeout_s: float = 600.0,
        # Infrastructure.
        sandbox_backend: str = "docker",
        sandbox_image: str = "",
        task_id: str = "",
        task_json: str = "",
    ) -> SWERolloutResult:
        """Run one SWE rollout and return the typed result.

        Args:
            instance_id: SWE-bench Lite instance id.
            repo: GitHub ``org/repo`` to clone.
            base_commit: Commit to reset the repo to.
            instruction: Problem statement for the agent.
            setup: Bash commands run before the agent starts.
            verify: Bash commands run after the agent exits.
                Reward = ``passed / total`` unless a command writes a float
                to ``/home/user/logs/verifier/reward.txt`` (override).
            timeout_s: Total timeout for the task.
            agent: Harness CLI (``"pi"`` or ``"opencode"``).
            base_url: OpenAI-compatible LLM endpoint.
            api_key: Bearer token for the LLM.
            model: Model id for the LLM endpoint.
            agent_timeout_s: Wall-clock budget for the agent run.
            sandbox_backend: ``"docker"`` / ``"e2b"`` / ``"hf"``.
            sandbox_image: Docker image or E2B template. Empty = default.
            task_id: Echoed back in the result for traceability.
            task_json: Complete SWETask as JSON string (overrides individual fields).

        Returns:
            A :class:`SWERolloutResult` with reward, verify results,
            file outputs, and diagnostic tails.
        """
        raw = self.call_tool(
            "run_swe_rollout",
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            instruction=instruction,
            setup=list(setup or []),
            verify=list(verify or []),
            timeout_s=timeout_s,
            agent=agent,
            base_url=base_url,
            api_key=api_key,
            model=model,
            agent_timeout_s=agent_timeout_s,
            sandbox_backend=sandbox_backend,
            sandbox_image=sandbox_image,
            task_id=task_id,
            task_json=task_json,
        )
        return SWERolloutResult.model_validate_json(_extract_text(raw))


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
