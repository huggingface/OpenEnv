# SPDX-License-Identifier: BSD-3-Clause

"""Claude Code as the agent of a τ²-bench conversation, through `HarnessEnvironment`.

`tau2_env` holds the task: the simulated customer, the domain's database and its
tools. `HarnessEnvironment` holds Claude Code, with the domain's tools injected
over MCP. Each customer message is one `step()`, Claude Code's reply goes back to
the simulated customer, and τ²-bench scores the conversation when the customer
is done.
"""

from __future__ import annotations

import asyncio

from claude_code_adapter import ClaudeCodeAdapter
from fastmcp import FastMCP
from openenv.core.harness import HarnessAction, HarnessConfig, HarnessEnvironment
from tau2_env.server.tau2_environment import Tau2Environment, without_end_tokens

# Claude Code talks to the customer through its replies, not through these tools.
CONVERSATION_TOOLS = ("respond_to_user", "done")
AGENT_PROMPT = (
    "You are a customer service agent. Each message you receive is from the "
    "customer, and your reply is sent back to them. Use the tools to look things "
    "up and to make changes, and follow this policy. When the policy states the "
    "current time, use it instead of today's date:\n\n"
)


def domain_tools(tau2: Tau2Environment) -> FastMCP:
    """The domain's tools of a τ²-bench environment, to inject into the harness."""
    mcp = FastMCP(f"tau2_{tau2.domain}")
    for tool in asyncio.run(tau2.mcp_server.list_tools()):
        if tool.name not in CONVERSATION_TOOLS:
            mcp.add_tool(tool)
    return mcp


def harness_for(
    tau2: Tau2Environment, policy: str, config: HarnessConfig
) -> HarnessEnvironment:
    """Claude Code with the domain's tools and policy, for one τ²-bench task."""
    return HarnessEnvironment(
        adapter=ClaudeCodeAdapter(config, system_prompt=AGENT_PROMPT + policy),
        mcp=domain_tools(tau2),
    )


def converse(
    tau2: Tau2Environment,
    harness: HarnessEnvironment,
    first_message: str,
    max_turns: int = 30,
):
    """
    Run the conversation until the customer is done.

    Yields `(customer_message, observation)` for each turn, where the observation
    holds Claude Code's reply and the turn's events.
    """
    message = first_message
    for _ in range(max_turns):
        observation = harness.step(HarnessAction(message=message))
        yield message, observation
        if observation.done:  # Claude Code crashed or timed out
            return
        reply = observation.metadata.get("response") or "(no reply)"
        message = without_end_tokens(tau2.act(reply))
        if tau2.state.done:
            if message:
                yield message, None
            return
