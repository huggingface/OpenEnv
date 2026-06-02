# envs/finqa_env/client.py
"""
Client for the FinQA environment.

This client connects to a running FinQA environment server and provides
a Python interface for interacting with it via MCP tools. Async by default.

Example:
    >>> from envs.finqa_env import FinQAEnv
    >>>
    >>> async with FinQAEnv(base_url="http://localhost:8000") as env:
    ...     await env.reset()
    ...     tools = await env.list_tools()
    ...     result = await env.call_tool("get_descriptions", company_name="alphabet")
    ...     print(result)
    ...     result = await env.call_tool("submit_answer", answer="6.118")
"""

from openenv.core.mcp_client import MCPToolClient


class FinQAEnv(MCPToolClient):
    """
    Client for the FinQA environment.

    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available tools
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment
    - step(action): Execute an action
    """

    pass  # MCPToolClient provides all needed functionality
