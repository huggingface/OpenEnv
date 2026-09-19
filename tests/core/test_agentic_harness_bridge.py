# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the loopback MCP bridge (real HTTP server)."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from fastmcp import Client, FastMCP
from openenv.core.harness import (
    AgenticHarnessAdapter,
    HarnessConfig,
    HarnessEnvironment,
    HarnessEvent,
    HarnessEventType,
    HarnessMCPBridge,
)


def make_mcp() -> FastMCP:
    mcp = FastMCP("domain")

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return mcp


class RecordingAdapter(AgenticHarnessAdapter):
    BUILTIN_TOOL_NAMES = frozenset({"read_file"})

    def __init__(self):
        super().__init__(HarnessConfig(name="fake", command=["fake"]))
        self.bridge_url: Optional[str] = "unset"
        self.alive = False

    async def start(self, working_directory: str) -> None:
        self.alive = True

    async def stop(self) -> None:
        self.alive = False

    async def inject_tools(self, tools, bridge_url: Optional[str] = None) -> None:
        self.bridge_url = bridge_url

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        yield HarnessEvent(type=HarnessEventType.TURN_COMPLETE, data={"response": "ok"})

    async def is_alive(self) -> bool:
        return self.alive


class TestBridgeStandalone:
    async def test_serves_tools_over_http(self):
        bridge = HarnessMCPBridge(make_mcp())
        url = bridge.start()
        try:
            assert url.startswith("http://127.0.0.1:")
            assert url.endswith("/mcp")
            async with Client(url) as client:
                tools = await client.list_tools()
                assert [tool.name for tool in tools] == ["add"]
                result = await client.call_tool("add", {"a": 2, "b": 3})
                assert result.content[0].text == "5"
        finally:
            bridge.stop()

    def test_two_bridges_get_distinct_ports(self):
        bridge_a = HarnessMCPBridge(make_mcp())
        bridge_b = HarnessMCPBridge(make_mcp())
        try:
            url_a = bridge_a.start()
            url_b = bridge_b.start()
            assert url_a != url_b
        finally:
            bridge_a.stop()
            bridge_b.stop()

    def test_stop_is_idempotent(self):
        bridge = HarnessMCPBridge(make_mcp())
        bridge.start()
        bridge.stop()
        bridge.stop()
        assert bridge.url is None

    def test_stop_before_start_is_noop(self):
        bridge = HarnessMCPBridge(make_mcp())
        bridge.stop()
        assert bridge.url is None

    def test_start_is_idempotent_while_running(self):
        bridge = HarnessMCPBridge(make_mcp())
        try:
            assert bridge.start() == bridge.start()
        finally:
            bridge.stop()


class TestBridgeEnvironmentIntegration:
    async def test_reset_passes_live_bridge_url(self):
        adapter = RecordingAdapter()
        env = HarnessEnvironment(adapter=adapter, mcp=make_mcp())
        try:
            await env.reset_async()
            assert adapter.bridge_url is not None
            async with Client(adapter.bridge_url) as client:
                tools = await client.list_tools()
                assert [tool.name for tool in tools] == ["add"]
        finally:
            env.close()
        assert env._bridge is None

    async def test_no_tools_no_bridge(self):
        adapter = RecordingAdapter()
        env = HarnessEnvironment(adapter=adapter, mcp=None)
        try:
            await env.reset_async()
            assert adapter.bridge_url is None
        finally:
            env.close()


class TestRenamedToolsAreServed:
    """Conflict resolution renames tools before injection; the bridge must
    answer to the injected names or the harness calls a name that 404s."""

    @staticmethod
    def make_colliding_mcp() -> FastMCP:
        mcp = FastMCP("domain")

        @mcp.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @mcp.tool
        def read_file(path: str) -> str:
            """Collides with the adapter's built-in read_file."""
            return f"contents of {path}"

        return mcp

    async def test_renamed_tool_is_listed_and_callable(self):
        adapter = RecordingAdapter()  # BUILTIN_TOOL_NAMES == {"read_file"}
        env = HarnessEnvironment(adapter=adapter, mcp=self.make_colliding_mcp())
        try:
            obs = await env.reset_async()
            assert sorted(obs.metadata["injected_tools"]) == ["add", "env_read_file"]

            async with Client(adapter.bridge_url) as client:
                served = sorted(tool.name for tool in await client.list_tools())
                assert served == ["add", "env_read_file"]

                # The renamed tool must actually resolve, not just be listed.
                result = await client.call_tool("env_read_file", {"path": "a.py"})
                assert result.content[0].text == "contents of a.py"

                # The un-renamed one is untouched.
                assert (await client.call_tool("add", {"a": 2, "b": 3})).content[
                    0
                ].text == "5"
        finally:
            env.close()

    async def test_no_renames_serves_the_source_server_directly(self):
        adapter = RecordingAdapter()
        env = HarnessEnvironment(adapter=adapter, mcp=make_mcp())  # only "add"
        try:
            await env.reset_async()
            async with Client(adapter.bridge_url) as client:
                assert [t.name for t in await client.list_tools()] == ["add"]
        finally:
            env.close()
