"""Tests for the ToolProvider protocol."""

from typing import Any

from uac.core.interface.models import ToolResult
from uac.protocols.provider import ToolProvider


class _DummyProvider:
    """Minimal class that satisfies ToolProvider."""

    async def discover_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "noop", "parameters": {}}}]

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult.from_text(tool_call_id="x", text="ok")


class TestToolProvider:
    def test_runtime_checkable(self) -> None:
        provider = _DummyProvider()
        assert isinstance(provider, ToolProvider)

    def test_non_provider_fails(self) -> None:
        assert not isinstance("not a provider", ToolProvider)

    async def test_discover_tools(self) -> None:
        provider = _DummyProvider()
        tools = await provider.discover_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "noop"

    async def test_execute_tool(self) -> None:
        provider = _DummyProvider()
        result = await provider.execute_tool("noop", {})
        assert result.content[0].text == "ok"  # type: ignore[union-attr]
