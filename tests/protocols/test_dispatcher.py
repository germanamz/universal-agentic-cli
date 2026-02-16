"""Tests for ToolDispatcher routing."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from uac.core.interface.models import ToolCall, ToolResult
from uac.protocols.dispatcher import ToolDispatcher
from uac.protocols.errors import ToolNotFoundError


def _make_provider(
    tools: list[dict[str, Any]] | None = None,
    result_text: str = "done",
) -> MagicMock:
    provider = MagicMock()
    provider.discover_tools = AsyncMock(
        return_value=tools
        or [
            {
                "type": "function",
                "function": {"name": "tool_a", "description": "", "parameters": {}},
            }
        ]
    )
    provider.execute_tool = AsyncMock(
        return_value=ToolResult.from_text(tool_call_id="", text=result_text)
    )
    return provider


class TestToolDispatcher:
    async def test_register_discovers_tools(self) -> None:
        dispatcher = ToolDispatcher()
        provider = _make_provider()
        await dispatcher.register(provider)
        provider.discover_tools.assert_awaited_once()

    async def test_all_tools_returns_merged(self) -> None:
        dispatcher = ToolDispatcher()
        p1 = _make_provider(
            [{"type": "function", "function": {"name": "a", "parameters": {}}}]
        )
        p2 = _make_provider(
            [{"type": "function", "function": {"name": "b", "parameters": {}}}]
        )
        await dispatcher.register(p1)
        await dispatcher.register(p2)
        tools = dispatcher.all_tools()
        names = {t["function"]["name"] for t in tools}
        assert names == {"a", "b"}

    async def test_execute_routes_to_correct_provider(self) -> None:
        dispatcher = ToolDispatcher()
        p1 = _make_provider(
            [{"type": "function", "function": {"name": "alpha", "parameters": {}}}],
            result_text="from-alpha",
        )
        p2 = _make_provider(
            [{"type": "function", "function": {"name": "beta", "parameters": {}}}],
            result_text="from-beta",
        )
        await dispatcher.register(p1)
        await dispatcher.register(p2)

        call = ToolCall(name="beta", arguments={"x": 1})
        result = await dispatcher.execute(call)
        p2.execute_tool.assert_awaited_once_with("beta", {"x": 1})
        assert result.content[0].text == "from-beta"  # type: ignore[union-attr]

    async def test_execute_unknown_tool_raises(self) -> None:
        dispatcher = ToolDispatcher()
        call = ToolCall(name="nonexistent", arguments={})
        with pytest.raises(ToolNotFoundError, match="nonexistent"):
            await dispatcher.execute(call)

    async def test_execute_all_concurrent(self) -> None:
        dispatcher = ToolDispatcher()
        provider = _make_provider(
            [{"type": "function", "function": {"name": "t", "parameters": {}}}],
            result_text="r",
        )
        await dispatcher.register(provider)

        calls = [ToolCall(name="t", arguments={}), ToolCall(name="t", arguments={})]
        results = await dispatcher.execute_all(calls)
        assert len(results) == 2
        assert provider.execute_tool.await_count == 2
