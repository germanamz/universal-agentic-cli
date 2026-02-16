"""ToolDispatcher — routes tool calls to the correct ToolProvider."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from uac.protocols.errors import ToolNotFoundError

if TYPE_CHECKING:
    from uac.core.interface.models import ToolCall, ToolResult
    from uac.protocols.provider import ToolProvider


class ToolDispatcher:
    """Maintains a name-to-provider map and dispatches tool calls.

    Usage::

        dispatcher = ToolDispatcher()
        await dispatcher.register(mcp_client)
        await dispatcher.register(utcp_executor)

        tools = dispatcher.all_tools()          # merged list
        result = await dispatcher.execute(call)  # routes to correct provider
    """

    def __init__(self) -> None:
        self._providers: list[ToolProvider] = []
        self._tool_map: dict[str, ToolProvider] = {}
        self._tool_schemas: list[dict[str, Any]] = []

    async def register(self, provider: ToolProvider) -> None:
        """Discover tools from *provider* and add them to the routing table."""
        self._providers.append(provider)
        schemas = await provider.discover_tools()
        for schema in schemas:
            name: str = schema["function"]["name"]
            self._tool_map[name] = provider
            self._tool_schemas.append(schema)

    def all_tools(self) -> list[dict[str, Any]]:
        """Return the merged list of all tool schemas across providers."""
        return list(self._tool_schemas)

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Route a single tool call to its owning provider."""
        provider = self._tool_map.get(tool_call.name)
        if provider is None:
            raise ToolNotFoundError(tool_call.name)
        return await provider.execute_tool(tool_call.name, tool_call.arguments)

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls concurrently."""
        return list(await asyncio.gather(*[self.execute(tc) for tc in tool_calls]))
