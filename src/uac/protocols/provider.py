"""ToolProvider protocol — the common interface for all protocol implementations.

Every protocol adapter (MCP, A2A, UTCP) satisfies this protocol so that the
:class:`ToolDispatcher` can route tool calls without knowing the underlying
transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uac.core.interface.models import ToolResult


@runtime_checkable
class ToolProvider(Protocol):
    """Discovers and executes tools exposed by an external service."""

    async def discover_tools(self) -> list[dict[str, Any]]:
        """Return tools as OpenAI-compatible function schemas.

        Each dict follows the shape::

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { ... }   # JSON Schema
                }
            }
        """
        ...

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name and return its result."""
        ...
