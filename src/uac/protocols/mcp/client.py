"""MCPClient — connects to an MCP server and exposes tools.

Implements tool discovery (``tools/list``) and execution (``tools/call``)
over an :class:`MCPTransport`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from uac.core.interface.models import ToolResult
from uac.protocols.errors import ConnectionError, ToolExecutionError, ToolNotFoundError
from uac.protocols.mcp.models import JsonRpcRequest, JsonRpcResponse, MCPToolDef
from uac.protocols.mcp.transport import MCPTransport, StdioTransport, WebSocketTransport

if TYPE_CHECKING:
    from uac.core.orchestration.models import MCPServerRef


class MCPClient:
    """Async context manager that connects to an MCP server.

    Satisfies the :class:`~uac.protocols.provider.ToolProvider` protocol.

    Usage::

        ref = MCPServerRef(name="fs", command="npx @mcp/filesystem")
        async with MCPClient(ref) as client:
            tools = await client.discover_tools()
            result = await client.execute_tool("read_file", {"path": "/tmp/x"})
    """

    def __init__(self, server_ref: MCPServerRef) -> None:
        self._ref = server_ref
        self._transport: MCPTransport | None = None
        self._tools: dict[str, MCPToolDef] = {}
        self._next_id = 1

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Create the transport, connect, and perform the initialize handshake."""
        self._transport = self._create_transport()
        try:
            await self._transport.connect()
        except Exception as exc:
            raise ConnectionError(str(exc)) from exc
        await self._handshake()

    async def close(self) -> None:
        """Close the underlying transport."""
        if self._transport is not None:
            await self._transport.close()
            self._transport = None

    async def discover_tools(self) -> list[dict[str, Any]]:
        """Send ``tools/list`` and convert results to OpenAI function schemas."""
        response = await self._send_request("tools/list")
        raw_tools: list[dict[str, Any]] = (
            cast("list[dict[str, Any]]", response.result.get("tools", []))
            if response.result
            else []
        )
        self._tools.clear()

        schemas: list[dict[str, Any]] = []
        for raw in raw_tools:
            tool_def = MCPToolDef.model_validate(raw)
            self._tools[tool_def.name] = tool_def
            schemas.append(self._to_function_schema(tool_def))
        return schemas

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Send ``tools/call`` for the named tool."""
        if name not in self._tools:
            raise ToolNotFoundError(name)

        response = await self._send_request(
            "tools/call",
            params={"name": name, "arguments": arguments},
        )

        if response.error is not None:
            raise ToolExecutionError(name, response.error.message)

        # Extract text from the response content
        text = self._extract_content(response)
        return ToolResult.from_text(tool_call_id="", text=text)

    def _create_transport(self) -> MCPTransport:
        """Build the appropriate transport from the server reference."""
        if self._ref.transport == "stdio":
            if not self._ref.command:
                msg = "MCPServerRef with stdio transport must specify 'command'"
                raise ValueError(msg)
            env = dict(self._ref.env) if self._ref.env else None
            return StdioTransport(command=self._ref.command, env=env)
        if not self._ref.url:
            msg = "MCPServerRef with websocket transport must specify 'url'"
            raise ValueError(msg)
        return WebSocketTransport(url=self._ref.url)

    async def _handshake(self) -> None:
        """Perform the MCP initialize handshake."""
        await self._send_request(
            "initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "uac", "version": "0.1.0"},
            },
        )

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> JsonRpcResponse:
        """Send a JSON-RPC request and wait for the response."""
        if self._transport is None:
            msg = "Client not connected"
            raise RuntimeError(msg)

        request_id = self._next_id
        self._next_id += 1

        request = JsonRpcRequest(
            method=method,
            id=request_id,
            params=params or {},
        )
        await self._transport.send(request.model_dump())
        raw = await self._transport.receive()
        return JsonRpcResponse.model_validate(raw)

    @staticmethod
    def _to_function_schema(tool_def: MCPToolDef) -> dict[str, Any]:
        """Convert an MCPToolDef to an OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": tool_def.input_schema or {"type": "object", "properties": {}},
            },
        }

    @staticmethod
    def _extract_content(response: JsonRpcResponse) -> str:
        """Extract text from a tools/call response."""
        if response.result is None:
            return ""
        content = cast("list[dict[str, Any]]", response.result.get("content", []))
        parts: list[str] = []
        for item in content:
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts) if parts else str(response.result)
