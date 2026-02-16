"""Tests for MCPClient with mocked transport."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uac.core.orchestration.models import MCPServerRef
from uac.protocols.errors import ConnectionError, ToolExecutionError, ToolNotFoundError
from uac.protocols.mcp.client import MCPClient


def _make_transport(responses: list[dict] | None = None) -> MagicMock:
    """Create a mock transport that returns a sequence of JSON-RPC responses."""
    transport = MagicMock()
    transport.connect = AsyncMock()
    transport.close = AsyncMock()
    transport.send = AsyncMock()

    if responses is None:
        responses = [
            # Default: initialize response
            {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}},
        ]

    transport.receive = AsyncMock(side_effect=responses)
    return transport


def _tools_list_response(request_id: int = 2) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
                {
                    "name": "write_file",
                    "description": "Write a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            ]
        },
    }


def _tool_call_response(text: str = "file contents", request_id: int = 3) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": text}],
        },
    }


class TestMCPClientConnect:
    async def test_connect_performs_handshake(self) -> None:
        ref = MCPServerRef(name="test", command="echo test")
        transport = _make_transport()

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            client = MCPClient(ref)
            await client.connect()

            transport.connect.assert_awaited_once()
            # Should have sent an initialize request
            transport.send.assert_awaited_once()
            sent = transport.send.call_args[0][0]
            assert sent["method"] == "initialize"

    async def test_connect_error_raises(self) -> None:
        ref = MCPServerRef(name="test", command="bad-command")
        transport = MagicMock()
        transport.connect = AsyncMock(side_effect=OSError("spawn failed"))

        with (
            patch.object(MCPClient, "_create_transport", return_value=transport),
            pytest.raises(ConnectionError),
        ):
            client = MCPClient(ref)
            await client.connect()

    async def test_context_manager(self) -> None:
        ref = MCPServerRef(name="test", command="echo test")
        transport = _make_transport()

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            async with MCPClient(ref) as _client:
                pass
            transport.close.assert_awaited_once()


class TestMCPClientDiscovery:
    async def test_discover_tools(self) -> None:
        ref = MCPServerRef(name="test", command="echo test")
        transport = _make_transport([
            {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}},
            _tools_list_response(),
        ])

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            async with MCPClient(ref) as client:
                tools = await client.discover_tools()

        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"read_file", "write_file"}
        # Verify schema structure
        read_file = next(t for t in tools if t["function"]["name"] == "read_file")
        assert read_file["type"] == "function"
        assert "path" in read_file["function"]["parameters"]["properties"]

    async def test_discover_empty_tools(self) -> None:
        ref = MCPServerRef(name="test", command="echo test")
        transport = _make_transport([
            {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}},
        ])

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            async with MCPClient(ref) as client:
                tools = await client.discover_tools()

        assert tools == []


class TestMCPClientExecution:
    async def test_execute_tool_success(self) -> None:
        ref = MCPServerRef(name="test", command="echo test")
        transport = _make_transport([
            {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}},
            _tools_list_response(),
            _tool_call_response("hello world"),
        ])

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            async with MCPClient(ref) as client:
                await client.discover_tools()
                result = await client.execute_tool("read_file", {"path": "/tmp/test"})

        assert result.content[0].text == "hello world"  # type: ignore[union-attr]

    async def test_execute_unknown_tool_raises(self) -> None:
        ref = MCPServerRef(name="test", command="echo test")
        transport = _make_transport([
            {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}},
            _tools_list_response(),
        ])

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            async with MCPClient(ref) as client:
                await client.discover_tools()
                with pytest.raises(ToolNotFoundError, match="nonexistent"):
                    await client.execute_tool("nonexistent", {})

    async def test_execute_tool_error_response(self) -> None:
        ref = MCPServerRef(name="test", command="echo test")
        transport = _make_transport([
            {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}},
            _tools_list_response(),
            {
                "jsonrpc": "2.0",
                "id": 3,
                "error": {"code": -32000, "message": "File not found"},
            },
        ])

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            async with MCPClient(ref) as client:
                await client.discover_tools()
                with pytest.raises(ToolExecutionError, match="read_file"):
                    await client.execute_tool("read_file", {"path": "/bad"})


class TestMCPClientTransportFactory:
    def test_stdio_transport(self) -> None:
        ref = MCPServerRef(name="test", transport="stdio", command="npx @mcp/fs")
        client = MCPClient(ref)
        transport = client._create_transport()
        assert transport.__class__.__name__ == "StdioTransport"

    def test_websocket_transport(self) -> None:
        ref = MCPServerRef(name="test", transport="websocket", url="ws://localhost:8080")
        client = MCPClient(ref)
        transport = client._create_transport()
        assert transport.__class__.__name__ == "WebSocketTransport"

    def test_stdio_without_command_raises(self) -> None:
        ref = MCPServerRef(name="test", transport="stdio")
        client = MCPClient(ref)
        with pytest.raises(ValueError, match="command"):
            client._create_transport()

    def test_websocket_without_url_raises(self) -> None:
        ref = MCPServerRef(name="test", transport="websocket")
        client = MCPClient(ref)
        with pytest.raises(ValueError, match="url"):
            client._create_transport()
