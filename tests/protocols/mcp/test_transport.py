"""Tests for MCP transports (stdio and websocket) with mocks."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uac.protocols.mcp.transport import MCPTransport, StdioTransport, WebSocketTransport


class TestMCPTransportProtocol:
    def test_stdio_satisfies_protocol(self) -> None:
        transport = StdioTransport(command="echo test")
        assert isinstance(transport, MCPTransport)

    def test_websocket_satisfies_protocol(self) -> None:
        transport = WebSocketTransport(url="ws://localhost:8080")
        assert isinstance(transport, MCPTransport)


class TestStdioTransport:
    async def test_connect_launches_subprocess(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            transport = StdioTransport(command="echo hello")
            await transport.connect()
            mock_exec.assert_awaited_once()

    async def test_send_writes_json_line(self) -> None:
        mock_proc = AsyncMock()
        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        mock_proc.stdin = mock_stdin

        transport = StdioTransport(command="echo test")
        transport._process = mock_proc

        data = {"jsonrpc": "2.0", "method": "test"}
        await transport.send(data)

        written = mock_stdin.write.call_args[0][0]
        assert json.loads(written.decode()) == data

    async def test_receive_reads_json_line(self) -> None:
        expected = {"jsonrpc": "2.0", "result": {"ok": True}}
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            return_value=(json.dumps(expected) + "\n").encode()
        )

        mock_proc = AsyncMock()
        mock_proc.stdout = mock_stdout

        transport = StdioTransport(command="echo test")
        transport._process = mock_proc

        result = await transport.receive()
        assert result == expected

    async def test_receive_empty_raises(self) -> None:
        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=b"")

        mock_proc = AsyncMock()
        mock_proc.stdout = mock_stdout

        transport = StdioTransport(command="echo test")
        transport._process = mock_proc

        with pytest.raises(RuntimeError, match="closed"):
            await transport.receive()

    async def test_send_without_connect_raises(self) -> None:
        transport = StdioTransport(command="echo test")
        with pytest.raises(RuntimeError, match="not connected"):
            await transport.send({"test": True})

    async def test_close_terminates_process(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.close = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock()

        transport = StdioTransport(command="echo test")
        transport._process = mock_proc

        await transport.close()
        mock_proc.terminate.assert_called_once()
        assert transport._process is None

    async def test_connect_with_env(self) -> None:
        env = {"API_KEY": "secret"}
        mock_proc = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            transport = StdioTransport(command="tool serve", env=env)
            await transport.connect()
            call_kwargs = mock_exec.call_args
            assert call_kwargs.kwargs["env"] == env


class TestWebSocketTransport:
    async def test_connect_opens_websocket(self) -> None:
        mock_ws = AsyncMock()
        mock_module = MagicMock()
        mock_module.connect = AsyncMock(return_value=mock_ws)

        with patch.dict("sys.modules", {"websockets": mock_module}):
            transport = WebSocketTransport(url="ws://localhost:8080")
            await transport.connect()
            assert transport._ws is mock_ws

    async def test_send_writes_json(self) -> None:
        mock_ws = AsyncMock()
        transport = WebSocketTransport(url="ws://localhost:8080")
        transport._ws = mock_ws

        data = {"method": "test"}
        await transport.send(data)
        mock_ws.send.assert_awaited_once_with(json.dumps(data))

    async def test_receive_reads_json(self) -> None:
        expected = {"result": "ok"}
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps(expected))

        transport = WebSocketTransport(url="ws://localhost:8080")
        transport._ws = mock_ws

        result = await transport.receive()
        assert result == expected

    async def test_close_closes_websocket(self) -> None:
        mock_ws = AsyncMock()
        transport = WebSocketTransport(url="ws://localhost:8080")
        transport._ws = mock_ws

        await transport.close()
        mock_ws.close.assert_awaited_once()
        assert transport._ws is None

    async def test_send_without_connect_raises(self) -> None:
        transport = WebSocketTransport(url="ws://localhost:8080")
        with pytest.raises(RuntimeError, match="not connected"):
            await transport.send({"test": True})

    async def test_connect_without_websockets_raises(self) -> None:
        transport = WebSocketTransport(url="ws://localhost:8080")
        with (
            patch.dict("sys.modules", {"websockets": None}),
            pytest.raises(ImportError, match="websockets"),
        ):
            await transport.connect()
