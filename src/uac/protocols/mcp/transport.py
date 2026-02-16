"""MCP transports — stdio and websocket communication layers.

Each transport satisfies the :class:`MCPTransport` protocol, providing
``connect``, ``send``, ``receive``, and ``close`` methods.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MCPTransport(Protocol):
    """Abstract transport for MCP JSON-RPC communication."""

    async def connect(self) -> None: ...
    async def send(self, data: dict[str, Any]) -> None: ...
    async def receive(self) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class StdioTransport:
    """Communicates with an MCP server via subprocess stdin/stdout.

    Sends and receives newline-delimited JSON.
    """

    def __init__(self, command: str, env: dict[str, str] | None = None) -> None:
        self._command = command
        self._env = env
        self._process: asyncio.subprocess.Process | None = None

    async def connect(self) -> None:
        """Launch the subprocess."""
        import shlex

        parts = shlex.split(self._command)
        self._process = await asyncio.create_subprocess_exec(
            *parts,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )

    async def send(self, data: dict[str, Any]) -> None:
        """Write a JSON line to stdin."""
        if self._process is None or self._process.stdin is None:
            msg = "Transport not connected"
            raise RuntimeError(msg)
        line = json.dumps(data) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        """Read a JSON line from stdout."""
        if self._process is None or self._process.stdout is None:
            msg = "Transport not connected"
            raise RuntimeError(msg)
        line = await self._process.stdout.readline()
        if not line:
            msg = "Transport closed"
            raise RuntimeError(msg)
        return json.loads(line)  # type: ignore[no-any-return]

    async def close(self) -> None:
        """Terminate the subprocess."""
        if self._process is not None:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.terminate()
            await self._process.wait()
            self._process = None


class WebSocketTransport:
    """Communicates with an MCP server over WebSocket.

    Requires the ``websockets`` package (optional dependency ``mcp-ws``).
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._ws: Any = None  # websockets.WebSocketClientProtocol

    async def connect(self) -> None:
        """Open the WebSocket connection."""
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = "websockets package required — install with: pip install uac[mcp-ws]"
            raise ImportError(msg) from exc
        self._ws = await websockets.connect(self._url)  # type: ignore[no-untyped-call]

    async def send(self, data: dict[str, Any]) -> None:
        """Send a JSON message over the WebSocket."""
        if self._ws is None:
            msg = "Transport not connected"
            raise RuntimeError(msg)
        await self._ws.send(json.dumps(data))

    async def receive(self) -> dict[str, Any]:
        """Receive a JSON message from the WebSocket."""
        if self._ws is None:
            msg = "Transport not connected"
            raise RuntimeError(msg)
        raw = await self._ws.recv()
        return json.loads(raw)  # type: ignore[no-any-return]

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
