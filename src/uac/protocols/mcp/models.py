"""MCP models — JSON-RPC 2.0 messages and tool definitions.

Implements the message format used by the Model Context Protocol for
tool discovery (``tools/list``) and execution (``tools/call``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 envelope
# ---------------------------------------------------------------------------


class JsonRpcRequest(BaseModel):
    """A JSON-RPC 2.0 request message."""

    jsonrpc: str = "2.0"
    method: str
    id: int | str = 1
    params: dict[str, Any] = {}


class JsonRpcError(BaseModel):
    """A JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any = None


class JsonRpcResponse(BaseModel):
    """A JSON-RPC 2.0 response message."""

    jsonrpc: str = "2.0"
    id: int | str = 1
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


# ---------------------------------------------------------------------------
# MCP-specific payloads
# ---------------------------------------------------------------------------


class MCPToolDef(BaseModel):
    """A tool definition as returned by ``tools/list``."""

    model_config = {"populate_by_name": True}

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict, alias="inputSchema")
