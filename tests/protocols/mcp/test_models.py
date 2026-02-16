"""Tests for MCP JSON-RPC models."""

from uac.protocols.mcp.models import JsonRpcError, JsonRpcRequest, JsonRpcResponse, MCPToolDef


class TestJsonRpcRequest:
    def test_defaults(self) -> None:
        req = JsonRpcRequest(method="tools/list")
        assert req.jsonrpc == "2.0"
        assert req.id == 1
        assert req.params == {}

    def test_custom_values(self) -> None:
        req = JsonRpcRequest(method="tools/call", id=42, params={"name": "test"})
        assert req.method == "tools/call"
        assert req.id == 42
        assert req.params["name"] == "test"

    def test_round_trip(self) -> None:
        req = JsonRpcRequest(method="initialize", id="abc", params={"version": "1.0"})
        data = req.model_dump()
        restored = JsonRpcRequest.model_validate(data)
        assert restored == req

    def test_serialization_shape(self) -> None:
        req = JsonRpcRequest(method="tools/list")
        data = req.model_dump()
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "tools/list"


class TestJsonRpcError:
    def test_basic(self) -> None:
        err = JsonRpcError(code=-32600, message="Invalid request")
        assert err.code == -32600
        assert err.data is None

    def test_with_data(self) -> None:
        err = JsonRpcError(code=-32601, message="Not found", data={"tool": "x"})
        assert err.data["tool"] == "x"


class TestJsonRpcResponse:
    def test_success(self) -> None:
        resp = JsonRpcResponse(result={"tools": []})
        assert resp.result is not None
        assert resp.error is None

    def test_error(self) -> None:
        resp = JsonRpcResponse(
            error=JsonRpcError(code=-32600, message="Bad request")
        )
        assert resp.error is not None
        assert resp.result is None

    def test_round_trip(self) -> None:
        resp = JsonRpcResponse(id=5, result={"data": "value"})
        data = resp.model_dump()
        restored = JsonRpcResponse.model_validate(data)
        assert restored.id == 5
        assert restored.result == {"data": "value"}


class TestMCPToolDef:
    def test_defaults(self) -> None:
        tool = MCPToolDef(name="read_file")
        assert tool.description == ""
        assert tool.input_schema == {}

    def test_with_schema(self) -> None:
        tool = MCPToolDef(
            name="search",
            description="Search files",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        assert tool.input_schema["properties"]["query"]["type"] == "string"

    def test_round_trip(self) -> None:
        tool = MCPToolDef(name="test", description="Test tool")
        data = tool.model_dump()
        restored = MCPToolDef.model_validate(data)
        assert restored == tool
