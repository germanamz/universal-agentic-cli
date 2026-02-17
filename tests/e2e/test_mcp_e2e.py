"""E2E tests for MCP tool discovery, dispatch, and orchestration integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.e2e.conftest import make_mock_litellm_response
from uac.core.interface.client import ModelClient
from uac.core.interface.config import ModelConfig
from uac.core.orchestration.models import AgentManifest, MCPServerRef
from uac.core.orchestration.primitives import AgentNode
from uac.core.orchestration.topologies.pipeline import PipelineOrchestrator
from uac.protocols.dispatcher import ToolDispatcher
from uac.protocols.mcp.client import MCPClient


def _make_mock_transport(tool_defs: list[dict[str, Any]]) -> MagicMock:
    """Create a mock MCP transport that responds to initialize, tools/list, and tools/call."""
    responses: list[dict[str, Any]] = []

    # initialize response
    responses.append({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "mock-mcp", "version": "1.0"},
        },
    })

    # tools/list response
    responses.append({
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"tools": tool_defs},
    })

    transport = MagicMock()
    transport.connect = AsyncMock()
    transport.close = AsyncMock()
    transport.send = AsyncMock()

    call_count = {"n": 0}

    async def _receive() -> dict[str, Any]:
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(responses):
            return responses[idx]
        # Default: tools/call response
        return {
            "jsonrpc": "2.0",
            "id": idx + 1,
            "result": {
                "content": [{"type": "text", "text": "Tool result output."}],
            },
        }

    transport.receive = AsyncMock(side_effect=_receive)
    return transport


_SAMPLE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from disk.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
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


class TestMCPToolDiscovery:
    @pytest.mark.asyncio
    async def test_tool_discovery_and_registration(self) -> None:
        """MCPClient discovers tools and registers them on ToolDispatcher."""
        ref = MCPServerRef(name="fs", command="echo test", transport="stdio")
        transport = _make_mock_transport(_SAMPLE_TOOLS)

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            async with MCPClient(ref) as client:
                dispatcher = ToolDispatcher()
                await dispatcher.register(client)

                all_tools = dispatcher.all_tools()

        assert len(all_tools) == 2
        tool_names = {t["function"]["name"] for t in all_tools}
        assert tool_names == {"read_file", "write_file"}

    @pytest.mark.asyncio
    async def test_tool_execution_via_dispatcher(self) -> None:
        """ToolDispatcher routes execution to the correct MCP provider."""
        from uac.core.interface.models import ToolCall

        ref = MCPServerRef(name="fs", command="echo test", transport="stdio")
        transport = _make_mock_transport(_SAMPLE_TOOLS)

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            async with MCPClient(ref) as client:
                dispatcher = ToolDispatcher()
                await dispatcher.register(client)

                call = ToolCall(name="read_file", arguments={"path": "/tmp/test.txt"})
                result = await dispatcher.execute(call)

        assert result.tool_call_id == ""
        # Content should contain the mock response text
        content_text = "".join(
            p.text for p in result.content if hasattr(p, "text")
        )
        assert "Tool result output" in content_text


class TestMCPWithOrchestration:
    @pytest.mark.asyncio
    async def test_agent_receives_mcp_tools_in_call(self) -> None:
        """MCP-discovered tools are passed to the agent's LLM call."""
        ref = MCPServerRef(name="fs", command="echo test", transport="stdio")
        transport = _make_mock_transport(_SAMPLE_TOOLS)
        call_kwargs: list[dict[str, Any]] = []

        async def capture_call(**kwargs: Any) -> MagicMock:
            call_kwargs.append(dict(kwargs))
            return make_mock_litellm_response(content="Used tool.")

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            async with MCPClient(ref) as client:
                dispatcher = ToolDispatcher()
                await dispatcher.register(client)
                tools = dispatcher.all_tools()

            manifest = AgentManifest(name="agent", system_prompt_template="You are agent.")
            config = ModelConfig(model="openai/gpt-4o")
            mc = ModelClient(config)
            node = AgentNode(manifest=manifest, client=mc, tools=tools)

            with patch("uac.core.interface.client.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(side_effect=capture_call)

                orch = PipelineOrchestrator(agents={"agent": node}, order=["agent"])
                await orch.run("Use tools")

        assert len(call_kwargs) == 1
        passed_tools = call_kwargs[0].get("tools", [])
        assert len(passed_tools) == 2
        names = {t["function"]["name"] for t in passed_tools}
        assert names == {"read_file", "write_file"}

    @pytest.mark.asyncio
    async def test_multi_server_tool_merging(self) -> None:
        """Tools from multiple MCP servers are merged in the dispatcher."""
        tools_a = [{"name": "tool_a", "description": "Tool A.", "inputSchema": {}}]
        tools_b = [{"name": "tool_b", "description": "Tool B.", "inputSchema": {}}]

        ref_a = MCPServerRef(name="server_a", command="echo a", transport="stdio")
        ref_b = MCPServerRef(name="server_b", command="echo b", transport="stdio")

        transport_a = _make_mock_transport(tools_a)
        transport_b = _make_mock_transport(tools_b)

        transports = iter([transport_a, transport_b])

        with patch.object(MCPClient, "_create_transport", side_effect=lambda: next(transports)):
            dispatcher = ToolDispatcher()

            async with MCPClient(ref_a) as client_a:
                await dispatcher.register(client_a)

            async with MCPClient(ref_b) as client_b:
                await dispatcher.register(client_b)

        all_tools = dispatcher.all_tools()
        assert len(all_tools) == 2
        names = {t["function"]["name"] for t in all_tools}
        assert names == {"tool_a", "tool_b"}


class TestMCPWithPromptedStrategy:
    @pytest.mark.asyncio
    async def test_react_prompt_mentions_mcp_tool_names(self) -> None:
        """When using PromptedStrategy, the ReAct prompt includes MCP tool names."""
        ref = MCPServerRef(name="fs", command="echo test", transport="stdio")
        transport = _make_mock_transport(_SAMPLE_TOOLS)
        call_kwargs: list[dict[str, Any]] = []

        async def capture_call(**kwargs: Any) -> MagicMock:
            call_kwargs.append(dict(kwargs))
            return make_mock_litellm_response(
                content="Thought: I need to read a file.\nFinal Answer: Done.",
                model="ollama/mistral-7b",
            )

        with patch.object(MCPClient, "_create_transport", return_value=transport):
            async with MCPClient(ref) as client:
                dispatcher = ToolDispatcher()
                await dispatcher.register(client)
                tools = dispatcher.all_tools()

            manifest = AgentManifest(name="agent", system_prompt_template="You are agent.")
            config = ModelConfig(model="ollama/mistral-7b")
            mc = ModelClient(config)
            node = AgentNode(manifest=manifest, client=mc, tools=tools)

            with patch("uac.core.interface.client.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(side_effect=capture_call)

                orch = PipelineOrchestrator(agents={"agent": node}, order=["agent"])
                await orch.run("Read a file")

        assert len(call_kwargs) == 1
        messages = call_kwargs[0].get("messages", [])
        system_texts = [m["content"] for m in messages if m.get("role") == "system"]
        combined = " ".join(system_texts)
        assert "read_file" in combined
        assert "write_file" in combined
        # Tools should be stripped (PromptedStrategy)
        assert "tools" not in call_kwargs[0] or call_kwargs[0].get("tools") is None
