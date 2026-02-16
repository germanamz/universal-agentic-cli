"""Tests for A2AAgentNode mesh adapter."""

from unittest.mock import AsyncMock, MagicMock

from uac.core.blackboard.models import ContextSlice
from uac.core.interface.models import ToolResult
from uac.protocols.a2a.mesh_adapter import A2AAgentNode


def _make_context(belief: str = "test state", artifacts: dict | None = None) -> ContextSlice:
    return ContextSlice(
        belief_state=belief,
        trace=[],
        artifacts=artifacts or {},
        pending_tasks=[],
    )


def _make_mock_client(response_text: str = "response") -> MagicMock:
    client = MagicMock()
    client.execute_tool = AsyncMock(
        return_value=ToolResult.from_text(tool_call_id="", text=response_text)
    )
    return client


class TestA2AAgentNode:
    def test_name(self) -> None:
        client = _make_mock_client()
        node = A2AAgentNode("remote-agent", client)
        assert node.name == "remote-agent"

    async def test_step_delegates_to_client(self) -> None:
        client = _make_mock_client("delegated result")
        node = A2AAgentNode("remote", client)

        await node.step(_make_context("analyze this"))

        client.execute_tool.assert_awaited_once()
        args = client.execute_tool.call_args
        assert args[0][0] == "remote"  # skill_id defaults to name
        assert "analyze this" in args[0][1]["message"]

    async def test_step_returns_delta_with_trace(self) -> None:
        client = _make_mock_client("result")
        node = A2AAgentNode("remote", client)

        delta = await node.step(_make_context())

        assert len(delta.trace_entries) == 1
        assert delta.trace_entries[0].agent_id == "remote"
        assert delta.trace_entries[0].action == "a2a_delegate"

    async def test_step_includes_response_in_artifacts(self) -> None:
        client = _make_mock_client("the answer")
        node = A2AAgentNode("remote", client)

        delta = await node.step(_make_context())

        assert delta.artifacts["last_response"]["remote"] == "the answer"

    async def test_custom_skill_id(self) -> None:
        client = _make_mock_client()
        node = A2AAgentNode("remote", client, skill_id="custom_skill")

        await node.step(_make_context())

        args = client.execute_tool.call_args
        assert args[0][0] == "custom_skill"

    async def test_artifacts_included_in_message(self) -> None:
        client = _make_mock_client()
        node = A2AAgentNode("remote", client)

        await node.step(_make_context(artifacts={"key": "value"}))

        args = client.execute_tool.call_args
        message = args[0][1]["message"]
        assert "key" in message
        assert "value" in message
