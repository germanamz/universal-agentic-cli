"""Tests for SafeDispatcher."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from uac.core.interface.models import ToolCall, ToolResult
from uac.runtime.dispatcher import SafeDispatcher
from uac.runtime.errors import ApprovalDeniedError
from uac.runtime.gatekeeper.gatekeeper import AutoApproveGatekeeper
from uac.runtime.gatekeeper.models import (
    ApprovalResult,
    GatekeeperConfig,
    PolicyAction,
    ToolPolicy,
)
from uac.protocols.dispatcher import ToolDispatcher


def _make_dispatcher(
    tools: list[dict[str, Any]] | None = None,
    result_text: str = "done",
) -> ToolDispatcher:
    """Create a ToolDispatcher with a mock provider pre-registered."""
    dispatcher = ToolDispatcher()
    provider = MagicMock()
    provider.discover_tools = AsyncMock(
        return_value=tools
        or [
            {
                "type": "function",
                "function": {"name": "tool_a", "description": "", "parameters": {}},
            }
        ]
    )
    provider.execute_tool = AsyncMock(
        return_value=ToolResult.from_text(tool_call_id="", text=result_text)
    )
    # We need to register synchronously for the test setup,
    # so we'll do it in the test method itself.
    return dispatcher, provider


class TestSafeDispatcherForwarding:
    async def test_register_forwards(self) -> None:
        dispatcher, provider = _make_dispatcher()
        safe = SafeDispatcher(dispatcher, config=GatekeeperConfig(enabled=False))
        await safe.register(provider)
        assert len(safe.all_tools()) == 1

    async def test_all_tools_forwards(self) -> None:
        dispatcher, provider = _make_dispatcher()
        await dispatcher.register(provider)
        safe = SafeDispatcher(dispatcher, config=GatekeeperConfig(enabled=False))
        tools = safe.all_tools()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "tool_a"


class TestSafeDispatcherPolicy:
    async def test_allow_policy_executes(self) -> None:
        dispatcher, provider = _make_dispatcher()
        await dispatcher.register(provider)

        config = GatekeeperConfig(
            policies=[ToolPolicy(pattern="tool_a", action=PolicyAction.ALLOW)],
        )
        safe = SafeDispatcher(dispatcher, config=config)

        call = ToolCall(name="tool_a", arguments={})
        result = await safe.execute(call)
        assert result.content[0].text == "done"  # type: ignore[union-attr]

    async def test_deny_policy_raises(self) -> None:
        dispatcher, provider = _make_dispatcher()
        await dispatcher.register(provider)

        config = GatekeeperConfig(
            policies=[ToolPolicy(pattern="tool_a", action=PolicyAction.DENY)],
        )
        safe = SafeDispatcher(dispatcher, config=config)

        call = ToolCall(name="tool_a", arguments={})
        with pytest.raises(ApprovalDeniedError, match="denied by policy"):
            await safe.execute(call)

    async def test_ask_policy_with_auto_approve(self) -> None:
        dispatcher, provider = _make_dispatcher()
        await dispatcher.register(provider)

        config = GatekeeperConfig(
            policies=[ToolPolicy(pattern="tool_a", action=PolicyAction.ASK)],
        )
        gk = AutoApproveGatekeeper()
        safe = SafeDispatcher(dispatcher, gatekeeper=gk, config=config)

        call = ToolCall(name="tool_a", arguments={})
        result = await safe.execute(call)
        assert result.content[0].text == "done"  # type: ignore[union-attr]

    async def test_ask_policy_denied_by_gatekeeper(self) -> None:
        dispatcher, provider = _make_dispatcher()
        await dispatcher.register(provider)

        config = GatekeeperConfig(
            policies=[ToolPolicy(pattern="tool_a", action=PolicyAction.ASK)],
        )
        gk = MagicMock()
        gk.request_approval = AsyncMock(
            return_value=ApprovalResult(approved=False, reason="user said no")
        )
        safe = SafeDispatcher(dispatcher, gatekeeper=gk, config=config)

        call = ToolCall(name="tool_a", arguments={})
        with pytest.raises(ApprovalDeniedError, match="user said no"):
            await safe.execute(call)

    async def test_ask_with_no_gatekeeper_allows(self) -> None:
        """When policy says ASK but no gatekeeper is configured, allow."""
        dispatcher, provider = _make_dispatcher()
        await dispatcher.register(provider)

        config = GatekeeperConfig(
            policies=[ToolPolicy(pattern="tool_a", action=PolicyAction.ASK)],
        )
        safe = SafeDispatcher(dispatcher, config=config)

        call = ToolCall(name="tool_a", arguments={})
        result = await safe.execute(call)
        assert result.content[0].text == "done"  # type: ignore[union-attr]


class TestSafeDispatcherExecuteAll:
    async def test_sequential_with_gatekeeper(self) -> None:
        dispatcher, provider = _make_dispatcher()
        await dispatcher.register(provider)

        config = GatekeeperConfig(enabled=True, default_action=PolicyAction.ALLOW)
        gk = AutoApproveGatekeeper()
        safe = SafeDispatcher(dispatcher, gatekeeper=gk, config=config)

        calls = [
            ToolCall(name="tool_a", arguments={}),
            ToolCall(name="tool_a", arguments={}),
        ]
        results = await safe.execute_all(calls)
        assert len(results) == 2

    async def test_concurrent_when_disabled(self) -> None:
        dispatcher, provider = _make_dispatcher()
        await dispatcher.register(provider)

        config = GatekeeperConfig(enabled=False)
        safe = SafeDispatcher(dispatcher, config=config)

        calls = [
            ToolCall(name="tool_a", arguments={}),
            ToolCall(name="tool_a", arguments={}),
        ]
        results = await safe.execute_all(calls)
        assert len(results) == 2

    async def test_concurrent_when_no_gatekeeper(self) -> None:
        dispatcher, provider = _make_dispatcher()
        await dispatcher.register(provider)

        config = GatekeeperConfig(enabled=True, default_action=PolicyAction.ALLOW)
        safe = SafeDispatcher(dispatcher, config=config)

        calls = [
            ToolCall(name="tool_a", arguments={}),
            ToolCall(name="tool_a", arguments={}),
        ]
        results = await safe.execute_all(calls)
        assert len(results) == 2


class TestSafeDispatcherProperties:
    def test_config_property(self) -> None:
        dispatcher = ToolDispatcher()
        config = GatekeeperConfig(enabled=False)
        safe = SafeDispatcher(dispatcher, config=config)
        assert safe.config is config

    def test_sandbox_property(self) -> None:
        dispatcher = ToolDispatcher()
        safe = SafeDispatcher(dispatcher)
        assert safe.sandbox is None

    def test_sandbox_property_with_executor(self) -> None:
        dispatcher = ToolDispatcher()
        mock_sandbox = MagicMock()
        safe = SafeDispatcher(dispatcher, sandbox=mock_sandbox)
        assert safe.sandbox is mock_sandbox
