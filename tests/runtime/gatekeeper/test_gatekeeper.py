"""Tests for Gatekeeper protocol and implementations."""

from unittest.mock import AsyncMock, patch

import pytest

from uac.runtime.errors import ApprovalTimeoutError
from uac.runtime.gatekeeper.gatekeeper import (
    AutoApproveGatekeeper,
    CLIGatekeeper,
    Gatekeeper,
)
from uac.runtime.gatekeeper.models import ApprovalRequest


class TestGatekeeperProtocol:
    def test_auto_approve_satisfies_protocol(self) -> None:
        assert isinstance(AutoApproveGatekeeper(), Gatekeeper)

    def test_cli_gatekeeper_satisfies_protocol(self) -> None:
        assert isinstance(CLIGatekeeper(), Gatekeeper)


class TestAutoApproveGatekeeper:
    async def test_always_approves(self) -> None:
        gk = AutoApproveGatekeeper()
        req = ApprovalRequest(tool_name="dangerous_tool", arguments={"x": 1})
        result = await gk.request_approval(req)
        assert result.approved is True
        assert result.reason == "auto-approved"


class TestCLIGatekeeper:
    async def test_approve_yes(self) -> None:
        gk = CLIGatekeeper()
        req = ApprovalRequest(tool_name="deploy")

        with patch.object(CLIGatekeeper, "_read_input", return_value="y"):
            with patch.object(CLIGatekeeper, "_print_summary"):
                result = await gk.request_approval(req)

        assert result.approved is True

    async def test_approve_yes_uppercase(self) -> None:
        gk = CLIGatekeeper()
        req = ApprovalRequest(tool_name="deploy")

        with patch.object(CLIGatekeeper, "_read_input", return_value="YES"):
            with patch.object(CLIGatekeeper, "_print_summary"):
                result = await gk.request_approval(req)

        assert result.approved is True

    async def test_deny_no(self) -> None:
        gk = CLIGatekeeper()
        req = ApprovalRequest(tool_name="deploy")

        with patch.object(CLIGatekeeper, "_read_input", return_value="n"):
            with patch.object(CLIGatekeeper, "_print_summary"):
                result = await gk.request_approval(req)

        assert result.approved is False
        assert result.reason == "denied by user"

    async def test_deny_empty_input(self) -> None:
        gk = CLIGatekeeper()
        req = ApprovalRequest(tool_name="deploy")

        with patch.object(CLIGatekeeper, "_read_input", return_value=""):
            with patch.object(CLIGatekeeper, "_print_summary"):
                result = await gk.request_approval(req)

        assert result.approved is False

    async def test_timeout_raises(self) -> None:
        gk = CLIGatekeeper(timeout=0.01)
        req = ApprovalRequest(tool_name="deploy")

        async def slow_input(*args, **kwargs):
            import asyncio
            await asyncio.sleep(10)
            return "y"

        with patch.object(CLIGatekeeper, "_print_summary"):
            with patch(
                "uac.runtime.gatekeeper.gatekeeper.asyncio.get_running_loop"
            ) as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(side_effect=slow_input)
                with pytest.raises(ApprovalTimeoutError) as exc_info:
                    await gk.request_approval(req)
                assert exc_info.value.tool_name == "deploy"
