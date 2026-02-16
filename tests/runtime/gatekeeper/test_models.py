"""Tests for gatekeeper data models."""

from uac.runtime.gatekeeper.models import (
    ApprovalRequest,
    ApprovalResult,
    GatekeeperConfig,
    PolicyAction,
    ToolPolicy,
)


class TestPolicyAction:
    def test_values(self) -> None:
        assert PolicyAction.ALLOW == "allow"
        assert PolicyAction.DENY == "deny"
        assert PolicyAction.ASK == "ask"


class TestToolPolicy:
    def test_minimal(self) -> None:
        policy = ToolPolicy(pattern="file_*", action=PolicyAction.DENY)
        assert policy.pattern == "file_*"
        assert policy.action == PolicyAction.DENY
        assert policy.reason == ""

    def test_with_reason(self) -> None:
        policy = ToolPolicy(
            pattern="rm_*",
            action=PolicyAction.ASK,
            reason="Destructive operation",
        )
        assert policy.reason == "Destructive operation"


class TestGatekeeperConfig:
    def test_defaults(self) -> None:
        cfg = GatekeeperConfig()
        assert cfg.enabled is True
        assert cfg.default_action == PolicyAction.ASK
        assert cfg.safe_tools == []
        assert cfg.policies == []
        assert cfg.approval_timeout == 300.0

    def test_custom(self) -> None:
        cfg = GatekeeperConfig(
            enabled=False,
            default_action=PolicyAction.ALLOW,
            safe_tools=["read_file"],
            policies=[
                ToolPolicy(pattern="delete_*", action=PolicyAction.DENY),
            ],
        )
        assert cfg.enabled is False
        assert len(cfg.policies) == 1


class TestApprovalRequest:
    def test_minimal(self) -> None:
        req = ApprovalRequest(tool_name="deploy")
        assert req.tool_name == "deploy"
        assert req.arguments == {}

    def test_with_arguments(self) -> None:
        req = ApprovalRequest(
            tool_name="deploy",
            arguments={"env": "production"},
            reason="production deployment",
        )
        assert req.arguments == {"env": "production"}
        assert req.reason == "production deployment"


class TestApprovalResult:
    def test_approved(self) -> None:
        result = ApprovalResult(approved=True)
        assert result.approved is True
        assert result.reason == ""

    def test_denied(self) -> None:
        result = ApprovalResult(approved=False, reason="too risky")
        assert result.approved is False
        assert result.reason == "too risky"
