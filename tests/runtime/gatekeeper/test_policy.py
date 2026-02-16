"""Tests for PolicyEngine."""

from uac.runtime.gatekeeper.models import GatekeeperConfig, PolicyAction, ToolPolicy
from uac.runtime.gatekeeper.policy import PolicyEngine


class TestPolicyEngine:
    def test_disabled_always_allows(self) -> None:
        engine = PolicyEngine(GatekeeperConfig(enabled=False))
        assert engine.evaluate("anything") == PolicyAction.ALLOW

    def test_safe_tools_fast_path(self) -> None:
        engine = PolicyEngine(
            GatekeeperConfig(
                safe_tools=["read_file", "list_dir"],
                default_action=PolicyAction.DENY,
            )
        )
        assert engine.evaluate("read_file") == PolicyAction.ALLOW
        assert engine.evaluate("list_dir") == PolicyAction.ALLOW
        assert engine.evaluate("delete_file") == PolicyAction.DENY

    def test_first_match_wins(self) -> None:
        engine = PolicyEngine(
            GatekeeperConfig(
                policies=[
                    ToolPolicy(pattern="deploy_*", action=PolicyAction.ASK),
                    ToolPolicy(pattern="deploy_staging", action=PolicyAction.ALLOW),
                ],
                default_action=PolicyAction.DENY,
            )
        )
        # First rule matches, second is never reached
        assert engine.evaluate("deploy_staging") == PolicyAction.ASK

    def test_glob_pattern_matching(self) -> None:
        engine = PolicyEngine(
            GatekeeperConfig(
                policies=[
                    ToolPolicy(pattern="file_*", action=PolicyAction.ALLOW),
                    ToolPolicy(pattern="db_*", action=PolicyAction.DENY),
                ],
                default_action=PolicyAction.ASK,
            )
        )
        assert engine.evaluate("file_read") == PolicyAction.ALLOW
        assert engine.evaluate("file_write") == PolicyAction.ALLOW
        assert engine.evaluate("db_drop") == PolicyAction.DENY
        assert engine.evaluate("unknown_tool") == PolicyAction.ASK

    def test_wildcard_catches_all(self) -> None:
        engine = PolicyEngine(
            GatekeeperConfig(
                policies=[
                    ToolPolicy(pattern="safe_*", action=PolicyAction.ALLOW),
                    ToolPolicy(pattern="*", action=PolicyAction.DENY),
                ],
                default_action=PolicyAction.ASK,
            )
        )
        assert engine.evaluate("safe_read") == PolicyAction.ALLOW
        assert engine.evaluate("anything_else") == PolicyAction.DENY

    def test_default_action_fallback(self) -> None:
        engine = PolicyEngine(
            GatekeeperConfig(
                policies=[],
                default_action=PolicyAction.ALLOW,
            )
        )
        assert engine.evaluate("any_tool") == PolicyAction.ALLOW

    def test_exact_match(self) -> None:
        engine = PolicyEngine(
            GatekeeperConfig(
                policies=[
                    ToolPolicy(pattern="specific_tool", action=PolicyAction.DENY),
                ],
                default_action=PolicyAction.ALLOW,
            )
        )
        assert engine.evaluate("specific_tool") == PolicyAction.DENY
        assert engine.evaluate("specific_tool_extra") == PolicyAction.ALLOW

    def test_safe_tools_take_priority_over_policies(self) -> None:
        engine = PolicyEngine(
            GatekeeperConfig(
                safe_tools=["read_file"],
                policies=[
                    ToolPolicy(pattern="read_*", action=PolicyAction.DENY),
                ],
                default_action=PolicyAction.DENY,
            )
        )
        # safe_tools checked before policies
        assert engine.evaluate("read_file") == PolicyAction.ALLOW
        # But other read_* tools hit the policy
        assert engine.evaluate("read_dir") == PolicyAction.DENY

    def test_config_property(self) -> None:
        cfg = GatekeeperConfig()
        engine = PolicyEngine(cfg)
        assert engine.config is cfg
