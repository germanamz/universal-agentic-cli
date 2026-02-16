"""PolicyEngine — evaluates tool names against an ordered policy list.

Pure logic, no I/O.  The engine checks ``safe_tools`` first (fast-path),
then walks the ``policies`` list (first-match-wins), and finally falls back
to ``default_action``.
"""

from __future__ import annotations

import fnmatch

from uac.runtime.gatekeeper.models import GatekeeperConfig, PolicyAction, ToolPolicy


class PolicyEngine:
    """Evaluate a tool name against a :class:`GatekeeperConfig`."""

    def __init__(self, config: GatekeeperConfig) -> None:
        self._config = config

    @property
    def config(self) -> GatekeeperConfig:
        return self._config

    def evaluate(self, tool_name: str) -> PolicyAction:
        """Return the action for *tool_name*.

        Resolution order:
        1. ``safe_tools`` — if the tool name is listed, return ``ALLOW``.
        2. ``policies`` — first matching rule wins.
        3. ``default_action`` — fallback.
        """
        if not self._config.enabled:
            return PolicyAction.ALLOW

        # Fast-path: safe tools are always allowed.
        if tool_name in self._config.safe_tools:
            return PolicyAction.ALLOW

        # Walk ordered policies, first match wins.
        for policy in self._config.policies:
            if self._matches(policy, tool_name):
                return policy.action

        return self._config.default_action

    @staticmethod
    def _matches(policy: ToolPolicy, tool_name: str) -> bool:
        """Check whether *policy.pattern* matches *tool_name*.

        Supports exact match and Unix-style glob patterns (``fnmatch``).
        """
        return fnmatch.fnmatch(tool_name, policy.pattern)
