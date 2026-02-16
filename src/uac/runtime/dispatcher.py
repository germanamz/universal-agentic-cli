"""SafeDispatcher — wraps ToolDispatcher with gatekeeper and sandbox checks.

Same wrapper pattern as :class:`~uac.core.context.manager.ContextManager`
wrapping :class:`~uac.core.interface.client.ModelClient`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from uac.runtime.errors import ApprovalDeniedError
from uac.runtime.gatekeeper.models import ApprovalRequest, GatekeeperConfig, PolicyAction
from uac.runtime.gatekeeper.policy import PolicyEngine

if TYPE_CHECKING:
    from uac.core.interface.models import ToolCall, ToolResult
    from uac.protocols.dispatcher import ToolDispatcher
    from uac.protocols.provider import ToolProvider
    from uac.runtime.gatekeeper.gatekeeper import Gatekeeper
    from uac.runtime.sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)


class SafeDispatcher:
    """Token-safe, gatekeeper-aware wrapper around :class:`ToolDispatcher`.

    Intercepts ``execute()`` calls with:
    1. **Policy evaluation** — check whether the tool is allowed, denied, or
       requires interactive approval.
    2. **Gatekeeper prompt** — if the policy says "ask", delegate to the
       configured :class:`Gatekeeper` implementation.
    3. **Delegation** — forward the (approved) call to the underlying
       :class:`ToolDispatcher`.

    ``register()`` and ``all_tools()`` are forwarded directly.
    """

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        *,
        gatekeeper: Gatekeeper | None = None,
        sandbox: SandboxExecutor | None = None,
        config: GatekeeperConfig | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._gatekeeper = gatekeeper
        self._sandbox = sandbox
        self._config = config or GatekeeperConfig()
        self._policy_engine = PolicyEngine(self._config)

    @property
    def config(self) -> GatekeeperConfig:
        return self._config

    @property
    def sandbox(self) -> SandboxExecutor | None:
        return self._sandbox

    async def register(self, provider: ToolProvider) -> None:
        """Forward registration to the underlying dispatcher."""
        await self._dispatcher.register(provider)

    def all_tools(self) -> list[dict[str, Any]]:
        """Forward tool listing to the underlying dispatcher."""
        return self._dispatcher.all_tools()

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Evaluate policy, optionally prompt for approval, then execute."""
        action = self._policy_engine.evaluate(tool_call.name)

        if action == PolicyAction.DENY:
            raise ApprovalDeniedError(tool_call.name, reason="denied by policy")

        if action == PolicyAction.ASK:
            await self._request_approval(tool_call)

        return await self._dispatcher.execute(tool_call)

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls.

        Runs sequentially when the gatekeeper is active (user can't answer
        N prompts concurrently in a CLI), or concurrently when disabled.
        """
        if self._config.enabled and self._gatekeeper is not None:
            # Sequential — user must approve one at a time.
            results: list[ToolResult] = []
            for tc in tool_calls:
                results.append(await self.execute(tc))
            return results

        # Concurrent — gatekeeper is disabled or absent.
        return list(await asyncio.gather(*[self.execute(tc) for tc in tool_calls]))

    async def _request_approval(self, tool_call: ToolCall) -> None:
        """Ask the gatekeeper for approval; raise on denial."""
        if self._gatekeeper is None:
            logger.warning(
                "Policy says 'ask' for tool %s but no gatekeeper is configured — allowing.",
                tool_call.name,
            )
            return

        request = ApprovalRequest(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
        )
        result = await self._gatekeeper.request_approval(request)

        if not result.approved:
            raise ApprovalDeniedError(tool_call.name, reason=result.reason)
