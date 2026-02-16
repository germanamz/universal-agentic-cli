"""Gatekeeper protocol and implementations.

- ``Gatekeeper`` — runtime-checkable protocol for approval gates.
- ``CLIGatekeeper`` — prompts the user via stdin/stdout.
- ``AutoApproveGatekeeper`` — always approves (for testing/CI).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from uac.runtime.errors import ApprovalTimeoutError

if TYPE_CHECKING:
    from uac.runtime.gatekeeper.models import ApprovalRequest, ApprovalResult

logger = logging.getLogger(__name__)


@runtime_checkable
class Gatekeeper(Protocol):
    """Decides whether a tool execution should proceed."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        """Ask for approval and return the decision."""
        ...


class AutoApproveGatekeeper:
    """Always approves — suitable for tests and CI pipelines.

    Satisfies the :class:`Gatekeeper` protocol.
    """

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        from uac.runtime.gatekeeper.models import ApprovalResult

        logger.debug("AutoApproveGatekeeper: auto-approving %s", request.tool_name)
        return ApprovalResult(approved=True, reason="auto-approved")


class CLIGatekeeper:
    """Prompts the user at the terminal for approval.

    Satisfies the :class:`Gatekeeper` protocol.

    Uses ``loop.run_in_executor(None, input)`` to read from stdin without
    blocking the event loop.  Auto-denies if no response within *timeout*.
    """

    def __init__(self, *, timeout: float = 300.0) -> None:
        self._timeout = timeout

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        from uac.runtime.gatekeeper.models import ApprovalResult

        self._print_summary(request)

        loop = asyncio.get_running_loop()
        try:
            answer: str = await asyncio.wait_for(
                loop.run_in_executor(None, self._read_input),
                timeout=self._timeout,
            )
        except TimeoutError:
            raise ApprovalTimeoutError(request.tool_name, self._timeout)

        approved = answer.strip().lower() in ("y", "yes")
        reason = "" if approved else "denied by user"
        return ApprovalResult(approved=approved, reason=reason)

    @staticmethod
    def _print_summary(request: ApprovalRequest) -> None:
        """Print a human-readable action summary to stdout."""
        sep = "-" * 60
        sys.stdout.write(f"\n{sep}\n")
        sys.stdout.write(f"  Tool:      {request.tool_name}\n")
        if request.arguments:
            sys.stdout.write(f"  Arguments: {request.arguments}\n")
        if request.reason:
            sys.stdout.write(f"  Reason:    {request.reason}\n")
        sys.stdout.write(f"{sep}\n")
        sys.stdout.write("  Approve? [y/N]: ")
        sys.stdout.flush()

    @staticmethod
    def _read_input() -> str:
        """Blocking read from stdin (run in executor)."""
        return input()
