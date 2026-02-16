"""Context pruning — protocol and sliding-window implementation.

Pruners reduce conversation history to fit within a token budget while
preserving the most important messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from uac.core.interface.models import ConversationHistory

if TYPE_CHECKING:
    from uac.core.context.counter import TokenCounter


@runtime_checkable
class ContextPruner(Protocol):
    """Protocol for pruning a conversation history to fit a token budget."""

    def prune(
        self,
        messages: ConversationHistory,
        max_tokens: int,
        counter: TokenCounter,
    ) -> ConversationHistory:
        """Return a pruned copy of *messages* that fits within *max_tokens*."""
        ...


class SlidingWindowPruner:
    """Drops the oldest non-system messages, preserving system messages and
    the *min_recent* most recent non-system messages.

    The original ``ConversationHistory`` is never mutated.
    """

    def __init__(self, min_recent: int = 2) -> None:
        self._min_recent = min_recent

    def prune(
        self,
        messages: ConversationHistory,
        max_tokens: int,
        counter: TokenCounter,
    ) -> ConversationHistory:
        system = messages.system_messages
        non_system = messages.non_system_messages

        # Always keep at least min_recent non-system messages.
        keep_min = max(0, min(self._min_recent, len(non_system)))

        # Start from keeping all non-system, drop from the front until we fit.
        for start in range(len(non_system) - keep_min + 1):
            candidate = ConversationHistory(
                messages=list(system) + list(non_system[start:])
            )
            if counter.count_messages(candidate) <= max_tokens:
                return candidate

        # Even min_recent doesn't fit — return system + min_recent anyway.
        return ConversationHistory(
            messages=list(system) + list(non_system[-keep_min:]) if keep_min else list(system)
        )
