"""ContextManager — middleware that wraps ModelClient with token budget enforcement.

Drop-in replacement for ``ModelClient`` (same ``generate()`` signature),
automatically pruning the conversation history when it exceeds the model's
context window minus a reserve for the output.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uac.core.context.counter import TokenCounter
    from uac.core.context.pruner import ContextPruner
    from uac.core.interface.client import ModelClient
    from uac.core.interface.config import ModelConfig
    from uac.core.interface.models import CanonicalMessage, ConversationHistory

_DEFAULT_CONTEXT_WINDOW = 4096
_DEFAULT_RESERVE_TOKENS = 1024


class ContextManager:
    """Token-budget-aware wrapper around ``ModelClient``.

    Counts tokens before each ``generate()`` call and applies the configured
    pruner when the conversation exceeds the budget.

    The budget is computed as ``context_window - reserve_tokens``, where
    *context_window* comes from (in priority order):
    1. The ``context_window`` constructor parameter (if provided)
    2. ``ModelConfig.context_window`` (if set on the client's config)
    3. ``_DEFAULT_CONTEXT_WINDOW`` (4096)
    """

    def __init__(
        self,
        client: ModelClient,
        counter: TokenCounter,
        pruner: ContextPruner | None = None,
        *,
        context_window: int | None = None,
        reserve_tokens: int = _DEFAULT_RESERVE_TOKENS,
    ) -> None:
        self._client = client
        self._counter = counter
        self._pruner = pruner
        self._reserve_tokens = reserve_tokens

        # Resolve context window: explicit > config > default
        if context_window is not None:
            self._context_window = context_window
        elif client.config.context_window is not None:
            self._context_window = client.config.context_window
        else:
            self._context_window = _DEFAULT_CONTEXT_WINDOW

    @property
    def config(self) -> ModelConfig:
        """Expose the underlying model configuration (duck-typing compat)."""
        return self._client.config

    @property
    def budget(self) -> int:
        """Return the maximum input tokens (context_window - reserve)."""
        return self._context_window - self._reserve_tokens

    async def generate(
        self,
        messages: ConversationHistory,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CanonicalMessage:
        """Generate a response, pruning messages if over budget."""
        pruned = await self._maybe_prune(messages)
        return await self._client.generate(pruned, tools=tools, **kwargs)

    async def _maybe_prune(self, messages: ConversationHistory) -> ConversationHistory:
        """Prune if current token count exceeds the budget."""
        if self._pruner is None:
            return messages

        current = self._counter.count_messages(messages)
        if current <= self.budget:
            return messages

        result = self._pruner.prune(messages, self.budget, self._counter)
        # Handle both sync and async pruners.
        if inspect.isawaitable(result):
            result = await result
        return result  # type: ignore[return-value]
