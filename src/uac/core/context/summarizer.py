"""Summarizer pruning — compresses old messages via a cheap LLM call.

Uses a separate ``ModelClient`` (typically a fast/cheap model) to generate
a summary of older messages, replacing them with a single system message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from uac.core.interface.models import CanonicalMessage, ConversationHistory

if TYPE_CHECKING:
    from uac.core.context.counter import TokenCounter
    from uac.core.interface.client import ModelClient

_SUMMARY_PREFIX = "[Conversation Summary]"

_SUMMARIZE_PROMPT = (
    "Summarize the following conversation concisely, preserving key facts, "
    "decisions, and context needed for the assistant to continue helpfully. "
    "Respond with only the summary, no preamble."
)


class SummarizerPruner:
    """Async pruner that summarizes old messages via a cheap model call.

    The summary is injected as a system message prefixed with
    ``[Conversation Summary]``.
    """

    def __init__(self, summarizer_client: ModelClient) -> None:
        self._client = summarizer_client

    async def prune(
        self,
        messages: ConversationHistory,
        max_tokens: int,
        counter: TokenCounter,
    ) -> ConversationHistory:
        if counter.count_messages(messages) <= max_tokens:
            return messages

        system = messages.system_messages
        non_system = messages.non_system_messages

        if len(non_system) <= 2:
            return messages

        # Split: summarize the older half, keep the recent half.
        split = len(non_system) // 2
        old_messages = non_system[:split]
        recent_messages = non_system[split:]

        # Build a prompt asking the summarizer to compress the old messages.
        summary_history = ConversationHistory(
            messages=[
                CanonicalMessage.system(_SUMMARIZE_PROMPT),
                CanonicalMessage.user(
                    "\n".join(f"{m.role}: {m.text}" for m in old_messages)
                ),
            ]
        )

        response = await self._client.generate(summary_history)

        summary_msg = CanonicalMessage.system(f"{_SUMMARY_PREFIX} {response.text}")
        return ConversationHistory(
            messages=[*system, summary_msg, *recent_messages]
        )
