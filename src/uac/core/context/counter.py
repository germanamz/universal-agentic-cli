"""Token counting — protocol and implementations for measuring token usage.

Provides accurate counting via tiktoken (for OpenAI-family models) and a
character-based estimator as a universal fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import tiktoken

if TYPE_CHECKING:
    from uac.core.interface.models import CanonicalMessage, ConversationHistory


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for counting tokens in messages."""

    def count_message(self, message: CanonicalMessage) -> int:
        """Return the token count for a single message."""
        ...

    def count_messages(self, messages: ConversationHistory) -> int:
        """Return the total token count for a conversation history."""
        ...


# ---------------------------------------------------------------------------
# Tiktoken-based counter (accurate for OpenAI models)
# ---------------------------------------------------------------------------

# Per-message overhead: every message has <|start|>{role}\n ... <|end|> framing.
_MSG_OVERHEAD = 4
# Reply priming tokens added once to the total (OpenAI convention).
_REPLY_PRIMING = 2


class TiktokenCounter:
    """Token counter using tiktoken encodings.

    Falls back to ``cl100k_base`` when the model's encoding is unknown.
    """

    def __init__(self, model: str) -> None:
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count_message(self, message: CanonicalMessage) -> int:
        """Count tokens in a single message including per-message overhead."""
        tokens = _MSG_OVERHEAD
        tokens += len(self._enc.encode(message.text))
        if message.tool_calls:
            import json

            for tc in message.tool_calls:
                tokens += len(self._enc.encode(tc.name))
                tokens += len(self._enc.encode(json.dumps(tc.arguments)))
        return tokens

    def count_messages(self, messages: ConversationHistory) -> int:
        """Count total tokens for a conversation, including reply priming."""
        total = sum(self.count_message(m) for m in messages)
        total += _REPLY_PRIMING
        return total


# ---------------------------------------------------------------------------
# Estimating counter (universal fallback)
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4


class EstimatingCounter:
    """Fallback token counter that estimates ~4 characters per token."""

    def count_message(self, message: CanonicalMessage) -> int:
        tokens = _MSG_OVERHEAD
        tokens += len(message.text) // _CHARS_PER_TOKEN
        if message.tool_calls:
            import json

            for tc in message.tool_calls:
                tokens += len(tc.name) // _CHARS_PER_TOKEN
                tokens += len(json.dumps(tc.arguments)) // _CHARS_PER_TOKEN
        return tokens

    def count_messages(self, messages: ConversationHistory) -> int:
        total = sum(self.count_message(m) for m in messages)
        total += _REPLY_PRIMING
        return total
