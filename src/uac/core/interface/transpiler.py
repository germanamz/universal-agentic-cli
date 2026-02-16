"""Transpiler protocol — converts between CMS and provider-specific formats.

Each provider (OpenAI, Anthropic, Gemini) has a concrete transpiler that
implements bidirectional conversion: CMS -> provider payload and
provider response -> CanonicalMessage.
"""

from typing import Any, Protocol

from uac.core.interface.models import CanonicalMessage, ConversationHistory


class Transpiler(Protocol):
    """Protocol for provider-specific message format transpilers."""

    def to_provider(self, history: ConversationHistory) -> dict[str, Any]:
        """Convert a CMS conversation history to a provider-specific payload.

        Returns a dict suitable for passing to the provider's API (via LiteLLM
        or directly). The exact structure depends on the provider.
        """
        ...

    def from_provider(self, response: dict[str, Any]) -> CanonicalMessage:
        """Convert a provider's raw response into a CanonicalMessage.

        Extracts text content, tool calls, and metadata from the provider's
        response format.
        """
        ...
