"""Vector offload pruning — stores old messages in a vector store and
retrieves relevant context on demand.

Provides a minimal ``InMemoryVectorStore`` stub (no real ranking). A
production deployment would swap in a proper vector DB backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from uac.core.interface.models import CanonicalMessage, ConversationHistory

if TYPE_CHECKING:
    from uac.core.context.counter import TokenCounter

_RETRIEVED_PREFIX = "[Retrieved Context]"


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for a simple text vector store."""

    async def store(self, texts: list[str]) -> None:
        """Store texts for later retrieval."""
        ...

    async def query(self, query: str, top_k: int = 3) -> list[str]:
        """Return the *top_k* most relevant stored texts for *query*."""
        ...


class InMemoryVectorStore:
    """Minimal in-memory vector store stub (returns most recent, no ranking)."""

    def __init__(self) -> None:
        self._texts: list[str] = []

    async def store(self, texts: list[str]) -> None:
        self._texts.extend(texts)

    async def query(self, query: str, top_k: int = 3) -> list[str]:
        # No real similarity — just return the most recent entries.
        return self._texts[-top_k:]


class VectorOffloadPruner:
    """Async pruner that offloads old messages to a vector store and injects
    retrieved context as a system message.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

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

        # Split: offload the older half, keep the recent half.
        split = len(non_system) // 2
        old_messages = non_system[:split]
        recent_messages = non_system[split:]

        # Store old messages.
        texts = [f"{m.role}: {m.text}" for m in old_messages]
        await self._store.store(texts)

        # Retrieve relevant context based on the most recent user message.
        query = recent_messages[-1].text if recent_messages else ""
        retrieved = await self._store.query(query, top_k=3)

        retrieved_msg = CanonicalMessage.system(
            f"{_RETRIEVED_PREFIX}\n" + "\n".join(retrieved)
        )
        return ConversationHistory(
            messages=[*system, retrieved_msg, *recent_messages]
        )
