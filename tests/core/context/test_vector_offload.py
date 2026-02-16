"""Tests for InMemoryVectorStore and VectorOffloadPruner."""

import pytest

from uac.core.context.counter import EstimatingCounter
from uac.core.context.vector_offload import (
    InMemoryVectorStore,
    VectorOffloadPruner,
    VectorStore,
    _RETRIEVED_PREFIX,
)
from uac.core.interface.models import CanonicalMessage, ConversationHistory


class TestInMemoryVectorStore:
    async def test_conforms_to_protocol(self) -> None:
        assert isinstance(InMemoryVectorStore(), VectorStore)

    async def test_store_and_query(self) -> None:
        store = InMemoryVectorStore()
        await store.store(["alpha", "beta", "gamma"])
        results = await store.query("anything", top_k=2)
        # Returns most recent.
        assert results == ["beta", "gamma"]

    async def test_query_empty(self) -> None:
        store = InMemoryVectorStore()
        results = await store.query("query")
        assert results == []

    async def test_query_fewer_than_top_k(self) -> None:
        store = InMemoryVectorStore()
        await store.store(["only one"])
        results = await store.query("query", top_k=5)
        assert results == ["only one"]

    async def test_multiple_stores_accumulate(self) -> None:
        store = InMemoryVectorStore()
        await store.store(["a", "b"])
        await store.store(["c"])
        results = await store.query("query", top_k=3)
        assert results == ["a", "b", "c"]


class TestVectorOffloadPruner:
    @pytest.fixture
    def counter(self) -> EstimatingCounter:
        return EstimatingCounter()

    @pytest.fixture
    def store(self) -> InMemoryVectorStore:
        return InMemoryVectorStore()

    @pytest.fixture
    def pruner(self, store: InMemoryVectorStore) -> VectorOffloadPruner:
        return VectorOffloadPruner(vector_store=store)

    async def test_returns_unchanged_when_under_budget(
        self, pruner: VectorOffloadPruner, counter: EstimatingCounter
    ) -> None:
        history = ConversationHistory(
            messages=[CanonicalMessage.user("Hello")]
        )
        result = await pruner.prune(history, max_tokens=10000, counter=counter)
        assert len(result.messages) == 1

    async def test_injects_retrieved_context(
        self, pruner: VectorOffloadPruner, counter: EstimatingCounter
    ) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("sys"),
                CanonicalMessage.user("msg1"),
                CanonicalMessage.assistant("resp1"),
                CanonicalMessage.user("msg2"),
                CanonicalMessage.assistant("resp2"),
            ]
        )
        result = await pruner.prune(history, max_tokens=10, counter=counter)
        retrieved_msgs = [m for m in result if m.text.startswith(_RETRIEVED_PREFIX)]
        assert len(retrieved_msgs) == 1

    async def test_stores_old_messages(
        self,
        pruner: VectorOffloadPruner,
        store: InMemoryVectorStore,
        counter: EstimatingCounter,
    ) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("sys"),
                CanonicalMessage.user("old1"),
                CanonicalMessage.assistant("old2"),
                CanonicalMessage.user("recent1"),
                CanonicalMessage.assistant("recent2"),
            ]
        )
        await pruner.prune(history, max_tokens=10, counter=counter)
        # The store should have received the old messages.
        assert len(store._texts) > 0

    async def test_keeps_recent_messages(
        self, pruner: VectorOffloadPruner, counter: EstimatingCounter
    ) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("sys"),
                CanonicalMessage.user("old1"),
                CanonicalMessage.assistant("old2"),
                CanonicalMessage.user("recent1"),
                CanonicalMessage.assistant("recent2"),
            ]
        )
        result = await pruner.prune(history, max_tokens=10, counter=counter)
        texts = [m.text for m in result if m.role != "system"]
        assert "recent1" in texts
        assert "recent2" in texts
