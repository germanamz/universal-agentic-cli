"""Tests for SlidingWindowPruner."""

import pytest

from uac.core.context.counter import EstimatingCounter
from uac.core.context.pruner import ContextPruner, SlidingWindowPruner
from uac.core.interface.models import CanonicalMessage, ConversationHistory


class TestSlidingWindowPruner:
    @pytest.fixture
    def counter(self) -> EstimatingCounter:
        return EstimatingCounter()

    @pytest.fixture
    def pruner(self) -> SlidingWindowPruner:
        return SlidingWindowPruner(min_recent=2)

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(SlidingWindowPruner(), ContextPruner)

    def test_preserves_system_messages(
        self, pruner: SlidingWindowPruner, counter: EstimatingCounter
    ) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("System prompt"),
                CanonicalMessage.user("msg1"),
                CanonicalMessage.user("msg2"),
                CanonicalMessage.user("msg3"),
            ]
        )
        result = pruner.prune(history, max_tokens=50, counter=counter)
        system_msgs = [m for m in result if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].text == "System prompt"

    def test_drops_oldest_non_system(
        self, pruner: SlidingWindowPruner, counter: EstimatingCounter
    ) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("sys"),
                CanonicalMessage.user("old message one"),
                CanonicalMessage.user("old message two"),
                CanonicalMessage.user("recent one"),
                CanonicalMessage.user("recent two"),
            ]
        )
        # Set a tight budget that can't fit all messages.
        result = pruner.prune(history, max_tokens=40, counter=counter)
        texts = [m.text for m in result if m.role != "system"]
        # Should have dropped the oldest first.
        assert "recent two" in texts
        assert "recent one" in texts

    def test_respects_min_recent(self, counter: EstimatingCounter) -> None:
        pruner = SlidingWindowPruner(min_recent=3)
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("sys"),
                CanonicalMessage.user("a"),
                CanonicalMessage.user("b"),
                CanonicalMessage.user("c"),
                CanonicalMessage.user("d"),
            ]
        )
        # Very tight budget — but min_recent=3 means we keep at least 3.
        result = pruner.prune(history, max_tokens=10, counter=counter)
        non_system = [m for m in result if m.role != "system"]
        assert len(non_system) >= 3

    def test_immutable_original(
        self, pruner: SlidingWindowPruner, counter: EstimatingCounter
    ) -> None:
        messages = [
            CanonicalMessage.system("sys"),
            CanonicalMessage.user("a"),
            CanonicalMessage.user("b"),
            CanonicalMessage.user("c"),
        ]
        history = ConversationHistory(messages=list(messages))
        pruner.prune(history, max_tokens=30, counter=counter)
        # Original must be unchanged.
        assert len(history.messages) == 4

    def test_no_prune_when_under_budget(
        self, pruner: SlidingWindowPruner, counter: EstimatingCounter
    ) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("sys"),
                CanonicalMessage.user("hi"),
            ]
        )
        result = pruner.prune(history, max_tokens=10000, counter=counter)
        assert len(result.messages) == 2

    def test_no_non_system_messages(
        self, pruner: SlidingWindowPruner, counter: EstimatingCounter
    ) -> None:
        history = ConversationHistory(
            messages=[CanonicalMessage.system("sys")]
        )
        result = pruner.prune(history, max_tokens=10, counter=counter)
        assert len(result.messages) == 1
