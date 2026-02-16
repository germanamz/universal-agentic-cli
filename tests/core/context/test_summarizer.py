"""Tests for SummarizerPruner — mock ModelClient, verify summary injection."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from uac.core.context.counter import EstimatingCounter
from uac.core.context.summarizer import _SUMMARY_PREFIX, SummarizerPruner
from uac.core.interface.models import CanonicalMessage, ConversationHistory


def _make_mock_client(summary_text: str = "Summary of old messages.") -> MagicMock:
    """Create a mock ModelClient that returns a canned summary."""
    client = MagicMock()
    response = CanonicalMessage.assistant(summary_text)
    client.generate = AsyncMock(return_value=response)
    return client


class TestSummarizerPruner:
    @pytest.fixture
    def counter(self) -> EstimatingCounter:
        return EstimatingCounter()

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        return _make_mock_client()

    @pytest.fixture
    def pruner(self, mock_client: MagicMock) -> SummarizerPruner:
        return SummarizerPruner(summarizer_client=mock_client)

    async def test_returns_unchanged_when_under_budget(
        self, pruner: SummarizerPruner, counter: EstimatingCounter
    ) -> None:
        history = ConversationHistory(
            messages=[CanonicalMessage.user("Hello")]
        )
        result = await pruner.prune(history, max_tokens=10000, counter=counter)
        assert len(result.messages) == 1

    async def test_injects_summary_message(
        self, pruner: SummarizerPruner, counter: EstimatingCounter, mock_client: MagicMock
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

        # Should contain a summary system message.
        summary_msgs = [m for m in result if m.text.startswith(_SUMMARY_PREFIX)]
        assert len(summary_msgs) == 1
        assert "Summary of old messages." in summary_msgs[0].text

    async def test_calls_summarizer_client(
        self, pruner: SummarizerPruner, counter: EstimatingCounter, mock_client: MagicMock
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
        await pruner.prune(history, max_tokens=10, counter=counter)
        mock_client.generate.assert_awaited_once()

    async def test_preserves_system_messages(
        self, pruner: SummarizerPruner, counter: EstimatingCounter
    ) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("original system"),
                CanonicalMessage.user("msg1"),
                CanonicalMessage.assistant("resp1"),
                CanonicalMessage.user("msg2"),
                CanonicalMessage.assistant("resp2"),
            ]
        )
        result = await pruner.prune(history, max_tokens=10, counter=counter)
        system_msgs = [m for m in result if m.role == "system"]
        # Original system + summary system message.
        assert any(m.text == "original system" for m in system_msgs)

    async def test_keeps_recent_messages(
        self, pruner: SummarizerPruner, counter: EstimatingCounter
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
