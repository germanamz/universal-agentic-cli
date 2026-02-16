"""Tests for TokenCounter implementations — TiktokenCounter and EstimatingCounter."""

import pytest

from uac.core.context.counter import (
    EstimatingCounter,
    TiktokenCounter,
    TokenCounter,
    _MSG_OVERHEAD,
    _REPLY_PRIMING,
)
from uac.core.interface.models import CanonicalMessage, ConversationHistory, ToolCall


class TestProtocolConformance:
    def test_tiktoken_is_token_counter(self) -> None:
        counter = TiktokenCounter("gpt-4o")
        assert isinstance(counter, TokenCounter)

    def test_estimating_is_token_counter(self) -> None:
        counter = EstimatingCounter()
        assert isinstance(counter, TokenCounter)


class TestTiktokenCounter:
    @pytest.fixture
    def counter(self) -> TiktokenCounter:
        return TiktokenCounter("gpt-4o")

    def test_count_message_text(self, counter: TiktokenCounter) -> None:
        msg = CanonicalMessage.user("Hello world")
        count = counter.count_message(msg)
        # Must be > overhead (text has tokens) and deterministic.
        assert count > _MSG_OVERHEAD
        assert count == counter.count_message(msg)  # stable

    def test_count_message_empty(self, counter: TiktokenCounter) -> None:
        msg = CanonicalMessage.user("")
        count = counter.count_message(msg)
        assert count == _MSG_OVERHEAD  # only overhead, no text tokens

    def test_count_message_with_tool_calls(self, counter: TiktokenCounter) -> None:
        tc = ToolCall(id="abc", name="calculator", arguments={"expr": "2+2"})
        msg = CanonicalMessage.assistant("", tool_calls=[tc])
        count_with_tools = counter.count_message(msg)

        msg_no_tools = CanonicalMessage.assistant("")
        count_without_tools = counter.count_message(msg_no_tools)
        assert count_with_tools > count_without_tools

    def test_count_messages_includes_priming(self, counter: TiktokenCounter) -> None:
        history = ConversationHistory(messages=[CanonicalMessage.user("Hi")])
        total = counter.count_messages(history)
        single = counter.count_message(CanonicalMessage.user("Hi"))
        assert total == single + _REPLY_PRIMING

    def test_count_messages_multiple(self, counter: TiktokenCounter) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("You are helpful."),
                CanonicalMessage.user("Hello"),
                CanonicalMessage.assistant("Hi there!"),
            ]
        )
        total = counter.count_messages(history)
        individual = sum(counter.count_message(m) for m in history)
        assert total == individual + _REPLY_PRIMING

    def test_unknown_model_falls_back_to_cl100k(self) -> None:
        # Should not raise — falls back to cl100k_base.
        counter = TiktokenCounter("totally-unknown-model-xyz")
        msg = CanonicalMessage.user("test")
        assert counter.count_message(msg) > 0


class TestEstimatingCounter:
    @pytest.fixture
    def counter(self) -> EstimatingCounter:
        return EstimatingCounter()

    def test_count_message_text(self, counter: EstimatingCounter) -> None:
        msg = CanonicalMessage.user("Hello world!!")  # 14 chars -> ~3 tokens
        count = counter.count_message(msg)
        assert count == _MSG_OVERHEAD + 14 // 4

    def test_count_message_empty(self, counter: EstimatingCounter) -> None:
        msg = CanonicalMessage.user("")
        assert counter.count_message(msg) == _MSG_OVERHEAD

    def test_count_message_with_tool_calls(self, counter: EstimatingCounter) -> None:
        tc = ToolCall(id="abc", name="calculator", arguments={"expr": "2+2"})
        msg = CanonicalMessage.assistant("", tool_calls=[tc])
        count = counter.count_message(msg)
        assert count > _MSG_OVERHEAD

    def test_count_messages_includes_priming(self, counter: EstimatingCounter) -> None:
        history = ConversationHistory(messages=[CanonicalMessage.user("Hi")])
        total = counter.count_messages(history)
        single = counter.count_message(CanonicalMessage.user("Hi"))
        assert total == single + _REPLY_PRIMING
