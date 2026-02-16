"""Tests for ContextManager — pruning triggers, passthrough, config exposure."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from uac.core.context.counter import EstimatingCounter
from uac.core.context.manager import ContextManager, _DEFAULT_CONTEXT_WINDOW, _DEFAULT_RESERVE_TOKENS
from uac.core.context.pruner import SlidingWindowPruner
from uac.core.interface.config import ModelConfig
from uac.core.interface.models import CanonicalMessage, ConversationHistory


def _make_mock_model_client(
    config: ModelConfig | None = None,
) -> MagicMock:
    """Create a mock ModelClient."""
    client = MagicMock()
    client.config = config or ModelConfig(model="openai/gpt-4o")
    response = CanonicalMessage.assistant("Hello!")
    client.generate = AsyncMock(return_value=response)
    return client


class TestContextManager:
    @pytest.fixture
    def config(self) -> ModelConfig:
        return ModelConfig(model="openai/gpt-4o", context_window=8192)

    @pytest.fixture
    def counter(self) -> EstimatingCounter:
        return EstimatingCounter()

    @pytest.fixture
    def mock_client(self, config: ModelConfig) -> MagicMock:
        return _make_mock_model_client(config)

    @pytest.fixture
    def history(self) -> ConversationHistory:
        return ConversationHistory(
            messages=[
                CanonicalMessage.system("You are helpful."),
                CanonicalMessage.user("Hello"),
            ]
        )

    def test_exposes_config(self, mock_client: MagicMock, counter: EstimatingCounter) -> None:
        manager = ContextManager(mock_client, counter)
        assert manager.config is mock_client.config

    def test_context_window_explicit(
        self, mock_client: MagicMock, counter: EstimatingCounter
    ) -> None:
        manager = ContextManager(mock_client, counter, context_window=16384)
        assert manager.budget == 16384 - _DEFAULT_RESERVE_TOKENS

    def test_context_window_from_config(
        self, mock_client: MagicMock, counter: EstimatingCounter
    ) -> None:
        manager = ContextManager(mock_client, counter)
        # config.context_window is 8192
        assert manager.budget == 8192 - _DEFAULT_RESERVE_TOKENS

    def test_context_window_default(self, counter: EstimatingCounter) -> None:
        config = ModelConfig(model="openai/gpt-4o")  # no context_window
        client = _make_mock_model_client(config)
        manager = ContextManager(client, counter)
        assert manager.budget == _DEFAULT_CONTEXT_WINDOW - _DEFAULT_RESERVE_TOKENS

    def test_custom_reserve_tokens(
        self, mock_client: MagicMock, counter: EstimatingCounter
    ) -> None:
        manager = ContextManager(
            mock_client, counter, context_window=4096, reserve_tokens=512
        )
        assert manager.budget == 4096 - 512

    async def test_passthrough_when_under_budget(
        self,
        mock_client: MagicMock,
        counter: EstimatingCounter,
        history: ConversationHistory,
    ) -> None:
        pruner = SlidingWindowPruner()
        manager = ContextManager(mock_client, counter, pruner, context_window=100000)

        result = await manager.generate(history)

        assert result.text == "Hello!"
        # Should have called generate with the original history (no pruning).
        call_args = mock_client.generate.call_args
        passed_history: ConversationHistory = call_args.args[0]
        assert len(passed_history.messages) == 2

    async def test_prunes_when_over_budget(
        self,
        mock_client: MagicMock,
        counter: EstimatingCounter,
    ) -> None:
        # Create a history that will be large.
        messages = [CanonicalMessage.system("sys")]
        for i in range(20):
            messages.append(CanonicalMessage.user(f"Message number {i} " * 10))
        history = ConversationHistory(messages=messages)

        pruner = SlidingWindowPruner(min_recent=2)
        manager = ContextManager(
            mock_client, counter, pruner, context_window=200, reserve_tokens=50
        )

        await manager.generate(history)

        call_args = mock_client.generate.call_args
        passed_history: ConversationHistory = call_args.args[0]
        # Should have fewer messages than original.
        assert len(passed_history.messages) < len(history.messages)

    async def test_no_pruner_passes_through(
        self,
        mock_client: MagicMock,
        counter: EstimatingCounter,
    ) -> None:
        messages = [CanonicalMessage.user(f"msg{i}") for i in range(10)]
        history = ConversationHistory(messages=messages)

        # No pruner — even if over budget, history passes through.
        manager = ContextManager(mock_client, counter, context_window=10)
        await manager.generate(history)

        call_args = mock_client.generate.call_args
        passed_history: ConversationHistory = call_args.args[0]
        assert len(passed_history.messages) == 10

    async def test_passes_tools_and_kwargs(
        self,
        mock_client: MagicMock,
        counter: EstimatingCounter,
        history: ConversationHistory,
    ) -> None:
        manager = ContextManager(mock_client, counter, context_window=100000)

        tools: list[dict[str, Any]] = [{"type": "function", "function": {"name": "test"}}]
        await manager.generate(history, tools=tools, temperature=0.5)

        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs["tools"] == tools
        assert call_kwargs.kwargs["temperature"] == 0.5

    async def test_works_with_async_pruner(
        self,
        mock_client: MagicMock,
        counter: EstimatingCounter,
    ) -> None:
        """Verify ContextManager handles async pruners (SummarizerPruner pattern)."""
        pruned_history = ConversationHistory(
            messages=[CanonicalMessage.user("summarized")]
        )

        async_pruner = MagicMock()
        async_pruner.prune = AsyncMock(return_value=pruned_history)

        messages = [CanonicalMessage.user(f"msg{i} " * 20) for i in range(10)]
        history = ConversationHistory(messages=messages)

        manager = ContextManager(
            mock_client, counter, async_pruner, context_window=50, reserve_tokens=10
        )
        await manager.generate(history)

        call_args = mock_client.generate.call_args
        passed_history: ConversationHistory = call_args.args[0]
        assert len(passed_history.messages) == 1
        assert passed_history.messages[0].text == "summarized"
