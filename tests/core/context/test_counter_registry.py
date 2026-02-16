"""Tests for counter_registry — provider-based auto-selection."""

from uac.core.context.counter import EstimatingCounter, TiktokenCounter
from uac.core.context.counter_registry import get_counter
from uac.core.interface.config import ModelConfig


class TestGetCounter:
    def test_openai_returns_tiktoken(self) -> None:
        config = ModelConfig(model="openai/gpt-4o")
        counter = get_counter(config)
        assert isinstance(counter, TiktokenCounter)

    def test_azure_returns_tiktoken(self) -> None:
        config = ModelConfig(model="azure/gpt-4")
        counter = get_counter(config)
        assert isinstance(counter, TiktokenCounter)

    def test_azure_ai_returns_tiktoken(self) -> None:
        config = ModelConfig(model="azure_ai/gpt-4o")
        counter = get_counter(config)
        assert isinstance(counter, TiktokenCounter)

    def test_anthropic_returns_estimating(self) -> None:
        config = ModelConfig(model="anthropic/claude-3-opus")
        counter = get_counter(config)
        assert isinstance(counter, EstimatingCounter)

    def test_gemini_returns_estimating(self) -> None:
        config = ModelConfig(model="gemini/gemini-pro")
        counter = get_counter(config)
        assert isinstance(counter, EstimatingCounter)

    def test_unknown_returns_estimating(self) -> None:
        config = ModelConfig(model="local/my-model")
        counter = get_counter(config)
        assert isinstance(counter, EstimatingCounter)

    def test_bare_model_name_defaults_openai(self) -> None:
        # No prefix → provider == "openai" → tiktoken
        config = ModelConfig(model="gpt-4o")
        counter = get_counter(config)
        assert isinstance(counter, TiktokenCounter)
