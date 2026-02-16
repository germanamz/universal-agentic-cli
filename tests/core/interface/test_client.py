"""Tests for ModelClient — unit tests with mocked LiteLLM."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uac.core.interface.client import ModelClient, get_transpiler
from uac.core.interface.config import ModelConfig
from uac.core.interface.models import CanonicalMessage, ConversationHistory
from uac.core.interface.transpilers.anthropic import AnthropicTranspiler
from uac.core.interface.transpilers.gemini import GeminiTranspiler
from uac.core.interface.transpilers.openai import OpenAITranspiler


class TestGetTranspiler:
    def test_openai(self) -> None:
        assert isinstance(get_transpiler("openai"), OpenAITranspiler)

    def test_anthropic(self) -> None:
        assert isinstance(get_transpiler("anthropic"), AnthropicTranspiler)

    def test_gemini(self) -> None:
        assert isinstance(get_transpiler("gemini"), GeminiTranspiler)

    def test_google(self) -> None:
        assert isinstance(get_transpiler("google"), GeminiTranspiler)

    def test_vertex_ai(self) -> None:
        assert isinstance(get_transpiler("vertex_ai"), GeminiTranspiler)

    def test_unknown_defaults_to_openai(self) -> None:
        assert isinstance(get_transpiler("unknown"), OpenAITranspiler)


class TestModelConfig:
    def test_provider_extraction(self) -> None:
        config = ModelConfig(model="openai/gpt-4o")
        assert config.provider == "openai"

    def test_provider_no_prefix(self) -> None:
        config = ModelConfig(model="gpt-4o")
        assert config.provider == "openai"

    def test_anthropic_provider(self) -> None:
        config = ModelConfig(model="anthropic/claude-3-opus")
        assert config.provider == "anthropic"

    def test_capabilities(self) -> None:
        config = ModelConfig(
            model="openai/gpt-4o",
            capabilities={"native_tool_calling": True, "vision": True},
        )
        assert config.capabilities["native_tool_calling"] is True


def _make_mock_response(
    content: str | None = "Hello!",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4o",
) -> MagicMock:
    """Create a mock LiteLLM response object."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model

    return response


class TestModelClient:
    @pytest.fixture
    def config(self) -> ModelConfig:
        return ModelConfig(model="openai/gpt-4o", api_key="test-key")

    @pytest.fixture
    def client(self, config: ModelConfig) -> ModelClient:
        return ModelClient(config)

    @pytest.fixture
    def history(self) -> ConversationHistory:
        return ConversationHistory(
            messages=[
                CanonicalMessage.system("You are helpful."),
                CanonicalMessage.user("Hello"),
            ]
        )

    @patch("uac.core.interface.client.litellm")
    async def test_generate_text_response(
        self, mock_litellm: MagicMock, client: ModelClient, history: ConversationHistory
    ) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_make_mock_response())

        result = await client.generate(history)

        assert result.role == "assistant"
        assert result.text == "Hello!"
        assert result.metadata["usage"]["total_tokens"] == 15
        assert result.metadata["finish_reason"] == "stop"

    @patch("uac.core.interface.client.litellm")
    async def test_generate_passes_model(
        self, mock_litellm: MagicMock, client: ModelClient, history: ConversationHistory
    ) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_make_mock_response())

        await client.generate(history)

        call_kwargs = mock_litellm.acompletion.call_args
        assert call_kwargs.kwargs["model"] == "openai/gpt-4o"
        assert call_kwargs.kwargs["api_key"] == "test-key"

    @patch("uac.core.interface.client.litellm")
    async def test_generate_with_tools(
        self, mock_litellm: MagicMock, client: ModelClient, history: ConversationHistory
    ) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_make_mock_response())

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        await client.generate(history, tools=tools)

        call_kwargs = mock_litellm.acompletion.call_args
        assert call_kwargs.kwargs["tools"] == tools

    @patch("uac.core.interface.client.litellm")
    async def test_generate_tool_call_response(
        self, mock_litellm: MagicMock, client: ModelClient, history: ConversationHistory
    ) -> None:
        tc_mock = MagicMock()
        tc_mock.id = "call-1"
        tc_mock.function.name = "calculator"
        tc_mock.function.arguments = '{"expression": "2+2"}'

        mock_litellm.acompletion = AsyncMock(
            return_value=_make_mock_response(
                content=None,
                tool_calls=[tc_mock],
                finish_reason="tool_calls",
            )
        )

        result = await client.generate(history)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments == {"expression": "2+2"}
        assert result.text == ""

    @patch("uac.core.interface.client.litellm")
    async def test_generate_passes_kwargs(
        self, mock_litellm: MagicMock, client: ModelClient, history: ConversationHistory
    ) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_make_mock_response())

        await client.generate(history, temperature=0.7, max_tokens=100)

        call_kwargs = mock_litellm.acompletion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7
        assert call_kwargs.kwargs["max_tokens"] == 100

    @patch("uac.core.interface.client.litellm")
    async def test_generate_with_api_base(
        self, mock_litellm: MagicMock, history: ConversationHistory
    ) -> None:
        config = ModelConfig(
            model="openai/gpt-4o",
            api_key="key",
            api_base="http://localhost:8000",
        )
        client = ModelClient(config)
        mock_litellm.acompletion = AsyncMock(return_value=_make_mock_response())

        await client.generate(history)

        call_kwargs = mock_litellm.acompletion.call_args
        assert call_kwargs.kwargs["api_base"] == "http://localhost:8000"

    def test_prepare_messages(self, client: ModelClient, history: ConversationHistory) -> None:
        messages = client._prepare_messages(history)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
