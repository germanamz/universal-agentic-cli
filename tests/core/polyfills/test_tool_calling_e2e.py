"""E2E smoke tests — native vs prompted tool calling through ModelClient."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uac.core.interface.client import ModelClient
from uac.core.interface.config import ModelConfig
from uac.core.interface.models import CanonicalMessage, ConversationHistory
from uac.core.polyfills.strategy import NativeStrategy, PromptedStrategy

SAMPLE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
            },
        },
    }
]


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


class TestNativeModeE2E:
    """GPT-4o auto-selects NativeStrategy — tools passed through to LiteLLM."""

    @pytest.fixture
    def client(self) -> ModelClient:
        return ModelClient(ModelConfig(model="openai/gpt-4o"))

    @pytest.fixture
    def history(self) -> ConversationHistory:
        return ConversationHistory(
            messages=[
                CanonicalMessage.system("You are helpful."),
                CanonicalMessage.user("What is 2+2?"),
            ]
        )

    def test_auto_selects_native_strategy(self, client: ModelClient) -> None:
        assert isinstance(client.strategy, NativeStrategy)

    @patch("uac.core.interface.client.litellm")
    async def test_tools_passed_to_litellm(
        self,
        mock_litellm: MagicMock,
        client: ModelClient,
        history: ConversationHistory,
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

        result = await client.generate(history, tools=SAMPLE_TOOLS)

        # Tools should have been passed to LiteLLM
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs["tools"] == SAMPLE_TOOLS

        # Tool calls should be parsed from structured response
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments == {"expression": "2+2"}


class TestPromptedModeE2E:
    """Llama auto-selects PromptedStrategy — ReAct prompt injected."""

    @pytest.fixture
    def client(self) -> ModelClient:
        return ModelClient(ModelConfig(model="ollama/llama-3-8b"))

    @pytest.fixture
    def history(self) -> ConversationHistory:
        return ConversationHistory(
            messages=[
                CanonicalMessage.system("You are helpful."),
                CanonicalMessage.user("What is 2+2?"),
            ]
        )

    def test_auto_selects_prompted_strategy(self, client: ModelClient) -> None:
        assert isinstance(client.strategy, PromptedStrategy)

    @patch("uac.core.interface.client.litellm")
    async def test_react_prompt_injected_tools_stripped(
        self,
        mock_litellm: MagicMock,
        client: ModelClient,
        history: ConversationHistory,
    ) -> None:
        react_response = (
            "Thought: I need to calculate 2+2\n"
            "Action: calculator\n"
            'Action Input: {"expression": "2+2"}'
        )
        mock_litellm.acompletion = AsyncMock(
            return_value=_make_mock_response(content=react_response)
        )

        result = await client.generate(history, tools=SAMPLE_TOOLS)

        # Tools should NOT have been passed to LiteLLM
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert "tools" not in call_kwargs

        # ReAct system prompt should have been injected
        messages = call_kwargs["messages"]
        assert any("calculator" in str(m.get("content", "")) for m in messages)

        # Tool calls should be extracted from text
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments == {"expression": "2+2"}


class TestFinalAnswer:
    """Prompted mode with Final Answer produces clean text response."""

    @patch("uac.core.interface.client.litellm")
    async def test_final_answer_extracted(self, mock_litellm: MagicMock) -> None:
        client = ModelClient(ModelConfig(model="ollama/llama-3-8b"))
        history = ConversationHistory(
            messages=[CanonicalMessage.user("What is the capital of France?")]
        )

        react_response = "Thought: I know this one\nFinal Answer: The capital of France is Paris."
        mock_litellm.acompletion = AsyncMock(
            return_value=_make_mock_response(content=react_response)
        )

        result = await client.generate(history, tools=SAMPLE_TOOLS)

        assert result.tool_calls is None
        assert result.text == "The capital of France is Paris."


class TestExplicitOverride:
    """Force PromptedStrategy on a model that would normally use Native."""

    def test_explicit_strategy_override(self) -> None:
        strategy = PromptedStrategy()
        client = ModelClient(
            ModelConfig(model="openai/gpt-4o"),
            strategy=strategy,
        )
        assert isinstance(client.strategy, PromptedStrategy)

    @patch("uac.core.interface.client.litellm")
    async def test_explicit_prompted_on_native_model(self, mock_litellm: MagicMock) -> None:
        client = ModelClient(
            ModelConfig(model="openai/gpt-4o"),
            strategy=PromptedStrategy(),
        )
        history = ConversationHistory(messages=[CanonicalMessage.user("What is 2+2?")])

        react_response = (
            'Thought: Calculate\nAction: calculator\nAction Input: {"expression": "2+2"}'
        )
        mock_litellm.acompletion = AsyncMock(
            return_value=_make_mock_response(content=react_response)
        )

        result = await client.generate(history, tools=SAMPLE_TOOLS)

        # Even though it's GPT-4o, tools should be stripped (PromptedStrategy)
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert "tools" not in call_kwargs

        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "calculator"
