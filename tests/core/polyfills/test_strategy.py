"""Tests for ToolCallingStrategy implementations."""

from typing import Any

from uac.core.interface.models import CanonicalMessage, ConversationHistory, TextContent
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


class TestNativeStrategy:
    def setup_method(self) -> None:
        self.strategy = NativeStrategy()

    def test_prepare_passthrough(self) -> None:
        history = ConversationHistory(messages=[CanonicalMessage.user("hi")])
        msgs, tools = self.strategy.prepare(history, SAMPLE_TOOLS)
        assert msgs is history
        assert tools is SAMPLE_TOOLS

    def test_prepare_no_tools(self) -> None:
        history = ConversationHistory(messages=[CanonicalMessage.user("hi")])
        msgs, tools = self.strategy.prepare(history, None)
        assert msgs is history
        assert tools is None

    def test_interpret_passthrough(self) -> None:
        response = CanonicalMessage.assistant("Hello!")
        result = self.strategy.interpret(response)
        assert result is response


class TestPromptedStrategy:
    def setup_method(self) -> None:
        self.strategy = PromptedStrategy()

    def test_prepare_injects_react_prompt(self) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("You are helpful."),
                CanonicalMessage.user("What is 2+2?"),
            ]
        )
        msgs, tools = self.strategy.prepare(history, SAMPLE_TOOLS)

        # Tools should be stripped
        assert tools is None

        # ReAct system message should be prepended
        assert len(msgs.messages) == 3
        assert msgs.messages[0].role == "system"
        assert "calculator" in msgs.messages[0].text
        assert "Thought:" in msgs.messages[0].text

    def test_prepare_no_tools_passthrough(self) -> None:
        history = ConversationHistory(messages=[CanonicalMessage.user("hi")])
        msgs, tools = self.strategy.prepare(history, None)
        assert msgs is history
        assert tools is None

    def test_interpret_tool_call(self) -> None:
        response = CanonicalMessage(
            role="assistant",
            content=[
                TextContent(
                    text=(
                        "Thought: I need to calculate\n"
                        "Action: calculator\n"
                        'Action Input: {"expression": "2+2"}'
                    )
                )
            ],
        )
        result = self.strategy.interpret(response)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments == {"expression": "2+2"}

    def test_interpret_final_answer(self) -> None:
        response = CanonicalMessage(
            role="assistant",
            content=[
                TextContent(text=("Thought: I know the answer\nFinal Answer: The answer is 4."))
            ],
        )
        result = self.strategy.interpret(response)
        assert result.tool_calls is None
        assert result.text == "The answer is 4."

    def test_interpret_empty_text(self) -> None:
        response = CanonicalMessage(role="assistant", content=[])
        result = self.strategy.interpret(response)
        assert result is response

    def test_original_history_not_mutated(self) -> None:
        history = ConversationHistory(messages=[CanonicalMessage.user("hi")])
        original_len = len(history)
        self.strategy.prepare(history, SAMPLE_TOOLS)
        assert len(history) == original_len
