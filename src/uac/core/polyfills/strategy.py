"""Tool-calling strategy — selects native or prompted (ReAct) mode.

The ``ToolCallingStrategy`` protocol defines a prepare/interpret pair
that wraps the LiteLLM call. Two implementations are provided:

* ``NativeStrategy`` — passthrough for models with native tool calling.
* ``PromptedStrategy`` — injects a ReAct system prompt and parses the
  resulting free-form text back into structured ``ToolCall`` objects.
"""

from typing import Any, Protocol

from uac.core.interface.models import CanonicalMessage, ConversationHistory, ToolCall
from uac.core.polyfills.react_injector import ReActInjector
from uac.core.polyfills.react_parser import ReActParser


class ToolCallingStrategy(Protocol):
    """Prepare tools before a model call and interpret the response."""

    def prepare(
        self,
        messages: ConversationHistory,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[ConversationHistory, list[dict[str, Any]] | None]:
        """Transform messages and tools before sending to the model."""
        ...

    def interpret(self, response: CanonicalMessage) -> CanonicalMessage:
        """Post-process the model response."""
        ...


class NativeStrategy:
    """Passthrough — the model handles tool calling natively."""

    def prepare(
        self,
        messages: ConversationHistory,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[ConversationHistory, list[dict[str, Any]] | None]:
        return messages, tools

    def interpret(self, response: CanonicalMessage) -> CanonicalMessage:
        return response


class PromptedStrategy:
    """ReAct prompt injection for models without native tool calling."""

    def __init__(self) -> None:
        self._injector = ReActInjector()
        self._parser = ReActParser()

    def prepare(
        self,
        messages: ConversationHistory,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[ConversationHistory, list[dict[str, Any]] | None]:
        """Inject a ReAct system prompt and strip tools."""
        if not tools:
            return messages, None

        react_prompt = self._injector.inject(tools)

        # Prepend the ReAct instruction as a system message
        new_history = ConversationHistory(
            messages=[CanonicalMessage.system(react_prompt), *messages.messages],
        )
        # Strip tools — the model receives them only via the prompt
        return new_history, None

    def interpret(self, response: CanonicalMessage) -> CanonicalMessage:
        """Parse ReAct patterns from the assistant's text response."""
        text = response.text
        if not text:
            return response

        result = self._parser.parse(text)

        if result.tool_call:
            tool_calls: list[ToolCall] = [result.tool_call]
            return CanonicalMessage(
                role="assistant",
                content=response.content,
                tool_calls=tool_calls,
                metadata=response.metadata,
            )

        if result.final_answer:
            return CanonicalMessage.assistant(result.final_answer, **response.metadata)

        return response
