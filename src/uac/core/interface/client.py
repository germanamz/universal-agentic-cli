"""ModelClient — unified async interface to LLMs via LiteLLM.

Wraps LiteLLM behind a CMS-native interface so the rest of the system
only ever works with CanonicalMessage and ConversationHistory.
"""

from typing import Any

import litellm

from uac.core.interface.config import ModelConfig
from uac.core.interface.models import (
    CanonicalMessage,
    ContentPart,
    ConversationHistory,
    TextContent,
    ToolCall,
)
from uac.core.interface.transpiler import Transpiler
from uac.core.interface.transpilers.anthropic import AnthropicTranspiler
from uac.core.interface.transpilers.gemini import GeminiTranspiler
from uac.core.interface.transpilers.openai import OpenAITranspiler
from uac.core.polyfills.capabilities import CapabilityRegistry
from uac.core.polyfills.strategy import NativeStrategy, PromptedStrategy, ToolCallingStrategy
from uac.utils.telemetry import (
    ATTR_FINISH_REASON,
    ATTR_MODEL,
    ATTR_PROVIDER,
    ATTR_STRATEGY,
    ATTR_TOKENS_COMPLETION,
    ATTR_TOKENS_PROMPT,
    ATTR_TOKENS_TOTAL,
    get_tracer,
)

_tracer = get_tracer(__name__)

_default_registry: CapabilityRegistry | None = None


def _get_default_registry() -> CapabilityRegistry:
    """Return (and cache) the default capability registry."""
    global _default_registry
    if _default_registry is None:
        from uac.core.polyfills.registry_data import build_default_registry

        _default_registry = build_default_registry()
    return _default_registry


def get_transpiler(provider: str) -> Transpiler:
    """Return the appropriate transpiler for a provider."""
    mapping: dict[str, Transpiler] = {
        "openai": OpenAITranspiler(),
        "anthropic": AnthropicTranspiler(),
        "gemini": GeminiTranspiler(),
        "google": GeminiTranspiler(),
        "vertex_ai": GeminiTranspiler(),
    }
    return mapping.get(provider, OpenAITranspiler())


class ToolDefinition(CanonicalMessage):
    """Schema for a tool that can be passed to the model.

    This is a lightweight wrapper — actual tool schemas are defined by
    the protocol layer (MCP/UTCP) and converted here.
    """


class ModelClient:
    """Async client for generating LLM responses via LiteLLM.

    Usage::

        config = ModelConfig(model="openai/gpt-4o")
        client = ModelClient(config)
        response = await client.generate(history)
    """

    def __init__(
        self,
        config: ModelConfig,
        registry: CapabilityRegistry | None = None,
        strategy: ToolCallingStrategy | None = None,
    ) -> None:
        self.config = config
        self.transpiler = get_transpiler(config.provider)

        resolved_registry = registry or _get_default_registry()
        profile = resolved_registry.resolve(config)

        if strategy is not None:
            self.strategy: ToolCallingStrategy = strategy
        elif profile.supports_native_tools:
            self.strategy = NativeStrategy()
        else:
            self.strategy = PromptedStrategy()

    async def generate(
        self,
        messages: ConversationHistory,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CanonicalMessage:
        """Generate a response from the configured model.

        Args:
            messages: The conversation history in CMS format.
            tools: Optional list of tool definitions in OpenAI function schema format.
            **kwargs: Additional parameters passed to LiteLLM.

        Returns:
            A CanonicalMessage representing the model's response.
        """
        with _tracer.start_as_current_span("model.generate") as span:
            span.set_attribute(ATTR_MODEL, self.config.model)
            span.set_attribute(ATTR_PROVIDER, self.config.provider)
            span.set_attribute(ATTR_STRATEGY, self.strategy.__class__.__name__)

            # Let the strategy transform messages/tools before calling
            prepared_messages, prepared_tools = self.strategy.prepare(messages, tools)

            # Build LiteLLM call parameters
            litellm_messages = self._prepare_messages(prepared_messages)
            call_kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": litellm_messages,
                **kwargs,
            }

            if self.config.api_key:
                call_kwargs["api_key"] = self.config.api_key
            if self.config.api_base:
                call_kwargs["api_base"] = self.config.api_base

            if prepared_tools:
                call_kwargs["tools"] = prepared_tools

            # Call LiteLLM (type stubs are incomplete)
            response = await litellm.acompletion(**call_kwargs)  # pyright: ignore[reportUnknownMemberType]

            # Convert response to CMS and let the strategy interpret it
            parsed = self._parse_response(response)
            result = self.strategy.interpret(parsed)

            # Record token usage and finish reason from response metadata
            usage: dict[str, Any] | None = result.metadata.get("usage")
            if isinstance(usage, dict):
                span.set_attribute(ATTR_TOKENS_PROMPT, int(usage.get("prompt_tokens", 0)))
                span.set_attribute(ATTR_TOKENS_COMPLETION, int(usage.get("completion_tokens", 0)))
                span.set_attribute(ATTR_TOKENS_TOTAL, int(usage.get("total_tokens", 0)))
            finish_reason = result.metadata.get("finish_reason")
            if finish_reason is not None:
                span.set_attribute(ATTR_FINISH_REASON, str(finish_reason))

            return result

    def _prepare_messages(self, history: ConversationHistory) -> list[dict[str, Any]]:
        """Convert CMS history to LiteLLM-compatible messages.

        LiteLLM uses OpenAI's message format, so we use the OpenAI transpiler
        for the message list. Provider-specific adjustments (e.g. system prompt
        extraction for Anthropic) are handled by LiteLLM internally.
        """
        # LiteLLM expects OpenAI-style messages; it handles provider adaptation
        openai_transpiler = OpenAITranspiler()
        payload = openai_transpiler.to_provider(history)
        result: list[dict[str, Any]] = payload["messages"]
        return result

    def _parse_response(self, response: Any) -> CanonicalMessage:
        """Convert a LiteLLM response to a CanonicalMessage.

        LiteLLM returns OpenAI-compatible response objects regardless of
        the underlying provider.
        """
        message = response.choices[0].message

        content: list[ContentPart] = []
        if message.content:
            content = [TextContent(text=message.content)]

        tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_parse_arguments(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]

        metadata: dict[str, Any] = {}
        if hasattr(response, "usage") and response.usage:
            metadata["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        metadata["finish_reason"] = response.choices[0].finish_reason
        metadata["model"] = response.model

        return CanonicalMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            metadata=metadata,
        )


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Parse JSON string arguments from a tool call."""
    import json

    try:
        result: dict[str, Any] = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        result = {"raw": raw}
    return result
