"""Integration tests for ModelClient — requires a real API key.

These tests are gated behind the UAC_INTEGRATION_TEST env var.
Set it to any truthy value to run:

    UAC_INTEGRATION_TEST=1 uv run pytest tests/core/interface/test_client_integration.py -v
"""

import os

import pytest

from uac.core.interface.client import ModelClient
from uac.core.interface.config import ModelConfig
from uac.core.interface.models import CanonicalMessage, ConversationHistory

SKIP_REASON = "Set UAC_INTEGRATION_TEST=1 and provide API keys to run"
requires_integration = pytest.mark.skipif(
    not os.environ.get("UAC_INTEGRATION_TEST"), reason=SKIP_REASON
)


@requires_integration
class TestModelClientIntegration:
    async def test_openai_simple_generation(self) -> None:
        config = ModelConfig(model="openai/gpt-4o-mini")
        client = ModelClient(config)
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("Reply with exactly one word."),
                CanonicalMessage.user("What color is the sky?"),
            ]
        )
        response = await client.generate(history, temperature=0.0)

        assert response.role == "assistant"
        assert len(response.text) > 0
        assert response.metadata.get("usage") is not None

    async def test_openai_tool_calling(self) -> None:
        config = ModelConfig(model="openai/gpt-4o-mini")
        client = ModelClient(config)
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("Use the calculator tool to answer."),
                CanonicalMessage.user("What is 2+2?"),
            ]
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The math expression",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]
        response = await client.generate(history, tools=tools, temperature=0.0)

        assert response.role == "assistant"
        # Should either call the tool or answer directly
        if response.tool_calls:
            assert response.tool_calls[0].name == "calculator"
