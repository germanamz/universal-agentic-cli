"""Shared helpers for E2E integration tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def make_mock_litellm_response(
    content: str = "",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    model: str = "openai/gpt-4o",
) -> MagicMock:
    """Create a ``MagicMock`` matching LiteLLM's response structure.

    The mock mirrors ``choices[0].message`` with content, tool_calls,
    plus top-level ``usage`` and ``model`` attributes.
    """
    message = MagicMock()
    message.content = content or None
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    usage.total_tokens = 30

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model

    return response
