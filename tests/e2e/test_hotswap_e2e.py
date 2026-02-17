"""E2E tests for model hot-swapping — NativeStrategy vs PromptedStrategy."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.e2e.conftest import make_mock_litellm_response
from uac.core.interface.client import ModelClient
from uac.core.interface.config import ModelConfig
from uac.core.orchestration.models import AgentManifest
from uac.core.orchestration.primitives import AgentNode
from uac.core.orchestration.topologies.pipeline import PipelineOrchestrator
from uac.core.polyfills.strategy import NativeStrategy, PromptedStrategy


def _make_response_router(
    responses: dict[str, str],
    model: str = "openai/gpt-4o",
) -> Any:
    """Return an async side-effect that dispatches by system prompt content."""

    async def _router(**kwargs: Any) -> MagicMock:
        messages = kwargs.get("messages", [])
        system_text = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_text += msg.get("content", "") + " "
        for key, text in responses.items():
            if key in system_text:
                return make_mock_litellm_response(content=text, model=model)
        return make_mock_litellm_response(content="fallback", model=model)

    return _router


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


def _make_agent(name: str, model_str: str) -> AgentNode:
    manifest = AgentManifest(name=name, system_prompt_template=f"You are {name}.")
    config = ModelConfig(model=model_str)
    client = ModelClient(config)
    return AgentNode(manifest=manifest, client=client, tools=_TOOLS)


async def _run_pipeline(model_str: str) -> tuple[Any, list[dict[str, Any]]]:
    """Run a 2-agent pipeline with the given model, return (board, call_kwargs)."""
    agents = {
        "analyzer": _make_agent("analyzer", model_str),
        "writer": _make_agent("writer", model_str),
    }
    responses = {
        "analyzer": "Analysis complete.",
        "writer": "Report written.",
    }
    call_kwargs: list[dict[str, Any]] = []

    async def capturing_router(**kwargs: Any) -> MagicMock:
        call_kwargs.append(dict(kwargs))
        return await _make_response_router(responses, model=model_str)(**kwargs)

    with patch("uac.core.interface.client.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(side_effect=capturing_router)

        orch = PipelineOrchestrator(
            agents=agents,
            order=["analyzer", "writer"],
        )
        board = await orch.run("Analyze and write")

    return board, call_kwargs


class TestHotSwapE2E:
    @pytest.mark.asyncio
    async def test_native_strategy_auto_selected(self) -> None:
        """openai/gpt-4o auto-selects NativeStrategy."""
        agent = _make_agent("test", "openai/gpt-4o")
        assert isinstance(agent.client.strategy, NativeStrategy)

    @pytest.mark.asyncio
    async def test_prompted_strategy_auto_selected(self) -> None:
        """ollama/mistral-7b auto-selects PromptedStrategy."""
        agent = _make_agent("test", "ollama/mistral-7b")
        assert isinstance(agent.client.strategy, PromptedStrategy)

    @pytest.mark.asyncio
    async def test_structural_equivalence(self) -> None:
        """Same pipeline with different models produces structurally equivalent traces."""
        board_native, _ = await _run_pipeline("openai/gpt-4o")
        board_prompted, _ = await _run_pipeline("ollama/mistral-7b")

        # Both should have the same number of generate trace entries
        native_agents = [
            e.agent_id for e in board_native.execution_trace if e.action == "generate"
        ]
        prompted_agents = [
            e.agent_id for e in board_prompted.execution_trace if e.action == "generate"
        ]
        assert native_agents == prompted_agents

        # Both should have the same belief_state
        assert board_native.belief_state == board_prompted.belief_state

    @pytest.mark.asyncio
    async def test_prompted_strategy_strips_tools(self) -> None:
        """PromptedStrategy strips tools from the litellm call and injects ReAct prompt."""
        _, call_kwargs = await _run_pipeline("ollama/mistral-7b")

        for call in call_kwargs:
            # Tools should not be passed to litellm
            assert "tools" not in call or call.get("tools") is None

    @pytest.mark.asyncio
    async def test_prompted_strategy_injects_react(self) -> None:
        """PromptedStrategy adds a ReAct system prompt mentioning tool names."""
        _, call_kwargs = await _run_pipeline("ollama/mistral-7b")

        for call in call_kwargs:
            messages = call.get("messages", [])
            system_texts = [m["content"] for m in messages if m.get("role") == "system"]
            combined = " ".join(system_texts)
            # The ReAct prompt should mention the tool name
            assert "search" in combined

    @pytest.mark.asyncio
    async def test_native_strategy_passes_tools(self) -> None:
        """NativeStrategy passes tools through to litellm."""
        _, call_kwargs = await _run_pipeline("openai/gpt-4o")

        for call in call_kwargs:
            assert "tools" in call
            tools = call["tools"]
            assert len(tools) == 1
            assert tools[0]["function"]["name"] == "search"
