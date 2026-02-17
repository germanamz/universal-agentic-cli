"""E2E tests for PipelineOrchestrator with real orchestration loop."""

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


def _make_response_router(
    responses: dict[str, str],
    model: str = "openai/gpt-4o",
) -> Any:
    """Return an async side-effect that dispatches responses by system prompt content."""

    async def _router(**kwargs: Any) -> MagicMock:
        messages = kwargs.get("messages", [])
        system_text = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
                break
        for key, text in responses.items():
            if key in system_text:
                return make_mock_litellm_response(content=text, model=model)
        return make_mock_litellm_response(content="fallback", model=model)

    return _router


def _make_agent(name: str, model_str: str = "openai/gpt-4o") -> AgentNode:
    manifest = AgentManifest(name=name, system_prompt_template=f"You are {name}.")
    config = ModelConfig(model=model_str)
    client = ModelClient(config)
    return AgentNode(manifest=manifest, client=client)


class TestPipelineE2E:
    @pytest.mark.asyncio
    async def test_execution_order_via_trace(self) -> None:
        """Three agents execute in declared order; trace records the sequence."""
        agents = {
            "researcher": _make_agent("researcher"),
            "drafter": _make_agent("drafter"),
            "reviewer": _make_agent("reviewer"),
        }
        responses = {
            "researcher": "Research findings: AI is transforming software.",
            "drafter": "Draft: Based on research, AI enables new paradigms.",
            "reviewer": "Review: Draft is well-structured. Approved.",
        }

        with patch("uac.core.interface.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_make_response_router(responses))

            orch = PipelineOrchestrator(
                agents=agents,
                order=["researcher", "drafter", "reviewer"],
            )
            board = await orch.run("Research, draft, and review a report")

        agent_ids = [e.agent_id for e in board.execution_trace if e.action == "generate"]
        assert agent_ids == ["researcher", "drafter", "reviewer"]

    @pytest.mark.asyncio
    async def test_blackboard_artifact_accumulation(self) -> None:
        """Each agent's response is stored in artifacts under last_response."""
        agents = {
            "researcher": _make_agent("researcher"),
            "drafter": _make_agent("drafter"),
            "reviewer": _make_agent("reviewer"),
        }
        responses = {
            "researcher": "Found key insights.",
            "drafter": "Drafted the report.",
            "reviewer": "Approved the draft.",
        }

        with patch("uac.core.interface.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_make_response_router(responses))

            orch = PipelineOrchestrator(
                agents=agents,
                order=["researcher", "drafter", "reviewer"],
            )
            board = await orch.run("Write a report")

        last_response = board.artifacts.get("last_response", {})
        # Deep-merge means all three should be present
        assert "researcher" in last_response
        assert "drafter" in last_response
        assert "reviewer" in last_response

    @pytest.mark.asyncio
    async def test_context_propagation(self) -> None:
        """Drafter sees researcher's trace entries in its context."""
        agents = {
            "researcher": _make_agent("researcher"),
            "drafter": _make_agent("drafter"),
        }
        responses = {
            "researcher": "Key finding: quantum computing.",
            "drafter": "Draft about quantum computing.",
        }
        calls_received: list[dict[str, Any]] = []

        original_router = _make_response_router(responses)

        async def capturing_router(**kwargs: Any) -> MagicMock:
            calls_received.append(dict(kwargs))
            return await original_router(**kwargs)

        with patch("uac.core.interface.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=capturing_router)

            orch = PipelineOrchestrator(
                agents=agents,
                order=["researcher", "drafter"],
            )
            await orch.run("Write about quantum computing")

        # The drafter's call (second) should include trace/artifact context
        assert len(calls_received) == 2
        drafter_messages = calls_received[1]["messages"]
        # Should have a user message with context (trace, artifacts)
        user_msgs = [m for m in drafter_messages if m.get("role") == "user"]
        assert len(user_msgs) > 0
        user_text = user_msgs[0].get("content", "")
        # The context should mention the researcher's trace
        assert "researcher" in user_text

    @pytest.mark.asyncio
    async def test_snapshot_and_restore(self) -> None:
        """Blackboard state can be snapshotted and restored after a pipeline run."""
        from uac.core.blackboard.blackboard import Blackboard

        agents = {
            "alpha": _make_agent("alpha"),
            "beta": _make_agent("beta"),
        }
        responses = {"alpha": "Alpha output.", "beta": "Beta output."}

        with patch("uac.core.interface.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_make_response_router(responses))

            orch = PipelineOrchestrator(
                agents=agents,
                order=["alpha", "beta"],
            )
            board = await orch.run("Snapshot test")

        snapshot = board.snapshot()
        restored = Blackboard.restore(snapshot)

        assert restored.belief_state == board.belief_state
        assert len(restored.execution_trace) == len(board.execution_trace)
        assert restored.artifacts == board.artifacts

    @pytest.mark.asyncio
    async def test_correct_model_in_litellm_call(self) -> None:
        """The model string passed to litellm matches the agent's config."""
        agents = {"solo": _make_agent("solo", "openai/gpt-4o")}
        call_kwargs: list[dict[str, Any]] = []

        async def capturing_router(**kwargs: Any) -> MagicMock:
            call_kwargs.append(dict(kwargs))
            return make_mock_litellm_response(content="Done.", model="openai/gpt-4o")

        with patch("uac.core.interface.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=capturing_router)

            orch = PipelineOrchestrator(agents=agents, order=["solo"])
            await orch.run("Model check")

        assert len(call_kwargs) == 1
        assert call_kwargs[0]["model"] == "openai/gpt-4o"
