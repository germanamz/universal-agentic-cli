"""E2E tests for StarOrchestrator with real orchestration loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.e2e.conftest import make_mock_litellm_response
from uac.core.interface.client import ModelClient
from uac.core.interface.config import ModelConfig
from uac.core.orchestration.models import AgentManifest
from uac.core.orchestration.primitives import AgentNode
from uac.core.orchestration.topologies.star import StarOrchestrator


def _make_response_router(
    responses: dict[str, list[str]],
    model: str = "openai/gpt-4o",
) -> Any:
    """Return an async side-effect that dispatches responses by agent name in system prompt.

    Each agent has a list of responses consumed in order.
    """
    counters: dict[str, int] = {k: 0 for k in responses}

    async def _router(**kwargs: Any) -> MagicMock:
        messages = kwargs.get("messages", [])
        system_text = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
                break
        for key, texts in responses.items():
            if key in system_text:
                idx = counters.get(key, 0)
                counters[key] = idx + 1
                text = texts[idx] if idx < len(texts) else texts[-1]
                return make_mock_litellm_response(content=text, model=model)
        return make_mock_litellm_response(content="fallback", model=model)

    return _router


def _make_agent(name: str) -> AgentNode:
    manifest = AgentManifest(name=name, system_prompt_template=f"You are {name}.")
    config = ModelConfig(model="openai/gpt-4o")
    client = ModelClient(config)
    return AgentNode(manifest=manifest, client=client)


class TestStarE2E:
    @pytest.mark.asyncio
    async def test_delegation_order(self) -> None:
        """Supervisor routes to coder, then tester, then says DONE."""
        agents = {
            "supervisor": _make_agent("supervisor"),
            "coder": _make_agent("coder"),
            "tester": _make_agent("tester"),
        }
        responses = {
            "supervisor": ["Route: coder", "Route: tester", "DONE"],
            "coder": ["def hello(): pass"],
            "tester": ["All tests pass."],
        }

        with patch("uac.core.interface.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_make_response_router(responses))

            orch = StarOrchestrator(
                agents=agents,
                supervisor="supervisor",
            )
            board = await orch.run("Implement and test a function")

        # Extract agent execution order from generate-action trace entries
        gen_agents = [e.agent_id for e in board.execution_trace if e.action == "generate"]
        assert gen_agents == ["supervisor", "coder", "supervisor", "tester", "supervisor"]

    @pytest.mark.asyncio
    async def test_termination_trace(self) -> None:
        """DONE directive produces a terminate trace entry."""
        agents = {
            "supervisor": _make_agent("supervisor"),
            "coder": _make_agent("coder"),
        }
        responses = {
            "supervisor": ["Route: coder", "DONE"],
            "coder": ["Code written."],
        }

        with patch("uac.core.interface.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_make_response_router(responses))

            orch = StarOrchestrator(agents=agents, supervisor="supervisor")
            board = await orch.run("Write code")

        term = [e for e in board.execution_trace if e.action == "terminate"]
        assert len(term) == 1
        assert term[0].agent_id == "supervisor"
        assert term[0].data.get("reason") == "DONE"

    @pytest.mark.asyncio
    async def test_immediate_done(self) -> None:
        """Supervisor says DONE immediately — no workers are invoked."""
        agents = {
            "supervisor": _make_agent("supervisor"),
            "worker": _make_agent("worker"),
        }
        responses = {
            "supervisor": ["DONE"],
            "worker": ["Should not run."],
        }

        with patch("uac.core.interface.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_make_response_router(responses))

            orch = StarOrchestrator(agents=agents, supervisor="supervisor")
            board = await orch.run("Quick exit")

        gen_agents = [e.agent_id for e in board.execution_trace if e.action == "generate"]
        assert gen_agents == ["supervisor"]

    @pytest.mark.asyncio
    async def test_max_iterations_cap(self) -> None:
        """Orchestrator stops at max_iterations even if supervisor never says DONE."""
        agents = {
            "supervisor": _make_agent("supervisor"),
            "worker": _make_agent("worker"),
        }
        responses = {
            "supervisor": ["Route: worker"] * 20,
            "worker": ["working..."] * 20,
        }

        with patch("uac.core.interface.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_make_response_router(responses))

            orch = StarOrchestrator(
                agents=agents,
                supervisor="supervisor",
                max_iterations=4,
            )
            board = await orch.run("Capped run")

        gen_agents = [e.agent_id for e in board.execution_trace if e.action == "generate"]
        assert len(gen_agents) <= 4

    @pytest.mark.asyncio
    async def test_worker_artifacts_in_blackboard(self) -> None:
        """Worker responses appear as artifacts on the blackboard."""
        agents = {
            "supervisor": _make_agent("supervisor"),
            "coder": _make_agent("coder"),
        }
        responses = {
            "supervisor": ["Route: coder", "DONE"],
            "coder": ["def add(a, b): return a + b"],
        }

        with patch("uac.core.interface.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_make_response_router(responses))

            orch = StarOrchestrator(agents=agents, supervisor="supervisor")
            board = await orch.run("Write an add function")

        last_response = board.artifacts.get("last_response", {})
        assert "coder" in last_response
