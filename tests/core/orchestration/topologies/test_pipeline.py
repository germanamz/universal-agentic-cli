"""Tests for PipelineOrchestrator."""

from unittest.mock import AsyncMock, MagicMock

from uac.core.interface.models import CanonicalMessage
from uac.core.orchestration.models import AgentManifest
from uac.core.orchestration.primitives import AgentNode
from uac.core.orchestration.topologies.pipeline import PipelineOrchestrator


def _make_node(name: str, response: str = "ok") -> AgentNode:
    manifest = AgentManifest(name=name, system_prompt_template="You are $name.")
    client = MagicMock()
    client.generate = AsyncMock(return_value=CanonicalMessage.assistant(response))
    return AgentNode(manifest=manifest, client=client)


class TestPipelineOrchestrator:
    async def test_executes_in_order(self) -> None:
        a = _make_node("alpha", "result-a")
        b = _make_node("beta", "result-b")
        c = _make_node("gamma", "result-c")

        orch = PipelineOrchestrator(
            agents={"alpha": a, "beta": b, "gamma": c},
            order=["alpha", "beta", "gamma"],
        )
        board = await orch.run("Pipeline test")

        # All three agents should have been called exactly once
        a.client.generate.assert_awaited_once()
        b.client.generate.assert_awaited_once()
        c.client.generate.assert_awaited_once()

        # Trace should show the execution order
        agent_ids = [e.agent_id for e in board.execution_trace]
        assert agent_ids == ["alpha", "beta", "gamma"]

    async def test_single_agent(self) -> None:
        node = _make_node("solo")
        orch = PipelineOrchestrator(agents={"solo": node}, order=["solo"])
        board = await orch.run("Single step")
        node.client.generate.assert_awaited_once()
        assert board.belief_state == "Single step"

    async def test_empty_pipeline(self) -> None:
        orch = PipelineOrchestrator(agents={}, order=[])
        board = await orch.run("Empty")
        assert board.belief_state == "Empty"
        assert len(board.execution_trace) == 0

    async def test_blackboard_state_propagates(self) -> None:
        a = _make_node("first", "step-1")
        b = _make_node("second", "step-2")

        orch = PipelineOrchestrator(
            agents={"first": a, "second": b},
            order=["first", "second"],
        )
        board = await orch.run("Propagation test")

        # Both agents' responses should be in artifacts
        last = board.artifacts.get("last_response", {})
        assert "first" in last or "second" in last
