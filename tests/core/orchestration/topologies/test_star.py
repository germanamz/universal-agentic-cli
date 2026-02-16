"""Tests for StarOrchestrator."""

from unittest.mock import AsyncMock, MagicMock

from uac.core.interface.models import CanonicalMessage
from uac.core.orchestration.models import AgentManifest
from uac.core.orchestration.primitives import AgentNode
from uac.core.orchestration.topologies.star import StarOrchestrator


def _make_node(name: str, responses: list[str] | None = None) -> AgentNode:
    manifest = AgentManifest(name=name, system_prompt_template="You are $name.")
    client = MagicMock()
    if responses:
        client.generate = AsyncMock(
            side_effect=[CanonicalMessage.assistant(r) for r in responses]
        )
    else:
        client.generate = AsyncMock(return_value=CanonicalMessage.assistant("ok"))
    return AgentNode(manifest=manifest, client=client)


class TestStarOrchestrator:
    async def test_supervisor_routes_to_worker(self) -> None:
        sup = _make_node("supervisor", ["Route: writer", "DONE"])
        writer = _make_node("writer", ["Written content"])

        orch = StarOrchestrator(
            agents={"supervisor": sup, "writer": writer},
            supervisor="supervisor",
        )
        board = await orch.run("Write something")

        # Supervisor called twice (route + done), writer called once
        assert sup.client.generate.await_count == 2
        writer.client.generate.assert_awaited_once()

    async def test_supervisor_done_immediately(self) -> None:
        sup = _make_node("supervisor", ["DONE"])
        worker = _make_node("worker")

        orch = StarOrchestrator(
            agents={"supervisor": sup, "worker": worker},
            supervisor="supervisor",
        )
        board = await orch.run("Quick stop")

        sup.client.generate.assert_awaited_once()
        worker.client.generate.assert_not_awaited()

        # Check termination trace
        term_traces = [e for e in board.execution_trace if e.action == "terminate"]
        assert len(term_traces) == 1

    async def test_supervisor_no_directive_stops(self) -> None:
        sup = _make_node("supervisor", ["I don't know what to do"])

        orch = StarOrchestrator(
            agents={"supervisor": sup},
            supervisor="supervisor",
        )
        board = await orch.run("Ambiguous")
        sup.client.generate.assert_awaited_once()

    async def test_route_case_insensitive(self) -> None:
        sup = _make_node("supervisor", ["route: myworker", "DONE"])
        worker = _make_node("myworker", ["result"])

        orch = StarOrchestrator(
            agents={"supervisor": sup, "myworker": worker},
            supervisor="supervisor",
        )
        await orch.run("Test case insensitivity")
        worker.client.generate.assert_awaited_once()

    async def test_done_case_insensitive(self) -> None:
        sup = _make_node("supervisor", ["done"])

        orch = StarOrchestrator(
            agents={"supervisor": sup},
            supervisor="supervisor",
        )
        await orch.run("Case insensitive done")
        sup.client.generate.assert_awaited_once()

    async def test_max_iterations(self) -> None:
        # Supervisor always routes, never says DONE
        responses = ["Route: worker"] * 20
        sup = _make_node("supervisor", responses)
        worker = _make_node("worker")

        orch = StarOrchestrator(
            agents={"supervisor": sup, "worker": worker},
            supervisor="supervisor",
            max_iterations=4,
        )
        await orch.run("Capped")
        # Should be capped at 4 total iterations
        total_calls = sup.client.generate.await_count + worker.client.generate.await_count
        assert total_calls <= 4

    async def test_multiple_rounds(self) -> None:
        sup = _make_node("supervisor", [
            "Route: agent_a",
            "Route: agent_b",
            "DONE",
        ])
        a = _make_node("agent_a", ["a-result"])
        b = _make_node("agent_b", ["b-result"])

        orch = StarOrchestrator(
            agents={"supervisor": sup, "agent_a": a, "agent_b": b},
            supervisor="supervisor",
        )
        await orch.run("Multi-round")

        a.client.generate.assert_awaited_once()
        b.client.generate.assert_awaited_once()
        assert sup.client.generate.await_count == 3
