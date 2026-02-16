"""Tests for AgentNode and Orchestrator base class."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from uac.core.blackboard.blackboard import Blackboard
from uac.core.blackboard.models import ContextSlice, StateDelta
from uac.core.interface.models import CanonicalMessage
from uac.core.orchestration.models import AgentManifest
from uac.core.orchestration.primitives import AgentNode, Orchestrator


def _make_manifest(name: str = "test-agent") -> AgentManifest:
    return AgentManifest(
        name=name,
        system_prompt_template="You are $name.",
    )


def _make_client(response_text: str = "Hello") -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(return_value=CanonicalMessage.assistant(response_text))
    return client


def _make_context() -> ContextSlice:
    return ContextSlice(
        belief_state="testing",
        trace=[],
        artifacts={},
        pending_tasks=[],
    )


class TestAgentNode:
    def test_name_from_manifest(self) -> None:
        node = AgentNode(manifest=_make_manifest("my-agent"), client=_make_client())
        assert node.name == "my-agent"

    async def test_step_calls_model(self) -> None:
        client = _make_client("Generated response")
        node = AgentNode(manifest=_make_manifest(), client=client)

        delta = await node.step(_make_context())

        client.generate.assert_awaited_once()
        assert len(delta.trace_entries) == 1
        assert delta.trace_entries[0].agent_id == "test-agent"
        assert delta.trace_entries[0].action == "generate"

    async def test_step_returns_delta_with_response(self) -> None:
        node = AgentNode(manifest=_make_manifest(), client=_make_client("Result text"))
        delta = await node.step(_make_context())

        assert "last_response" in delta.artifacts
        assert delta.artifacts["last_response"]["test-agent"] == "Result text"

    async def test_prompt_variables(self) -> None:
        manifest = AgentManifest(
            name="agent",
            system_prompt_template="$name with $extra",
        )
        client = _make_client()
        node = AgentNode(
            manifest=manifest,
            client=client,
            prompt_variables={"extra": "bonus"},
        )
        assert node._system_prompt == "agent with bonus"

    async def test_context_formatting_with_state(self) -> None:
        node = AgentNode(manifest=_make_manifest(), client=_make_client())
        context = ContextSlice(
            belief_state="planning",
            trace=[],
            artifacts={"key": "value"},
            pending_tasks=[],
        )
        text = node._format_context(context)
        assert "planning" in text
        assert "key" in text

    async def test_context_formatting_empty(self) -> None:
        node = AgentNode(manifest=_make_manifest(), client=_make_client())
        context = ContextSlice(
            belief_state="",
            trace=[],
            artifacts={},
            pending_tasks=[],
        )
        text = node._format_context(context)
        assert text == ""

    async def test_step_passes_tools(self) -> None:
        tools = [{"type": "function", "function": {"name": "search"}}]
        client = _make_client()
        node = AgentNode(manifest=_make_manifest(), client=client, tools=tools)
        await node.step(_make_context())
        _, kwargs = client.generate.call_args
        assert kwargs["tools"] == tools


class _SimpleOrchestrator(Orchestrator):
    """Minimal orchestrator for testing the base class."""

    def __init__(self, *args, steps: int = 2, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._steps = steps
        self._current = 0

    async def select_agent(self, iteration: int) -> AgentNode | None:
        if self._current >= self._steps:
            return None
        self._current += 1
        first_agent = next(iter(self.agents.values()))
        return first_agent

    async def is_done(self, iteration: int) -> bool:
        return self._current >= self._steps


class TestOrchestrator:
    async def test_run_sets_goal(self) -> None:
        node = AgentNode(manifest=_make_manifest(), client=_make_client())
        orch = _SimpleOrchestrator(agents={"test": node}, steps=0)
        board = await orch.run("Test goal")
        assert board.belief_state == "Test goal"

    async def test_run_executes_steps(self) -> None:
        client = _make_client()
        node = AgentNode(manifest=_make_manifest(), client=client)
        orch = _SimpleOrchestrator(agents={"test": node}, steps=3)
        board = await orch.run("Run three steps")
        assert client.generate.await_count == 3
        assert len(board.execution_trace) >= 3

    async def test_run_respects_max_iterations(self) -> None:
        client = _make_client()
        node = AgentNode(manifest=_make_manifest(), client=client)
        orch = _SimpleOrchestrator(agents={"test": node}, steps=100, max_iterations=5)
        await orch.run("Capped run")
        # max_iterations=5 but is_done fires at steps=100, so cap applies
        assert client.generate.await_count == 5

    async def test_run_returns_blackboard(self) -> None:
        board = Blackboard()
        board.set_artifact("pre", "existing")
        node = AgentNode(manifest=_make_manifest(), client=_make_client())
        orch = _SimpleOrchestrator(agents={"test": node}, blackboard=board, steps=1)
        result = await orch.run("Test")
        assert result is board
        assert result.get_artifact("pre") == "existing"

    async def test_run_stops_when_select_returns_none(self) -> None:
        client = _make_client()
        node = AgentNode(manifest=_make_manifest(), client=client)
        orch = _SimpleOrchestrator(agents={"test": node}, steps=0)
        await orch.run("No steps")
        assert client.generate.await_count == 0
