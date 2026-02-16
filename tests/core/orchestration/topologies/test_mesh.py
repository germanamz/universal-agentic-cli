"""Tests for MeshOrchestrator."""

from unittest.mock import AsyncMock, MagicMock

from uac.core.blackboard.models import StateDelta
from uac.core.interface.models import CanonicalMessage
from uac.core.orchestration.models import AgentManifest
from uac.core.orchestration.primitives import AgentNode
from uac.core.orchestration.topologies.mesh import (
    Event,
    EventBus,
    MeshOrchestrator,
    _glob_to_regex,
)


def _make_node(name: str, response: str = "ok") -> AgentNode:
    manifest = AgentManifest(name=name, system_prompt_template="You are $name.")
    client = MagicMock()
    client.generate = AsyncMock(return_value=CanonicalMessage.assistant(response))
    return AgentNode(manifest=manifest, client=client)


class TestGlobToRegex:
    def test_exact_match(self) -> None:
        pattern = _glob_to_regex("input.ready")
        assert pattern.match("input.ready")
        assert not pattern.match("input.notready")

    def test_single_wildcard(self) -> None:
        pattern = _glob_to_regex("input.*")
        assert pattern.match("input.ready")
        assert pattern.match("input.data")
        assert not pattern.match("input.a.b")

    def test_double_wildcard(self) -> None:
        pattern = _glob_to_regex("events.**")
        assert pattern.match("events.a")
        assert pattern.match("events.a.b.c")

    def test_no_match(self) -> None:
        pattern = _glob_to_regex("specific.topic")
        assert not pattern.match("other.topic")


class TestEventBus:
    async def test_publish_and_get(self) -> None:
        bus = EventBus()
        event = Event(topic="test.topic", payload={"key": "value"})
        await bus.publish(event)
        result = await bus.get()
        assert result is not None
        assert result.topic == "test.topic"
        assert result.payload["key"] == "value"

    async def test_close_sends_none(self) -> None:
        bus = EventBus()
        await bus.close()
        result = await bus.get()
        assert result is None

    async def test_get_nowait_empty(self) -> None:
        bus = EventBus()
        import asyncio

        with __import__("pytest").raises(asyncio.QueueEmpty):
            bus.get_nowait()


class TestMeshOrchestrator:
    async def test_start_event_triggers_subscriber(self) -> None:
        node = _make_node("starter")
        orch = MeshOrchestrator(
            agents={"starter": node},
            subscriptions={"starter": ["orchestration.start"]},
        )
        board = await orch.run("Mesh test")
        node.client.generate.assert_awaited_once()
        assert board.belief_state == "Mesh test"

    async def test_no_subscribers_drains(self) -> None:
        node = _make_node("unrelated")
        orch = MeshOrchestrator(
            agents={"unrelated": node},
            subscriptions={"unrelated": ["other.topic"]},
        )
        board = await orch.run("No match")
        # Start event has no matching subscribers, so it's consumed silently
        # Then queue drains
        node.client.generate.assert_not_awaited()

    async def test_multiple_subscribers(self) -> None:
        a = _make_node("agent-a")
        b = _make_node("agent-b")
        orch = MeshOrchestrator(
            agents={"agent-a": a, "agent-b": b},
            subscriptions={
                "agent-a": ["orchestration.*"],
                "agent-b": ["orchestration.*"],
            },
        )
        await orch.run("Multi-sub")
        a.client.generate.assert_awaited_once()
        b.client.generate.assert_awaited_once()

    async def test_wildcard_subscription(self) -> None:
        node = _make_node("catcher")
        orch = MeshOrchestrator(
            agents={"catcher": node},
            subscriptions={"catcher": ["**"]},
        )
        await orch.run("Catch all")
        node.client.generate.assert_awaited()

    async def test_event_chain_via_publish(self) -> None:
        """Test that agents can publish follow-up events via _publish artifact."""
        # First agent publishes a follow-up event
        manifest_a = AgentManifest(name="producer", system_prompt_template="You are $name.")
        client_a = MagicMock()
        client_a.generate = AsyncMock(return_value=CanonicalMessage.assistant("producing"))
        node_a = AgentNode(manifest=manifest_a, client=client_a)

        # Override step to inject _publish artifact
        original_step = node_a.step

        async def step_with_publish(*args, **kwargs):  # type: ignore[no-untyped-def]
            delta = await original_step(*args, **kwargs)
            return StateDelta(
                trace_entries=delta.trace_entries,
                artifacts={
                    **delta.artifacts,
                    "_publish": [{"topic": "data.ready", "payload": {"data": "hello"}}],
                },
            )

        node_a.step = step_with_publish  # type: ignore[assignment]

        consumer = _make_node("consumer")

        orch = MeshOrchestrator(
            agents={"producer": node_a, "consumer": consumer},
            subscriptions={
                "producer": ["orchestration.start"],
                "consumer": ["data.ready"],
            },
        )
        await orch.run("Chain test")

        client_a.generate.assert_awaited_once()
        consumer.client.generate.assert_awaited_once()

    async def test_max_iterations(self) -> None:
        node = _make_node("looper")
        orch = MeshOrchestrator(
            agents={"looper": node},
            subscriptions={"looper": ["orchestration.*"]},
            max_iterations=1,
        )
        await orch.run("Capped")
        # Only one iteration
        node.client.generate.assert_awaited_once()
