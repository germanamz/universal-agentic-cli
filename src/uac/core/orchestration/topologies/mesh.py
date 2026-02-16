"""Mesh topology — event-driven agent activation.

Agents subscribe to topic patterns.  An internal event bus broadcasts
messages; agents whose patterns match the topic are activated.  Built
on :mod:`asyncio.Queue`.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, cast

from uac.core.blackboard.blackboard import Blackboard
from uac.core.blackboard.models import StateDelta, TraceEntry
from uac.core.orchestration.primitives import AgentNode  # noqa: TC001


@dataclass
class Event:
    """A message published to the event bus."""

    topic: str
    payload: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    source: str = ""


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a glob-style topic pattern to a compiled regex.

    Supports ``*`` (single segment) and ``**`` (any number of segments).
    """
    # Escape dots, then convert glob wildcards
    escaped = re.escape(pattern)
    escaped = escaped.replace(r"\*\*", ".*")
    escaped = escaped.replace(r"\*", r"[^.]*")
    return re.compile(f"^{escaped}$")


class EventBus:
    """Lightweight pub/sub bus backed by an :class:`asyncio.Queue`."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)

    async def get(self) -> Event | None:
        return await self._queue.get()

    def get_nowait(self) -> Event | None:
        return self._queue.get_nowait()

    async def close(self) -> None:
        await self._queue.put(None)


class MeshOrchestrator:
    """Event-driven mesh orchestrator.

    Agents declare topic subscriptions.  When an event is published,
    all agents whose subscription pattern matches the topic are
    activated concurrently.  Each agent's response may publish new
    events, continuing the cycle.

    Usage::

        orchestrator = MeshOrchestrator(
            agents={"parser": node_p, "analyser": node_a, "formatter": node_f},
            subscriptions={
                "parser": ["input.*"],
                "analyser": ["parsed.**"],
                "formatter": ["analysis.complete"],
            },
        )
        result = await orchestrator.run("Process the data")
    """

    def __init__(
        self,
        agents: dict[str, AgentNode],
        subscriptions: dict[str, list[str]],
        blackboard: Blackboard | None = None,
        *,
        max_iterations: int = 50,
    ) -> None:
        self.agents = agents
        self.blackboard = blackboard or Blackboard()
        self.max_iterations = max_iterations
        self.bus = EventBus()

        # Pre-compile subscription patterns
        self._patterns: dict[str, list[re.Pattern[str]]] = {}
        for agent_name, patterns in subscriptions.items():
            self._patterns[agent_name] = [_glob_to_regex(p) for p in patterns]

    async def run(self, goal: str) -> Blackboard:
        """Execute the mesh orchestration loop.

        Publishes an initial ``orchestration.start`` event and then
        processes the event queue until it drains or max iterations.
        """
        self.blackboard.apply(StateDelta(belief_state=goal))

        # Seed the bus with a start event
        await self.bus.publish(Event(topic="orchestration.start", payload={"goal": goal}))

        for _iteration in range(self.max_iterations):
            event = await self._try_get_event()
            if event is None:
                break

            matching = self._match_agents(event)
            if not matching:
                # No subscribers — event is consumed silently
                continue

            # Activate matching agents concurrently
            results = await asyncio.gather(
                *[self._activate_agent(agent, event) for agent in matching]
            )

            # Publish any follow-up events from agent responses
            for follow_ups in results:
                for evt in follow_ups:
                    await self.bus.publish(evt)

        return self.blackboard

    async def _try_get_event(self) -> Event | None:
        """Non-blocking attempt to get the next event."""
        try:
            return self.bus.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _match_agents(self, event: Event) -> list[AgentNode]:
        """Return agents whose subscription patterns match the event topic."""
        matched: list[AgentNode] = []
        for agent_name, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.match(event.topic):
                    if agent_name in self.agents:
                        matched.append(self.agents[agent_name])
                    break
        return matched

    async def _activate_agent(self, agent: AgentNode, event: Event) -> list[Event]:
        """Run a single agent step and return any follow-up events."""
        context = self.blackboard.slice(agent_id=agent.name)
        delta = await agent.step(context)
        self.blackboard.apply(delta)

        # Record the activation on the trace
        self.blackboard.apply(
            StateDelta(
                trace_entries=[
                    TraceEntry(
                        agent_id=agent.name,
                        action="mesh_activate",
                        data={"trigger_topic": event.topic},
                    )
                ]
            )
        )

        # Check if the agent's response signals a new event to publish
        return self._extract_events(agent.name, delta)

    def _extract_events(self, agent_name: str, delta: StateDelta) -> list[Event]:
        """Extract follow-up events from a StateDelta.

        Convention: if the delta sets an artifact key ``_publish``, its
        value is treated as a list of ``{"topic": ..., "payload": ...}``
        dicts that become new events.
        """
        events: list[Event] = []
        publish_data = delta.artifacts.get("_publish")
        if not isinstance(publish_data, list):
            return events
        for raw in cast("list[dict[str, Any]]", publish_data):
            if "topic" in raw:
                topic = cast("str", raw["topic"])
                payload = cast("dict[str, Any]", raw.get("payload", {}))
                events.append(Event(topic=topic, payload=payload, source=agent_name))
        return events
