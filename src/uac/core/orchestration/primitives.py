"""Orchestration primitives — AgentNode and Orchestrator base class.

``AgentNode`` wraps a manifest + ModelClient + tools into a callable unit.
``Orchestrator`` provides the execution loop that subclasses (Pipeline,
Star, Mesh) customise with their own topology logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from uac.core.blackboard.blackboard import Blackboard
from uac.core.blackboard.models import ContextSlice, StateDelta, TraceEntry
from uac.core.interface.client import ModelClient  # noqa: TC001
from uac.core.interface.models import CanonicalMessage, ConversationHistory
from uac.core.orchestration.manifest import render_prompt
from uac.core.orchestration.models import AgentManifest  # noqa: TC001


class AgentNode:
    """An executable agent: manifest + model client + optional tools.

    Each call to :meth:`step` sends a context slice to the model and
    returns a :class:`StateDelta` to apply to the blackboard.
    """

    def __init__(
        self,
        manifest: AgentManifest,
        client: ModelClient,
        tools: list[dict[str, Any]] | None = None,
        *,
        prompt_variables: dict[str, Any] | None = None,
    ) -> None:
        self.manifest = manifest
        self.client = client
        self.tools = tools
        self._system_prompt = render_prompt(manifest, **(prompt_variables or {}))

    @property
    def name(self) -> str:
        return self.manifest.name

    async def step(self, context: ContextSlice) -> StateDelta:
        """Execute a single reasoning step.

        Builds a conversation from the system prompt and the context slice,
        calls the model, and returns a delta to apply to the blackboard.
        """
        history = self._build_history(context)
        response = await self.client.generate(history, tools=self.tools)
        return self._response_to_delta(response)

    def _build_history(self, context: ContextSlice) -> ConversationHistory:
        """Construct a ConversationHistory from the system prompt and context slice."""
        messages: list[CanonicalMessage] = [CanonicalMessage.system(self._system_prompt)]

        # Feed the current blackboard state as a user message
        state_text = self._format_context(context)
        if state_text:
            messages.append(CanonicalMessage.user(state_text))

        return ConversationHistory(messages=messages)

    def _format_context(self, context: ContextSlice) -> str:
        """Format a ContextSlice into a human-readable prompt section."""
        parts: list[str] = []

        if context.belief_state:
            parts.append(f"Current state: {context.belief_state}")

        if context.pending_tasks:
            task_lines = [
                f"- {t.description} (priority {t.priority})" for t in context.pending_tasks
            ]
            parts.append("Pending tasks:\n" + "\n".join(task_lines))

        if context.artifacts:
            import json

            parts.append(f"Artifacts: {json.dumps(context.artifacts, default=str)}")

        if context.trace:
            recent = context.trace[-5:]
            trace_lines = [f"- [{e.agent_id}] {e.action}" for e in recent]
            parts.append("Recent trace:\n" + "\n".join(trace_lines))

        return "\n\n".join(parts)

    def _response_to_delta(self, response: CanonicalMessage) -> StateDelta:
        """Convert a model response into a StateDelta."""
        trace = TraceEntry(
            agent_id=self.name,
            action="generate",
            data={"text": response.text, "has_tool_calls": response.tool_calls is not None},
        )
        return StateDelta(
            trace_entries=[trace],
            artifacts={"last_response": {self.name: response.text}},
        )


class Orchestrator(ABC):
    """Base class for topology-specific orchestrators.

    Subclasses implement :meth:`select_agent` and :meth:`is_done` to
    define the execution topology.
    """

    def __init__(
        self,
        agents: dict[str, AgentNode],
        blackboard: Blackboard | None = None,
        *,
        max_iterations: int = 20,
    ) -> None:
        self.agents = agents
        self.blackboard = blackboard or Blackboard()
        self.max_iterations = max_iterations

    async def run(self, goal: str) -> Blackboard:
        """Execute the orchestration loop until done or max iterations.

        Args:
            goal: The high-level objective for this orchestration run.

        Returns:
            The final blackboard state.
        """
        self.blackboard.apply(StateDelta(belief_state=goal))

        for iteration in range(self.max_iterations):
            agent = await self.select_agent(iteration)
            if agent is None:
                break

            context = self.blackboard.slice(agent_id=agent.name)
            delta = await agent.step(context)
            self.blackboard.apply(delta)

            if await self.is_done(iteration):
                break

        return self.blackboard

    @abstractmethod
    async def select_agent(self, iteration: int) -> AgentNode | None:
        """Choose the next agent to execute, or ``None`` to stop."""
        ...

    @abstractmethod
    async def is_done(self, iteration: int) -> bool:
        """Return ``True`` when the orchestration should terminate."""
        ...
