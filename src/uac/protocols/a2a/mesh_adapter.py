"""A2AAgentNode — adapts a remote A2A agent to work with MeshOrchestrator.

This module provides a duck-type compatible stand-in for
:class:`~uac.core.orchestration.primitives.AgentNode` so that remote
A2A agents can participate in mesh topology orchestration.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from uac.core.blackboard.models import ContextSlice, StateDelta, TraceEntry
from uac.core.interface.models import TextContent

if TYPE_CHECKING:
    from uac.protocols.a2a.client import A2AClient


class A2AAgentNode:
    """A remote agent that participates in mesh orchestration via A2A.

    Duck-type compatible with :class:`AgentNode` — exposes ``.name``
    and an async ``.step(context)`` method.
    """

    def __init__(self, name: str, client: A2AClient, *, skill_id: str | None = None) -> None:
        self.name = name
        self._client = client
        self._skill_id = skill_id or name

    async def step(self, context: ContextSlice) -> StateDelta:
        """Send the belief state to the remote agent and wrap the response."""
        message = context.belief_state
        if context.artifacts:
            message += "\n\nArtifacts: " + json.dumps(context.artifacts, default=str)

        result = await self._client.execute_tool(
            self._skill_id, {"message": message}
        )

        # Extract text from the ToolResult
        text = "".join(
            part.text for part in result.content if isinstance(part, TextContent)
        )

        trace = TraceEntry(
            agent_id=self.name,
            action="a2a_delegate",
            data={"skill": self._skill_id, "response_length": len(text)},
        )
        return StateDelta(
            trace_entries=[trace],
            artifacts={"last_response": {self.name: text}},
        )
