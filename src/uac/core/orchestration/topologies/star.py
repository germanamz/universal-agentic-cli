"""Star topology — supervisor-directed agent execution.

A designated supervisor agent inspects the blackboard each iteration
and emits a routing directive (``Route: <agent_name>``) or ``DONE``
to terminate.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from uac.core.blackboard.models import StateDelta, TraceEntry
from uac.core.orchestration.primitives import AgentNode, Orchestrator

if TYPE_CHECKING:
    from uac.core.blackboard.blackboard import Blackboard

_ROUTE_PATTERN = re.compile(r"Route:\s*(\S+)", re.IGNORECASE)
_DONE_PATTERN = re.compile(r"\bDONE\b", re.IGNORECASE)


class StarOrchestrator(Orchestrator):
    """Supervisor-directed star topology.

    Each iteration:

    1. The supervisor agent inspects the blackboard.
    2. If the supervisor emits ``DONE``, orchestration ends.
    3. If the supervisor emits ``Route: <name>``, that worker runs next.
    4. If no directive is found, orchestration ends (ambiguous output).

    Usage::

        orchestrator = StarOrchestrator(
            agents={"supervisor": sup, "writer": w, "reviewer": r},
            supervisor="supervisor",
        )
        result = await orchestrator.run("Write and review an essay")
    """

    def __init__(
        self,
        agents: dict[str, AgentNode],
        supervisor: str,
        blackboard: Blackboard | None = None,
        *,
        max_iterations: int = 20,
    ) -> None:
        super().__init__(agents, blackboard, max_iterations=max_iterations)
        self.supervisor_name = supervisor
        self._done = False
        self._next_worker: str | None = None
        self._phase: str = "supervisor"  # alternates: "supervisor" / "worker"

    async def select_agent(self, iteration: int) -> AgentNode | None:
        if self._done:
            return None

        if self._phase == "supervisor":
            return self.agents[self.supervisor_name]

        # Worker phase
        if self._next_worker and self._next_worker in self.agents:
            return self.agents[self._next_worker]

        # Unknown worker — stop
        self._done = True
        return None

    async def is_done(self, iteration: int) -> bool:
        if self._done:
            return True

        # After a step, inspect the last trace entry to determine what happened
        if not self.blackboard.execution_trace:
            return False

        last_trace = self.blackboard.execution_trace[-1]

        if self._phase == "supervisor":
            # Parse the supervisor's output for routing directives
            response_text = last_trace.data.get("text", "")
            return self._parse_supervisor_output(response_text)

        # After a worker step, switch back to supervisor
        self._phase = "supervisor"
        self._next_worker = None
        return False

    def _parse_supervisor_output(self, text: str) -> bool:
        """Parse supervisor output for ``DONE`` or ``Route: <name>``."""
        if _DONE_PATTERN.search(text):
            self._done = True
            self.blackboard.apply(
                StateDelta(
                    trace_entries=[
                        TraceEntry(
                            agent_id=self.supervisor_name,
                            action="terminate",
                            data={"reason": "DONE"},
                        )
                    ]
                )
            )
            return True

        match = _ROUTE_PATTERN.search(text)
        if match:
            self._next_worker = match.group(1)
            self._phase = "worker"
            return False

        # No recognisable directive — stop to avoid infinite loops
        self._done = True
        return True
