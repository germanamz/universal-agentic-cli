"""Pipeline topology — sequential agent execution.

Agents are executed in a fixed order.  The output of agent N is
available to agent N+1 via the blackboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from uac.core.orchestration.primitives import AgentNode, Orchestrator

if TYPE_CHECKING:
    from uac.core.blackboard.blackboard import Blackboard


class PipelineOrchestrator(Orchestrator):
    """Sequential pipeline: agents execute in declared order, one pass.

    Usage::

        orchestrator = PipelineOrchestrator(
            agents={"a": node_a, "b": node_b, "c": node_c},
            order=["a", "b", "c"],
        )
        result = await orchestrator.run("Process the document")
    """

    def __init__(
        self,
        agents: dict[str, AgentNode],
        order: list[str],
        blackboard: Blackboard | None = None,
        *,
        max_iterations: int = 100,
    ) -> None:
        super().__init__(agents, blackboard, max_iterations=max_iterations)
        self.order = order
        self._step = 0

    async def select_agent(self, iteration: int) -> AgentNode | None:
        if self._step >= len(self.order):
            return None
        name = self.order[self._step]
        self._step += 1
        return self.agents[name]

    async def is_done(self, iteration: int) -> bool:
        return self._step >= len(self.order)
