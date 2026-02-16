"""ContextSlicer — produces filtered views of a Blackboard for individual agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from uac.core.blackboard.models import ContextSlice

if TYPE_CHECKING:
    from uac.core.blackboard.blackboard import Blackboard


class ContextSlicer:
    """Produces a :class:`ContextSlice` from a :class:`Blackboard`.

    Filtering options:

    * **agent_id** — when set, only trace entries from that agent are included.
    * **artifact_keys** — when set, only the named artifacts are included.
    * **max_trace_entries** — caps the number of (most recent) trace entries.
    """

    def slice(
        self,
        board: Blackboard,
        *,
        agent_id: str | None = None,
        artifact_keys: list[str] | None = None,
        max_trace_entries: int = 50,
    ) -> ContextSlice:
        """Return a filtered snapshot of *board*."""
        trace = list(board.execution_trace)
        if agent_id is not None:
            trace = [e for e in trace if e.agent_id == agent_id]
        trace = trace[-max_trace_entries:]

        if artifact_keys is not None:
            artifacts = {k: v for k, v in board.artifacts.items() if k in artifact_keys}
        else:
            artifacts = dict(board.artifacts)

        return ContextSlice(
            belief_state=board.belief_state,
            trace=trace,
            artifacts=artifacts,
            pending_tasks=list(board.pending_tasks),
        )
