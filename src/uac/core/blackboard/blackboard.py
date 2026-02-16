"""Core Blackboard — shared mutable state store for multi-agent coordination.

The Blackboard holds belief state, an execution trace, artifacts, and a
prioritised task queue.  Agents read from and write to the Blackboard via
:class:`StateDelta` objects, keeping orchestration logic decoupled from
individual LLM context windows.
"""

from typing import Any, cast

from pydantic import BaseModel

from uac.core.blackboard.models import ContextSlice, StateDelta, TaskItem, TraceEntry


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *updates* into *base* (mutates *base*).

    * Dict values are merged recursively.
    * A value of ``None`` in *updates* deletes the corresponding key.
    * All other values overwrite.
    """
    for key, value in updates.items():
        if value is None:
            base.pop(key, None)
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], cast("dict[str, Any]", value))
        else:
            base[key] = value
    return base


class Blackboard(BaseModel):
    """Shared state store for a single orchestration session."""

    belief_state: str = ""
    execution_trace: list[TraceEntry] = []
    artifacts: dict[str, Any] = {}
    pending_tasks: list[TaskItem] = []

    # ------------------------------------------------------------------
    # Core mutation
    # ------------------------------------------------------------------

    def apply(self, delta: StateDelta) -> "Blackboard":
        """Apply a :class:`StateDelta` and return *self* for chaining.

        * ``belief_state`` — overwritten if ``delta.belief_state`` is not None.
        * ``execution_trace`` — ``delta.trace_entries`` are appended.
        * ``artifacts`` — deep-merged (``None`` values delete keys).
        * ``pending_tasks`` — ``delta.add_tasks`` appended,
          ``delta.remove_task_ids`` removed, then re-sorted by priority.
        """
        if delta.belief_state is not None:
            self.belief_state = delta.belief_state

        self.execution_trace.extend(delta.trace_entries)

        if delta.artifacts:
            _deep_merge(self.artifacts, delta.artifacts)

        if delta.add_tasks:
            self.pending_tasks.extend(delta.add_tasks)

        if delta.remove_task_ids:
            remove_set = set(delta.remove_task_ids)
            self.pending_tasks = [t for t in self.pending_tasks if t.id not in remove_set]

        if delta.add_tasks or delta.remove_task_ids:
            self.pending_tasks.sort(key=lambda t: t.priority)

        return self

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> bytes:
        """Serialise the entire board to JSON bytes."""
        return self.model_dump_json().encode()

    @classmethod
    def restore(cls, data: bytes) -> "Blackboard":
        """Deserialise a board from JSON bytes produced by :meth:`snapshot`."""
        return cls.model_validate_json(data)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_artifact(self, key: str, default: Any = None) -> Any:
        """Return a single artifact value, or *default* if absent."""
        return self.artifacts.get(key, default)

    def set_artifact(self, key: str, value: Any) -> None:
        """Set a single artifact value."""
        self.artifacts[key] = value

    def add_trace(self, agent_id: str, action: str, data: dict[str, Any] | None = None) -> None:
        """Append a single trace entry."""
        self.execution_trace.append(
            TraceEntry(agent_id=agent_id, action=action, data=data or {})
        )

    def pop_task(self) -> TaskItem | None:
        """Remove and return the highest-priority (lowest value) task, or ``None``."""
        if not self.pending_tasks:
            return None
        return self.pending_tasks.pop(0)

    # ------------------------------------------------------------------
    # Slicing (convenience shortcut)
    # ------------------------------------------------------------------

    def slice(
        self,
        *,
        agent_id: str | None = None,
        artifact_keys: list[str] | None = None,
        max_trace_entries: int = 50,
    ) -> ContextSlice:
        """Return a filtered :class:`ContextSlice` of this board.

        Delegates to :class:`~uac.core.blackboard.slicer.ContextSlicer`.
        """
        from uac.core.blackboard.slicer import ContextSlicer

        return ContextSlicer().slice(
            self,
            agent_id=agent_id,
            artifact_keys=artifact_keys,
            max_trace_entries=max_trace_entries,
        )
