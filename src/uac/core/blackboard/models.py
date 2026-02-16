"""Blackboard data models — shared state primitives for multi-agent coordination.

These models define the data structures stored on and exchanged via the
Blackboard: execution traces, pending tasks, partial updates (deltas),
and filtered views (context slices).
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceEntry(BaseModel):
    """A single execution trace entry recording an agent action."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent_id: str
    action: str
    data: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class TaskItem(BaseModel):
    """A pending task in the blackboard's priority queue."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    description: str
    priority: int = 0
    metadata: dict[str, Any] = {}


class StateDelta(BaseModel):
    """A partial update to apply to a Blackboard.

    Fields that are ``None`` or empty leave the corresponding Blackboard
    field unchanged.  Non-empty values are merged according to the rules
    documented on :pymethod:`Blackboard.apply`.
    """

    belief_state: str | None = None
    trace_entries: list[TraceEntry] = []
    artifacts: dict[str, Any] = {}
    add_tasks: list[TaskItem] = []
    remove_task_ids: list[str] = []


class ContextSlice(BaseModel):
    """A filtered, read-only view of a Blackboard for a specific agent."""

    belief_state: str
    trace: list[TraceEntry]
    artifacts: dict[str, Any]
    pending_tasks: list[TaskItem]
