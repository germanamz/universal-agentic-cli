"""Blackboard & State Management — shared state store for multi-agent coordination."""

from uac.core.blackboard.backend import BlackboardBackend, InMemoryBackend
from uac.core.blackboard.blackboard import Blackboard
from uac.core.blackboard.models import ContextSlice, StateDelta, TaskItem, TraceEntry
from uac.core.blackboard.slicer import ContextSlicer

__all__ = [
    "Blackboard",
    "BlackboardBackend",
    "ContextSlice",
    "ContextSlicer",
    "InMemoryBackend",
    "StateDelta",
    "TaskItem",
    "TraceEntry",
]
