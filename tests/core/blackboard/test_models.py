"""Tests for blackboard data models."""

from datetime import UTC, datetime

from uac.core.blackboard.models import ContextSlice, StateDelta, TaskItem, TraceEntry


class TestTraceEntry:
    def test_defaults(self) -> None:
        entry = TraceEntry(agent_id="agent-1", action="tool_call")
        assert entry.agent_id == "agent-1"
        assert entry.action == "tool_call"
        assert entry.data == {}
        assert entry.metadata == {}
        assert isinstance(entry.timestamp, datetime)
        assert entry.timestamp.tzinfo is not None

    def test_custom_fields(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        entry = TraceEntry(
            timestamp=ts,
            agent_id="planner",
            action="route",
            data={"target": "coder"},
            metadata={"source": "test"},
        )
        assert entry.timestamp == ts
        assert entry.data["target"] == "coder"
        assert entry.metadata["source"] == "test"

    def test_serialization_round_trip(self) -> None:
        entry = TraceEntry(agent_id="a", action="message", data={"text": "hi"})
        data = entry.model_dump()
        restored = TraceEntry.model_validate(data)
        assert restored.agent_id == entry.agent_id
        assert restored.action == entry.action
        assert restored.data == entry.data

    def test_json_round_trip(self) -> None:
        entry = TraceEntry(agent_id="a", action="message")
        json_bytes = entry.model_dump_json().encode()
        restored = TraceEntry.model_validate_json(json_bytes)
        assert restored.agent_id == entry.agent_id


class TestTaskItem:
    def test_auto_id(self) -> None:
        task = TaskItem(description="Do something")
        assert len(task.id) == 12
        assert task.priority == 0

    def test_unique_ids(self) -> None:
        t1 = TaskItem(description="a")
        t2 = TaskItem(description="b")
        assert t1.id != t2.id

    def test_custom_priority(self) -> None:
        task = TaskItem(description="urgent", priority=-1, metadata={"tag": "hot"})
        assert task.priority == -1
        assert task.metadata["tag"] == "hot"

    def test_serialization_round_trip(self) -> None:
        task = TaskItem(description="test task", priority=5)
        data = task.model_dump()
        restored = TaskItem.model_validate(data)
        assert restored.id == task.id
        assert restored.description == task.description
        assert restored.priority == task.priority


class TestStateDelta:
    def test_empty_delta(self) -> None:
        delta = StateDelta()
        assert delta.belief_state is None
        assert delta.trace_entries == []
        assert delta.artifacts == {}
        assert delta.add_tasks == []
        assert delta.remove_task_ids == []

    def test_partial_delta(self) -> None:
        delta = StateDelta(
            belief_state="planning",
            artifacts={"output": "hello"},
        )
        assert delta.belief_state == "planning"
        assert delta.artifacts == {"output": "hello"}
        assert delta.trace_entries == []

    def test_serialization_round_trip(self) -> None:
        entry = TraceEntry(agent_id="a", action="x")
        task = TaskItem(description="t")
        delta = StateDelta(
            belief_state="done",
            trace_entries=[entry],
            artifacts={"k": "v"},
            add_tasks=[task],
            remove_task_ids=["old-id"],
        )
        data = delta.model_dump()
        restored = StateDelta.model_validate(data)
        assert restored.belief_state == "done"
        assert len(restored.trace_entries) == 1
        assert len(restored.add_tasks) == 1
        assert restored.remove_task_ids == ["old-id"]


class TestContextSlice:
    def test_construction(self) -> None:
        cs = ContextSlice(
            belief_state="active",
            trace=[TraceEntry(agent_id="a", action="x")],
            artifacts={"key": "val"},
            pending_tasks=[TaskItem(description="do it")],
        )
        assert cs.belief_state == "active"
        assert len(cs.trace) == 1
        assert cs.artifacts["key"] == "val"
        assert len(cs.pending_tasks) == 1

    def test_serialization_round_trip(self) -> None:
        cs = ContextSlice(
            belief_state="idle",
            trace=[],
            artifacts={},
            pending_tasks=[],
        )
        data = cs.model_dump()
        restored = ContextSlice.model_validate(data)
        assert restored.belief_state == "idle"
