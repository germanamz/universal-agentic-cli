"""Tests for the Blackboard class."""

from uac.core.blackboard.blackboard import Blackboard, _deep_merge
from uac.core.blackboard.models import StateDelta, TaskItem, TraceEntry


class TestDeepMerge:
    def test_flat_overwrite(self) -> None:
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        _deep_merge(base, {"x": {"b": 99, "c": 3}})
        assert base == {"x": {"a": 1, "b": 99, "c": 3}}

    def test_none_deletes_key(self) -> None:
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": None})
        assert base == {"a": 1}

    def test_none_deletes_nested_key(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        _deep_merge(base, {"x": {"a": None}})
        assert base == {"x": {"b": 2}}

    def test_overwrite_non_dict_with_dict(self) -> None:
        base = {"a": "string"}
        _deep_merge(base, {"a": {"nested": True}})
        assert base == {"a": {"nested": True}}

    def test_empty_updates(self) -> None:
        base = {"a": 1}
        _deep_merge(base, {})
        assert base == {"a": 1}

    def test_delete_missing_key_is_noop(self) -> None:
        base = {"a": 1}
        _deep_merge(base, {"missing": None})
        assert base == {"a": 1}


class TestBlackboard:
    def setup_method(self) -> None:
        self.board = Blackboard()

    def test_defaults(self) -> None:
        assert self.board.belief_state == ""
        assert self.board.execution_trace == []
        assert self.board.artifacts == {}
        assert self.board.pending_tasks == []

    def test_apply_belief_state(self) -> None:
        delta = StateDelta(belief_state="planning")
        result = self.board.apply(delta)
        assert result is self.board
        assert self.board.belief_state == "planning"

    def test_apply_none_belief_state_does_not_overwrite(self) -> None:
        self.board.belief_state = "active"
        self.board.apply(StateDelta(belief_state=None))
        assert self.board.belief_state == "active"

    def test_apply_trace_entries(self) -> None:
        entries = [
            TraceEntry(agent_id="a", action="x"),
            TraceEntry(agent_id="b", action="y"),
        ]
        self.board.apply(StateDelta(trace_entries=entries))
        assert len(self.board.execution_trace) == 2
        # Apply more
        self.board.apply(StateDelta(trace_entries=[TraceEntry(agent_id="c", action="z")]))
        assert len(self.board.execution_trace) == 3

    def test_apply_artifacts_deep_merge(self) -> None:
        self.board.apply(StateDelta(artifacts={"config": {"verbose": True, "level": 1}}))
        self.board.apply(StateDelta(artifacts={"config": {"level": 2, "new_key": "ok"}}))
        assert self.board.artifacts == {"config": {"verbose": True, "level": 2, "new_key": "ok"}}

    def test_apply_artifacts_delete_key(self) -> None:
        self.board.artifacts = {"keep": 1, "remove": 2}
        self.board.apply(StateDelta(artifacts={"remove": None}))
        assert self.board.artifacts == {"keep": 1}

    def test_apply_add_tasks(self) -> None:
        t1 = TaskItem(description="low", priority=10)
        t2 = TaskItem(description="high", priority=1)
        self.board.apply(StateDelta(add_tasks=[t1, t2]))
        assert len(self.board.pending_tasks) == 2
        assert self.board.pending_tasks[0].priority == 1
        assert self.board.pending_tasks[1].priority == 10

    def test_apply_remove_tasks(self) -> None:
        t1 = TaskItem(id="aaa", description="keep")
        t2 = TaskItem(id="bbb", description="remove")
        self.board.pending_tasks = [t1, t2]
        self.board.apply(StateDelta(remove_task_ids=["bbb"]))
        assert len(self.board.pending_tasks) == 1
        assert self.board.pending_tasks[0].id == "aaa"

    def test_apply_add_and_remove_tasks(self) -> None:
        existing = TaskItem(id="old", description="old task", priority=5)
        self.board.pending_tasks = [existing]
        new_task = TaskItem(description="new task", priority=1)
        self.board.apply(StateDelta(add_tasks=[new_task], remove_task_ids=["old"]))
        assert len(self.board.pending_tasks) == 1
        assert self.board.pending_tasks[0].description == "new task"

    def test_apply_chaining(self) -> None:
        result = (
            self.board
            .apply(StateDelta(belief_state="step1"))
            .apply(StateDelta(belief_state="step2"))
        )
        assert result is self.board
        assert self.board.belief_state == "step2"


class TestBlackboardSnapshot:
    def test_snapshot_restore_round_trip(self) -> None:
        board = Blackboard(
            belief_state="testing",
            execution_trace=[TraceEntry(agent_id="a", action="test")],
            artifacts={"result": 42},
            pending_tasks=[TaskItem(description="verify")],
        )
        data = board.snapshot()
        restored = Blackboard.restore(data)
        assert restored.belief_state == "testing"
        assert len(restored.execution_trace) == 1
        assert restored.artifacts["result"] == 42
        assert len(restored.pending_tasks) == 1

    def test_snapshot_is_bytes(self) -> None:
        board = Blackboard()
        assert isinstance(board.snapshot(), bytes)

    def test_empty_board_round_trip(self) -> None:
        board = Blackboard()
        restored = Blackboard.restore(board.snapshot())
        assert restored.belief_state == ""
        assert restored.execution_trace == []


class TestBlackboardConvenience:
    def setup_method(self) -> None:
        self.board = Blackboard()

    def test_get_set_artifact(self) -> None:
        assert self.board.get_artifact("key") is None
        assert self.board.get_artifact("key", "default") == "default"
        self.board.set_artifact("key", "value")
        assert self.board.get_artifact("key") == "value"

    def test_add_trace(self) -> None:
        self.board.add_trace("agent-1", "tool_call", {"tool": "search"})
        assert len(self.board.execution_trace) == 1
        entry = self.board.execution_trace[0]
        assert entry.agent_id == "agent-1"
        assert entry.action == "tool_call"
        assert entry.data == {"tool": "search"}

    def test_add_trace_no_data(self) -> None:
        self.board.add_trace("agent-1", "message")
        assert self.board.execution_trace[0].data == {}

    def test_pop_task_empty(self) -> None:
        assert self.board.pop_task() is None

    def test_pop_task_returns_highest_priority(self) -> None:
        self.board.pending_tasks = [
            TaskItem(description="low", priority=10),
            TaskItem(description="high", priority=1),
        ]
        self.board.pending_tasks.sort(key=lambda t: t.priority)
        task = self.board.pop_task()
        assert task is not None
        assert task.description == "high"
        assert len(self.board.pending_tasks) == 1

    def test_slice_delegates_to_slicer(self) -> None:
        self.board.belief_state = "active"
        self.board.add_trace("agent-1", "x")
        self.board.add_trace("agent-2", "y")
        cs = self.board.slice(agent_id="agent-1")
        assert cs.belief_state == "active"
        assert len(cs.trace) == 1
        assert cs.trace[0].agent_id == "agent-1"
