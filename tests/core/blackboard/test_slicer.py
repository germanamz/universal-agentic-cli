"""Tests for ContextSlicer."""

from uac.core.blackboard.blackboard import Blackboard
from uac.core.blackboard.models import TaskItem, TraceEntry
from uac.core.blackboard.slicer import ContextSlicer


def _populated_board() -> Blackboard:
    """Create a board with varied trace entries and artifacts for slicing tests."""
    board = Blackboard(belief_state="active")
    board.execution_trace = [
        TraceEntry(agent_id="planner", action="route", data={"target": "coder"}),
        TraceEntry(agent_id="coder", action="tool_call", data={"tool": "write"}),
        TraceEntry(agent_id="planner", action="route", data={"target": "reviewer"}),
        TraceEntry(agent_id="reviewer", action="message", data={"text": "LGTM"}),
        TraceEntry(agent_id="coder", action="tool_call", data={"tool": "test"}),
    ]
    board.artifacts = {
        "code": "print('hello')",
        "review": "approved",
        "metrics": {"coverage": 95},
    }
    board.pending_tasks = [
        TaskItem(description="deploy", priority=1),
        TaskItem(description="docs", priority=5),
    ]
    return board


class TestContextSlicer:
    def setup_method(self) -> None:
        self.slicer = ContextSlicer()
        self.board = _populated_board()

    def test_unfiltered_slice(self) -> None:
        cs = self.slicer.slice(self.board)
        assert cs.belief_state == "active"
        assert len(cs.trace) == 5
        assert len(cs.artifacts) == 3
        assert len(cs.pending_tasks) == 2

    def test_filter_by_agent_id(self) -> None:
        cs = self.slicer.slice(self.board, agent_id="coder")
        assert len(cs.trace) == 2
        assert all(e.agent_id == "coder" for e in cs.trace)

    def test_filter_by_agent_id_no_matches(self) -> None:
        cs = self.slicer.slice(self.board, agent_id="nonexistent")
        assert cs.trace == []

    def test_filter_by_artifact_keys(self) -> None:
        cs = self.slicer.slice(self.board, artifact_keys=["code", "metrics"])
        assert set(cs.artifacts.keys()) == {"code", "metrics"}
        assert cs.artifacts["code"] == "print('hello')"

    def test_filter_by_artifact_keys_missing_key(self) -> None:
        cs = self.slicer.slice(self.board, artifact_keys=["code", "nonexistent"])
        assert set(cs.artifacts.keys()) == {"code"}

    def test_filter_artifact_keys_empty_list(self) -> None:
        cs = self.slicer.slice(self.board, artifact_keys=[])
        assert cs.artifacts == {}

    def test_max_trace_entries(self) -> None:
        cs = self.slicer.slice(self.board, max_trace_entries=2)
        assert len(cs.trace) == 2
        # Should be the last 2 entries
        assert cs.trace[0].agent_id == "reviewer"
        assert cs.trace[1].agent_id == "coder"

    def test_max_trace_entries_with_agent_filter(self) -> None:
        cs = self.slicer.slice(self.board, agent_id="planner", max_trace_entries=1)
        assert len(cs.trace) == 1
        # Should be the last planner entry
        assert cs.trace[0].data["target"] == "reviewer"

    def test_all_filters_combined(self) -> None:
        cs = self.slicer.slice(
            self.board,
            agent_id="coder",
            artifact_keys=["code"],
            max_trace_entries=1,
        )
        assert len(cs.trace) == 1
        assert cs.trace[0].data["tool"] == "test"
        assert set(cs.artifacts.keys()) == {"code"}
        assert cs.belief_state == "active"
        assert len(cs.pending_tasks) == 2

    def test_slice_returns_copies(self) -> None:
        cs = self.slicer.slice(self.board)
        cs.artifacts["new_key"] = "injected"
        assert "new_key" not in self.board.artifacts

    def test_empty_board(self) -> None:
        board = Blackboard()
        cs = self.slicer.slice(board)
        assert cs.belief_state == ""
        assert cs.trace == []
        assert cs.artifacts == {}
        assert cs.pending_tasks == []

    def test_pending_tasks_always_included(self) -> None:
        cs = self.slicer.slice(self.board, agent_id="coder", artifact_keys=[])
        assert len(cs.pending_tasks) == 2
