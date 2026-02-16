"""Tests for ``uac inspect`` CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from uac.cli import main
from uac.core.blackboard.blackboard import Blackboard

if TYPE_CHECKING:
    from pathlib import Path


def _write_snapshot(tmp_path: Path) -> Path:
    board = Blackboard(
        belief_state="research complete",
        artifacts={"result": "42"},
    )
    board.add_trace("agent-a", "generate", {"text": "hello"})

    f = tmp_path / "snapshot.json"
    f.write_bytes(board.snapshot())
    return f


class TestInspectCommand:
    def test_inspect_full(self, tmp_path: Path) -> None:
        f = _write_snapshot(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(f)])

        assert result.exit_code == 0
        assert "research complete" in result.output
        assert "Trace entries" in result.output

    def test_inspect_json(self, tmp_path: Path) -> None:
        f = _write_snapshot(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(f), "--json"])

        assert result.exit_code == 0
        assert "research complete" in result.output

    def test_inspect_section_belief(self, tmp_path: Path) -> None:
        f = _write_snapshot(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(f), "--section", "belief"])

        assert result.exit_code == 0
        assert "research complete" in result.output

    def test_inspect_section_artifacts(self, tmp_path: Path) -> None:
        f = _write_snapshot(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(f), "--section", "artifacts"])

        assert result.exit_code == 0
        assert "result" in result.output

    def test_inspect_section_trace(self, tmp_path: Path) -> None:
        f = _write_snapshot(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main, ["inspect", str(f), "--section", "trace", "--json"]
        )

        assert result.exit_code == 0
        assert "agent-a" in result.output

    def test_inspect_bad_file(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json")

        runner = CliRunner()
        result = runner.invoke(main, ["inspect", str(f)])

        assert result.exit_code != 0
        assert "Error loading snapshot" in result.output
