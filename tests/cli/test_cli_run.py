"""Tests for ``uac run`` CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from uac.cli import main
from uac.core.blackboard.blackboard import Blackboard

if TYPE_CHECKING:
    from pathlib import Path

_VALID_YAML = """\
version: "1"
name: test-pipeline
topology:
  type: pipeline
  order: [a, b]
model:
  model: openai/gpt-4o
  api_key: test-key
agents:
  a:
    manifest: agents/a.yaml
  b:
    manifest: agents/b.yaml
"""


class TestRunCommand:
    def test_dry_run(self, tmp_path: Path) -> None:
        f = tmp_path / "workflow.yaml"
        f.write_text(_VALID_YAML)

        runner = CliRunner()
        result = runner.invoke(main, ["run", str(f), "--dry-run"])

        assert result.exit_code == 0
        assert "validated successfully" in result.output
        assert "test-pipeline" in result.output
        assert "pipeline" in result.output

    def test_dry_run_invalid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("name: only-name\n")

        runner = CliRunner()
        result = runner.invoke(main, ["run", str(f), "--dry-run"])

        assert result.exit_code != 0
        assert "Validation error" in result.output

    def test_missing_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "/nonexistent/workflow.yaml"])

        assert result.exit_code != 0

    def test_verbose_flag(self, tmp_path: Path) -> None:
        f = tmp_path / "workflow.yaml"
        f.write_text(_VALID_YAML)

        mock_bb = Blackboard(belief_state="done")

        with patch("uac.sdk.workflow.WorkflowRunner") as mock_runner_cls:
            mock_instance = mock_runner_cls.return_value
            mock_instance.run = AsyncMock(return_value=mock_bb)

            runner = CliRunner()
            result = runner.invoke(
                main, ["run", str(f), "--verbose", "--goal", "test goal"]
            )

            assert result.exit_code == 0
            assert "Running workflow" in result.output

    def test_telemetry_flag(self, tmp_path: Path) -> None:
        f = tmp_path / "workflow.yaml"
        f.write_text(_VALID_YAML)

        runner = CliRunner()
        result = runner.invoke(main, ["run", str(f), "--dry-run", "--telemetry"])

        assert result.exit_code == 0
