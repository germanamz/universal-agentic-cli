"""Tests for ``uac agents`` CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from uac.cli import main

if TYPE_CHECKING:
    from pathlib import Path

_MANIFEST = """\
name: {name}
version: "1.0"
description: A test agent named {name}.
"""


class TestAgentsList:
    def test_list_agents(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "alpha.yaml").write_text(_MANIFEST.format(name="alpha"))
        (agents_dir / "beta.yaml").write_text(_MANIFEST.format(name="beta"))

        runner = CliRunner()
        result = runner.invoke(main, ["agents", "list", "--dir", str(agents_dir)])

        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" in result.output

    def test_list_agents_json(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "alpha.yaml").write_text(_MANIFEST.format(name="alpha"))

        runner = CliRunner()
        result = runner.invoke(
            main, ["agents", "list", "--dir", str(agents_dir), "--format", "json"]
        )

        assert result.exit_code == 0
        assert '"alpha"' in result.output

    def test_list_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, ["agents", "list", "--dir", str(empty)])

        assert result.exit_code == 0
        assert "No agent manifests found" in result.output

    def test_list_nonexistent_dir(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main, ["agents", "list", "--dir", str(tmp_path / "nope")]
        )

        assert result.exit_code == 0
        assert "not found" in result.output
