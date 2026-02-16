"""Tests for WorkflowLoader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from uac.sdk.errors import WorkflowValidationError
from uac.sdk.workflow import WorkflowLoader

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


class TestWorkflowLoader:
    def test_load_valid(self, tmp_path: Path) -> None:
        f = tmp_path / "workflow.yaml"
        f.write_text(_VALID_YAML)
        spec = WorkflowLoader(f).load()
        assert spec.name == "test-pipeline"
        assert spec.topology.type == "pipeline"

    def test_env_var_interpolation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "secret-123")
        content = _VALID_YAML.replace("test-key", "${MY_KEY}")
        f = tmp_path / "workflow.yaml"
        f.write_text(content)
        spec = WorkflowLoader(f).load()
        assert spec.model["api_key"] == "secret-123"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(WorkflowValidationError, match="Cannot read"):
            WorkflowLoader(tmp_path / "missing.yaml").load()

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("{{{{invalid")
        with pytest.raises(WorkflowValidationError, match="YAML parse error"):
            WorkflowLoader(f).load()

    def test_yaml_not_mapping(self, tmp_path: Path) -> None:
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(WorkflowValidationError, match="must be a mapping"):
            WorkflowLoader(f).load()

    def test_validation_error(self, tmp_path: Path) -> None:
        f = tmp_path / "incomplete.yaml"
        f.write_text("name: test\n")
        with pytest.raises(WorkflowValidationError):
            WorkflowLoader(f).load()
