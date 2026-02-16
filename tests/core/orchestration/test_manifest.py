"""Tests for the manifest loader and template rendering."""

import json
from pathlib import Path
from typing import Any

import pytest

from uac.core.orchestration.manifest import ManifestLoader, parse_manifest, render_prompt
from uac.core.orchestration.models import AgentManifest


class TestParseManifest:
    def test_parse_json(self) -> None:
        raw = json.dumps({"name": "test-agent", "description": "A test"})
        manifest = parse_manifest(raw, format="json")
        assert manifest.name == "test-agent"
        assert manifest.description == "A test"

    def test_parse_yaml(self) -> None:
        raw = "name: yaml-agent\ndescription: From YAML\n"
        manifest = parse_manifest(raw, format="yaml")
        assert manifest.name == "yaml-agent"
        assert manifest.description == "From YAML"

    def test_parse_yaml_with_nested(self) -> None:
        raw = (
            "name: complex\n"
            "model_requirements:\n"
            "  min_context_window: 16384\n"
            "  capabilities:\n"
            "    - vision\n"
        )
        manifest = parse_manifest(raw)
        assert manifest.model_requirements.min_context_window == 16384
        assert "vision" in manifest.model_requirements.capabilities

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(Exception):
            parse_manifest("{invalid", format="json")

    def test_missing_name_raises(self) -> None:
        with pytest.raises(Exception):
            parse_manifest(json.dumps({"description": "no name"}), format="json")


class TestRenderPrompt:
    def test_basic_substitution(self) -> None:
        manifest = AgentManifest(
            name="helper",
            description="A helper",
            system_prompt_template="You are $name, $description.",
        )
        result = render_prompt(manifest)
        assert result == "You are helper, A helper."

    def test_custom_variables(self) -> None:
        manifest = AgentManifest(
            name="agent",
            system_prompt_template="$name says $greeting",
        )
        result = render_prompt(manifest, greeting="hello")
        assert result == "agent says hello"

    def test_unknown_placeholder_safe(self) -> None:
        manifest = AgentManifest(
            name="agent",
            system_prompt_template="$name and $unknown_var",
        )
        result = render_prompt(manifest)
        assert "$unknown_var" in result
        assert "agent" in result

    def test_override_defaults(self) -> None:
        manifest = AgentManifest(
            name="original",
            system_prompt_template="I am $name",
        )
        result = render_prompt(manifest, name="overridden")
        assert result == "I am overridden"

    def test_empty_template(self) -> None:
        manifest = AgentManifest(name="agent", system_prompt_template="")
        assert render_prompt(manifest) == ""


class TestManifestLoader:
    def test_load_all_json(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {"name": "agent-a", "description": "Agent A"}
        (tmp_path / "agent_a.json").write_text(json.dumps(data))

        loader = ManifestLoader(tmp_path)
        manifests = loader.load_all()
        assert "agent-a" in manifests
        assert manifests["agent-a"].description == "Agent A"

    def test_load_all_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "agent_b.yaml").write_text("name: agent-b\ndescription: Agent B\n")

        loader = ManifestLoader(tmp_path)
        manifests = loader.load_all()
        assert "agent-b" in manifests

    def test_load_all_empty_dir(self, tmp_path: Path) -> None:
        loader = ManifestLoader(tmp_path)
        assert loader.load_all() == {}

    def test_load_all_nonexistent_dir(self, tmp_path: Path) -> None:
        loader = ManifestLoader(tmp_path / "nonexistent")
        assert loader.load_all() == {}

    def test_load_one(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {"name": "target", "version": "2.0"}
        (tmp_path / "target.json").write_text(json.dumps(data))

        loader = ManifestLoader(tmp_path)
        manifest = loader.load_one("target")
        assert manifest.name == "target"
        assert manifest.version == "2.0"

    def test_load_one_not_found(self, tmp_path: Path) -> None:
        loader = ManifestLoader(tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load_one("nonexistent")

    def test_load_one_caches(self, tmp_path: Path) -> None:
        data: dict[str, Any] = {"name": "cached"}
        (tmp_path / "cached.json").write_text(json.dumps(data))

        loader = ManifestLoader(tmp_path)
        first = loader.load_one("cached")
        second = loader.load_one("cached")
        assert first.name == second.name

    def test_ignores_non_manifest_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.md").write_text("# Not a manifest")
        (tmp_path / "agent.yaml").write_text("name: real\n")

        loader = ManifestLoader(tmp_path)
        manifests = loader.load_all()
        assert len(manifests) == 1
        assert "real" in manifests

    def test_multiple_manifests(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text("name: alpha\n")
        (tmp_path / "b.yml").write_text("name: beta\n")
        (tmp_path / "c.json").write_text(json.dumps({"name": "gamma"}))

        loader = ManifestLoader(tmp_path)
        manifests = loader.load_all()
        assert len(manifests) == 3
        assert set(manifests.keys()) == {"alpha", "beta", "gamma"}
