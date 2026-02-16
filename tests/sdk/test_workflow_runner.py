"""Tests for WorkflowRunner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uac.core.blackboard.blackboard import Blackboard
from uac.sdk.models import WorkflowSpec
from uac.sdk.workflow import WorkflowRunner

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

_MANIFEST_YAML = """\
name: {name}
version: "1.0"
description: Test agent
system_prompt_template: "You are $name."
"""


def _write_manifests(base: Path, names: list[str]) -> None:
    agents_dir = base / "agents"
    agents_dir.mkdir(exist_ok=True)
    for n in names:
        (agents_dir / f"{n}.yaml").write_text(_MANIFEST_YAML.format(name=n))


def _pipeline_spec(names: list[str], **extra: Any) -> dict[str, Any]:
    agents = {n: {"manifest": f"agents/{n}.yaml"} for n in names}
    data: dict[str, Any] = {
        "name": "test",
        "topology": {"type": "pipeline", "order": names},
        "model": {"model": "openai/gpt-4o", "api_key": "test"},
        "agents": agents,
    }
    data.update(extra)
    return data


def _star_spec(supervisor: str, workers: list[str], **extra: Any) -> dict[str, Any]:
    all_names = [supervisor, *workers]
    agents = {n: {"manifest": f"agents/{n}.yaml"} for n in all_names}
    data: dict[str, Any] = {
        "name": "test",
        "topology": {"type": "star", "supervisor": supervisor},
        "model": {"model": "openai/gpt-4o", "api_key": "test"},
        "agents": agents,
    }
    data.update(extra)
    return data


def _mesh_spec(names: list[str], subs: dict[str, list[str]], **extra: Any) -> dict[str, Any]:
    agents = {n: {"manifest": f"agents/{n}.yaml"} for n in names}
    data: dict[str, Any] = {
        "name": "test",
        "topology": {"type": "mesh", "subscriptions": subs},
        "model": {"model": "openai/gpt-4o", "api_key": "test"},
        "agents": agents,
    }
    data.update(extra)
    return data


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestWorkflowRunnerInit:
    def test_from_yaml(self, tmp_path: Path) -> None:
        _write_manifests(tmp_path, ["a", "b"])
        f = tmp_path / "workflow.yaml"
        import yaml

        f.write_text(yaml.dump(_pipeline_spec(["a", "b"])))

        runner = WorkflowRunner.from_yaml(f)
        assert runner.spec.topology.type == "pipeline"
        assert runner.base_dir == tmp_path


class TestWorkflowRunnerPipeline:
    @pytest.mark.asyncio
    async def test_creates_pipeline_orchestrator(self, tmp_path: Path) -> None:
        _write_manifests(tmp_path, ["a", "b"])
        spec = WorkflowSpec.model_validate(_pipeline_spec(["a", "b"]))
        runner = WorkflowRunner(spec, base_dir=tmp_path)

        mock_bb = Blackboard()

        with (
            patch("uac.sdk.workflow.ModelClient") as mock_client_cls,
            patch("uac.sdk.workflow.PipelineOrchestrator") as mock_pipeline_cls,
        ):
            mock_instance = mock_client_cls.return_value
            mock_orch = mock_pipeline_cls.return_value
            mock_orch.run = AsyncMock(return_value=mock_bb)
            mock_instance.generate = AsyncMock()

            result = await runner.run("test goal")

            mock_pipeline_cls.assert_called_once()
            call_args = mock_pipeline_cls.call_args
            assert call_args.kwargs["order"] == ["a", "b"]
            assert result is mock_bb


class TestWorkflowRunnerStar:
    @pytest.mark.asyncio
    async def test_creates_star_orchestrator(self, tmp_path: Path) -> None:
        _write_manifests(tmp_path, ["boss", "worker"])
        spec = WorkflowSpec.model_validate(_star_spec("boss", ["worker"]))
        runner = WorkflowRunner(spec, base_dir=tmp_path)

        mock_bb = Blackboard()

        with (
            patch("uac.sdk.workflow.ModelClient"),
            patch("uac.sdk.workflow.StarOrchestrator") as mock_star_cls,
        ):
            mock_orch = mock_star_cls.return_value
            mock_orch.run = AsyncMock(return_value=mock_bb)

            result = await runner.run("test goal")

            mock_star_cls.assert_called_once()
            assert mock_star_cls.call_args.kwargs["supervisor"] == "boss"
            assert result is mock_bb


class TestWorkflowRunnerMesh:
    @pytest.mark.asyncio
    async def test_creates_mesh_orchestrator(self, tmp_path: Path) -> None:
        _write_manifests(tmp_path, ["a", "b"])
        subs = {"a": ["topic.x"], "b": ["topic.y"]}
        spec = WorkflowSpec.model_validate(_mesh_spec(["a", "b"], subs))
        runner = WorkflowRunner(spec, base_dir=tmp_path)

        mock_bb = Blackboard()

        with (
            patch("uac.sdk.workflow.ModelClient"),
            patch("uac.sdk.workflow.MeshOrchestrator") as mock_mesh_cls,
        ):
            mock_orch = mock_mesh_cls.return_value
            mock_orch.run = AsyncMock(return_value=mock_bb)

            result = await runner.run("test goal")

            mock_mesh_cls.assert_called_once()
            assert mock_mesh_cls.call_args.kwargs["subscriptions"] == subs
            assert result is mock_bb


class TestWorkflowRunnerPerAgentOverride:
    @pytest.mark.asyncio
    async def test_per_agent_model_override(self, tmp_path: Path) -> None:
        _write_manifests(tmp_path, ["a", "b"])
        data = _pipeline_spec(["a", "b"])
        data["agents"]["b"]["model"] = {"model": "anthropic/claude-3-haiku", "api_key": "k2"}
        spec = WorkflowSpec.model_validate(data)
        runner = WorkflowRunner(spec, base_dir=tmp_path)

        configs_seen: list[str] = []

        with (
            patch("uac.sdk.workflow.ModelClient") as mock_client_cls,
            patch("uac.sdk.workflow.PipelineOrchestrator") as mock_pipeline_cls,
        ):
            def capture_config(cfg: Any, **kwargs: Any) -> MagicMock:
                configs_seen.append(cfg.model)
                return MagicMock()

            mock_client_cls.side_effect = capture_config
            mock_orch = mock_pipeline_cls.return_value
            mock_orch.run = AsyncMock(return_value=Blackboard())

            await runner.run("goal")

            assert "openai/gpt-4o" in configs_seen
            assert "anthropic/claude-3-haiku" in configs_seen


class TestWorkflowRunnerGatekeeper:
    @pytest.mark.asyncio
    async def test_gatekeeper_wiring(self, tmp_path: Path) -> None:
        _write_manifests(tmp_path, ["a"])
        data = _pipeline_spec(["a"], gatekeeper={"enabled": True, "safe_tools": ["read_file"]})
        spec = WorkflowSpec.model_validate(data)
        runner = WorkflowRunner(spec, base_dir=tmp_path)

        with (
            patch("uac.sdk.workflow.ModelClient"),
            patch("uac.sdk.workflow.SafeDispatcher") as mock_safe_cls,
            patch("uac.sdk.workflow.CLIGatekeeper"),
            patch("uac.sdk.workflow.PipelineOrchestrator") as mock_pipeline_cls,
        ):
            mock_safe = mock_safe_cls.return_value
            mock_safe.all_tools.return_value = []
            mock_orch = mock_pipeline_cls.return_value
            mock_orch.run = AsyncMock(return_value=Blackboard())

            await runner.run("goal")

            mock_safe_cls.assert_called_once()
            config_arg = mock_safe_cls.call_args.kwargs["config"]
            assert config_arg.safe_tools == ["read_file"]
