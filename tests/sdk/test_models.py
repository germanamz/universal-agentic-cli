"""Tests for SDK workflow models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uac.sdk.models import (
    AgentRef,
    GatekeeperSettings,
    TelemetrySettings,
    TopologyConfig,
    WorkflowSpec,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pipeline_spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "test",
        "topology": {"type": "pipeline", "order": ["a", "b"]},
        "agents": {
            "a": {"manifest": "agents/a.yaml"},
            "b": {"manifest": "agents/b.yaml"},
        },
    }
    base.update(overrides)
    return base


def _star_spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "test",
        "topology": {"type": "star", "supervisor": "boss"},
        "agents": {
            "boss": {"manifest": "agents/boss.yaml"},
            "worker": {"manifest": "agents/worker.yaml"},
        },
    }
    base.update(overrides)
    return base


def _mesh_spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "test",
        "topology": {"type": "mesh", "subscriptions": {"a": ["topic.x"]}},
        "agents": {
            "a": {"manifest": "agents/a.yaml"},
        },
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------
# WorkflowSpec — valid
# ------------------------------------------------------------------

class TestWorkflowSpecValid:
    def test_pipeline_minimal(self) -> None:
        spec = WorkflowSpec.model_validate(_pipeline_spec())
        assert spec.topology.type == "pipeline"
        assert spec.topology.order == ["a", "b"]
        assert spec.max_iterations == 30

    def test_star_minimal(self) -> None:
        spec = WorkflowSpec.model_validate(_star_spec())
        assert spec.topology.type == "star"
        assert spec.topology.supervisor == "boss"

    def test_mesh_minimal(self) -> None:
        spec = WorkflowSpec.model_validate(_mesh_spec())
        assert spec.topology.type == "mesh"
        assert spec.topology.subscriptions == {"a": ["topic.x"]}

    def test_optional_sections_default_none(self) -> None:
        spec = WorkflowSpec.model_validate(_pipeline_spec())
        assert spec.gatekeeper is None
        assert spec.telemetry is None

    def test_with_gatekeeper(self) -> None:
        spec = WorkflowSpec.model_validate(
            _pipeline_spec(gatekeeper={"enabled": True, "default_action": "allow"})
        )
        assert spec.gatekeeper is not None
        assert spec.gatekeeper.default_action == "allow"

    def test_with_telemetry(self) -> None:
        spec = WorkflowSpec.model_validate(
            _pipeline_spec(telemetry={"enabled": True})
        )
        assert spec.telemetry is not None
        assert spec.telemetry.enabled is True

    def test_model_override_per_agent(self) -> None:
        spec = WorkflowSpec.model_validate(_pipeline_spec())
        ref = spec.agents["a"]
        assert ref.manifest == "agents/a.yaml"


# ------------------------------------------------------------------
# WorkflowSpec — validation errors
# ------------------------------------------------------------------

class TestWorkflowSpecInvalid:
    def test_pipeline_missing_order(self) -> None:
        data = _pipeline_spec()
        assert isinstance(data["topology"], dict)
        data["topology"]["order"] = None  # type: ignore[index]
        with pytest.raises(ValidationError, match="order"):
            WorkflowSpec.model_validate(data)

    def test_pipeline_unknown_agent_in_order(self) -> None:
        data = _pipeline_spec()
        assert isinstance(data["topology"], dict)
        data["topology"]["order"] = ["a", "missing"]  # type: ignore[index]
        with pytest.raises(ValidationError, match="unknown agent"):
            WorkflowSpec.model_validate(data)

    def test_star_missing_supervisor(self) -> None:
        data = _star_spec()
        assert isinstance(data["topology"], dict)
        data["topology"]["supervisor"] = None  # type: ignore[index]
        with pytest.raises(ValidationError, match="supervisor"):
            WorkflowSpec.model_validate(data)

    def test_star_supervisor_not_in_agents(self) -> None:
        data = _star_spec()
        assert isinstance(data["topology"], dict)
        data["topology"]["supervisor"] = "ghost"  # type: ignore[index]
        with pytest.raises(ValidationError, match="ghost"):
            WorkflowSpec.model_validate(data)

    def test_mesh_missing_subscriptions(self) -> None:
        data = _mesh_spec()
        assert isinstance(data["topology"], dict)
        data["topology"]["subscriptions"] = None  # type: ignore[index]
        with pytest.raises(ValidationError, match="subscriptions"):
            WorkflowSpec.model_validate(data)

    def test_mesh_unknown_agent_in_subscriptions(self) -> None:
        data = _mesh_spec()
        assert isinstance(data["topology"], dict)
        data["topology"]["subscriptions"] = {"ghost": ["t"]}  # type: ignore[index]
        with pytest.raises(ValidationError, match="ghost"):
            WorkflowSpec.model_validate(data)


# ------------------------------------------------------------------
# Sub-models
# ------------------------------------------------------------------

class TestSubModels:
    def test_agent_ref_defaults(self) -> None:
        ref = AgentRef(manifest="agents/x.yaml")
        assert ref.model is None

    def test_topology_config_types(self) -> None:
        for t in ("pipeline", "star", "mesh"):
            TopologyConfig(type=t)  # type: ignore[arg-type]

    def test_gatekeeper_defaults(self) -> None:
        gs = GatekeeperSettings()
        assert gs.enabled is True
        assert gs.default_action == "ask"
        assert gs.safe_tools == []

    def test_telemetry_defaults(self) -> None:
        ts = TelemetrySettings()
        assert ts.enabled is False
        assert ts.otlp_endpoint is None
