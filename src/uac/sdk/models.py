"""Pydantic models for the workflow YAML schema consumed by ``uac run``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TelemetrySettings(BaseModel):
    """Optional telemetry configuration."""

    enabled: bool = False
    otlp_endpoint: str | None = None


class GatekeeperSettings(BaseModel):
    """Optional gatekeeper configuration."""

    enabled: bool = True
    default_action: Literal["allow", "deny", "ask"] = "ask"
    safe_tools: list[str] = []


class AgentRef(BaseModel):
    """Reference to an agent manifest with optional per-agent model override."""

    manifest: str
    model: dict[str, Any] | None = None


class TopologyConfig(BaseModel):
    """Topology type and type-specific parameters."""

    type: Literal["pipeline", "star", "mesh"]
    order: list[str] | None = None
    supervisor: str | None = None
    subscriptions: dict[str, list[str]] | None = None


class WorkflowSpec(BaseModel):
    """Top-level workflow specification parsed from YAML."""

    version: str = "1"
    name: str = ""
    topology: TopologyConfig
    model: dict[str, Any] = Field(default_factory=dict)
    agents: dict[str, AgentRef]
    max_iterations: int = 30
    gatekeeper: GatekeeperSettings | None = None
    telemetry: TelemetrySettings | None = None
    mcp_servers: list[dict[str, Any]] = []

    @model_validator(mode="after")
    def _validate_topology(self) -> WorkflowSpec:
        agent_names = set(self.agents)
        topo = self.topology

        if topo.type == "pipeline":
            if not topo.order:
                msg = "pipeline topology requires 'order'"
                raise ValueError(msg)
            for name in topo.order:
                if name not in agent_names:
                    msg = f"pipeline order references unknown agent '{name}'"
                    raise ValueError(msg)

        elif topo.type == "star":
            if not topo.supervisor:
                msg = "star topology requires 'supervisor'"
                raise ValueError(msg)
            if topo.supervisor not in agent_names:
                msg = f"star supervisor '{topo.supervisor}' not in agents"
                raise ValueError(msg)

        elif topo.type == "mesh":
            if not topo.subscriptions:
                msg = "mesh topology requires 'subscriptions'"
                raise ValueError(msg)
            for name in topo.subscriptions:
                if name not in agent_names:
                    msg = f"mesh subscription key '{name}' not in agents"
                    raise ValueError(msg)

        return self
