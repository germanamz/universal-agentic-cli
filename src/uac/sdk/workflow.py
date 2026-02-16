"""Workflow loading and execution for the UAC SDK."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError

from uac.core.interface.client import ModelClient
from uac.core.interface.config import ModelConfig
from uac.core.orchestration.manifest import parse_manifest
from uac.core.orchestration.models import MCPServerRef
from uac.core.orchestration.primitives import AgentNode
from uac.core.orchestration.topologies.mesh import MeshOrchestrator
from uac.core.orchestration.topologies.pipeline import PipelineOrchestrator
from uac.core.orchestration.topologies.star import StarOrchestrator
from uac.protocols.dispatcher import ToolDispatcher
from uac.protocols.mcp.client import MCPClient
from uac.runtime.dispatcher import SafeDispatcher
from uac.runtime.gatekeeper.gatekeeper import CLIGatekeeper
from uac.runtime.gatekeeper.models import GatekeeperConfig, PolicyAction
from uac.sdk.errors import WorkflowValidationError
from uac.sdk.models import WorkflowSpec
from uac.utils.telemetry import configure_telemetry

if TYPE_CHECKING:
    from uac.core.blackboard.blackboard import Blackboard


class WorkflowLoader:
    """Load and validate a workflow YAML file into a :class:`WorkflowSpec`."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> WorkflowSpec:
        """Read YAML, interpolate env vars, and validate.

        Environment variables in the form ``${VAR}`` or ``$VAR`` are expanded
        using :func:`os.path.expandvars` before YAML parsing.

        Raises:
            WorkflowValidationError: On YAML parse errors or schema validation failures.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise WorkflowValidationError(f"Cannot read {self._path}: {exc}") from exc

        expanded = os.path.expandvars(raw)

        try:
            data: Any = yaml.safe_load(expanded)
        except yaml.YAMLError as exc:
            raise WorkflowValidationError(f"YAML parse error: {exc}") from exc

        if not isinstance(data, dict):
            raise WorkflowValidationError("Workflow YAML must be a mapping")

        try:
            return WorkflowSpec.model_validate(data)
        except ValidationError as exc:
            raise WorkflowValidationError(str(exc)) from exc


class WorkflowRunner:
    """Execute a validated :class:`WorkflowSpec` end-to-end."""

    def __init__(
        self,
        spec: WorkflowSpec,
        *,
        base_dir: Path | None = None,
    ) -> None:
        self.spec = spec
        self.base_dir = base_dir or Path.cwd()

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkflowRunner:
        """Load a workflow YAML and return a ready-to-run runner."""
        p = Path(path)
        spec = WorkflowLoader(p).load()
        return cls(spec, base_dir=p.parent)

    async def run(self, goal: str) -> Blackboard:
        """Wire up all components and execute the orchestration loop.

        Steps:
        1. Parse agent manifests and create ModelClient per agent.
        2. Connect MCP clients, register on ToolDispatcher, discover tools.
        3. Optionally wrap dispatcher with SafeDispatcher (gatekeeper).
        4. Build AgentNode instances.
        5. Instantiate the correct orchestrator topology.
        6. Optionally configure telemetry.
        7. Run the orchestrator and return the final Blackboard.
        """
        default_model_cfg = ModelConfig(**self.spec.model) if self.spec.model else None

        # Collect MCP server refs across all agents + global
        mcp_refs: list[MCPServerRef] = [
            MCPServerRef(**s) for s in self.spec.mcp_servers
        ]

        # Parse manifests and create clients per agent
        manifests: dict[str, Any] = {}
        clients: dict[str, ModelClient] = {}

        for agent_name, agent_ref in self.spec.agents.items():
            manifest_path = self.base_dir / agent_ref.manifest
            raw = manifest_path.read_text(encoding="utf-8")
            fmt = "json" if manifest_path.suffix == ".json" else "yaml"
            manifest = parse_manifest(raw, format=fmt)
            manifests[agent_name] = manifest

            # Per-agent model override or fall back to workflow-level default
            if agent_ref.model:
                cfg = ModelConfig(**agent_ref.model)
            elif default_model_cfg:
                cfg = default_model_cfg
            else:
                preferred = manifest.model_requirements.preferred_model
                cfg = ModelConfig(model=preferred or "openai/gpt-4o")
            clients[agent_name] = ModelClient(cfg)

            # Collect per-agent MCP servers
            mcp_refs.extend(manifest.mcp_servers)

        # Connect MCP clients and discover tools
        dispatcher = ToolDispatcher()
        mcp_clients: list[MCPClient] = []

        for ref in mcp_refs:
            client = MCPClient(ref)
            await client.connect()
            await dispatcher.register(client)
            mcp_clients.append(client)

        # Wrap with SafeDispatcher if gatekeeper is enabled
        effective_dispatcher: ToolDispatcher | SafeDispatcher = dispatcher
        if self.spec.gatekeeper and self.spec.gatekeeper.enabled:
            gk_settings = self.spec.gatekeeper
            gk_config = GatekeeperConfig(
                enabled=True,
                default_action=PolicyAction(gk_settings.default_action),
                safe_tools=gk_settings.safe_tools,
            )
            effective_dispatcher = SafeDispatcher(
                dispatcher,
                gatekeeper=CLIGatekeeper(),
                config=gk_config,
            )

        # Build AgentNode instances
        all_tools = effective_dispatcher.all_tools()
        agents: dict[str, AgentNode] = {}
        for agent_name, manifest in manifests.items():
            agents[agent_name] = AgentNode(
                manifest=manifest,
                client=clients[agent_name],
                tools=all_tools if all_tools else None,
            )

        # Instantiate correct orchestrator
        topo = self.spec.topology
        if topo.type == "pipeline":
            assert topo.order is not None
            orchestrator = PipelineOrchestrator(
                agents,
                order=topo.order,
                max_iterations=self.spec.max_iterations,
            )
        elif topo.type == "star":
            assert topo.supervisor is not None
            orchestrator = StarOrchestrator(
                agents,
                supervisor=topo.supervisor,
                max_iterations=self.spec.max_iterations,
            )
        else:
            assert topo.subscriptions is not None
            orchestrator = MeshOrchestrator(
                agents,
                subscriptions=topo.subscriptions,
                max_iterations=self.spec.max_iterations,
            )

        # Configure telemetry if enabled
        if self.spec.telemetry and self.spec.telemetry.enabled:
            configure_telemetry(
                otlp_endpoint=self.spec.telemetry.otlp_endpoint,
            )

        try:
            return await orchestrator.run(goal)
        finally:
            for mc in mcp_clients:
                await mc.close()
