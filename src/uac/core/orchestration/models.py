"""Agent Manifest models — YAML-driven agent definitions.

An Agent Manifest declares everything needed to instantiate and run an
agent: model requirements, system prompt template, tool dependencies,
and input/output schemas.  Manifests live in the ``agents/`` directory
and are loaded + validated at orchestration start-up.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelRequirements(BaseModel):
    """Constraints the agent places on its backing LLM."""

    min_context_window: int = 4096
    capabilities: list[str] = []
    preferred_model: str | None = None


class IOSchema(BaseModel):
    """Loose JSON-schema-style descriptor for an agent's input or output."""

    type: str = "object"
    description: str = ""
    properties: dict[str, Any] = {}
    required: list[str] = []


class MCPServerRef(BaseModel):
    """Reference to an MCP server the agent needs at runtime."""

    name: str
    transport: Literal["stdio", "websocket"] = "stdio"
    command: str | None = None
    url: str | None = None
    env: dict[str, str] = {}


class AgentManifest(BaseModel):
    """Validated representation of an ``agents/*.yaml`` file.

    Example YAML::

        name: summariser
        version: "1.0"
        description: Summarises long documents into bullet points.
        model_requirements:
          min_context_window: 8192
          capabilities: [native_tool_calling]
        system_prompt_template: |
          You are {{ name }}, a summarisation agent.
          {{ extra_instructions }}
        mcp_servers:
          - name: filesystem
            transport: stdio
            command: npx @mcp/filesystem
        input_schema:
          type: object
          properties:
            document: { type: string }
          required: [document]
        output_schema:
          type: object
          properties:
            summary: { type: string }
    """

    name: str
    version: str = "1.0"
    description: str = ""
    model_requirements: ModelRequirements = Field(default_factory=ModelRequirements)
    system_prompt_template: str = ""
    mcp_servers: list[MCPServerRef] = []
    input_schema: IOSchema | None = None
    output_schema: IOSchema | None = None
    metadata: dict[str, Any] = {}
