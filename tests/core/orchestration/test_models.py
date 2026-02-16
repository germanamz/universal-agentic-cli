"""Tests for Agent Manifest models."""

from uac.core.orchestration.models import (
    AgentManifest,
    IOSchema,
    MCPServerRef,
    ModelRequirements,
)


class TestModelRequirements:
    def test_defaults(self) -> None:
        req = ModelRequirements()
        assert req.min_context_window == 4096
        assert req.capabilities == []
        assert req.preferred_model is None

    def test_custom_values(self) -> None:
        req = ModelRequirements(
            min_context_window=8192,
            capabilities=["native_tool_calling", "vision"],
            preferred_model="openai/gpt-4o",
        )
        assert req.min_context_window == 8192
        assert len(req.capabilities) == 2
        assert req.preferred_model == "openai/gpt-4o"


class TestIOSchema:
    def test_defaults(self) -> None:
        schema = IOSchema()
        assert schema.type == "object"
        assert schema.description == ""
        assert schema.properties == {}
        assert schema.required == []

    def test_custom(self) -> None:
        schema = IOSchema(
            type="object",
            properties={"text": {"type": "string"}},
            required=["text"],
        )
        assert "text" in schema.properties
        assert schema.required == ["text"]


class TestMCPServerRef:
    def test_defaults(self) -> None:
        ref = MCPServerRef(name="test")
        assert ref.transport == "stdio"
        assert ref.command is None
        assert ref.url is None
        assert ref.env == {}

    def test_websocket(self) -> None:
        ref = MCPServerRef(name="remote", transport="websocket", url="ws://localhost:3000")
        assert ref.transport == "websocket"
        assert ref.url == "ws://localhost:3000"


class TestAgentManifest:
    def test_minimal(self) -> None:
        manifest = AgentManifest(name="test-agent")
        assert manifest.name == "test-agent"
        assert manifest.version == "1.0"
        assert manifest.description == ""
        assert manifest.system_prompt_template == ""
        assert manifest.mcp_servers == []
        assert manifest.input_schema is None
        assert manifest.output_schema is None
        assert manifest.metadata == {}

    def test_full(self) -> None:
        manifest = AgentManifest(
            name="summariser",
            version="2.0",
            description="Summarises documents",
            model_requirements=ModelRequirements(
                min_context_window=8192,
                capabilities=["native_tool_calling"],
            ),
            system_prompt_template="You are $name.",
            mcp_servers=[MCPServerRef(name="fs", command="npx @mcp/fs")],
            input_schema=IOSchema(properties={"doc": {"type": "string"}}),
            output_schema=IOSchema(properties={"summary": {"type": "string"}}),
            metadata={"team": "nlp"},
        )
        assert manifest.name == "summariser"
        assert manifest.version == "2.0"
        assert manifest.model_requirements.min_context_window == 8192
        assert len(manifest.mcp_servers) == 1
        assert manifest.input_schema is not None
        assert manifest.metadata["team"] == "nlp"

    def test_round_trip_json(self) -> None:
        manifest = AgentManifest(
            name="test",
            description="A test agent",
            system_prompt_template="Hello $name",
        )
        data = manifest.model_dump()
        restored = AgentManifest.model_validate(data)
        assert restored.name == manifest.name
        assert restored.description == manifest.description
        assert restored.system_prompt_template == manifest.system_prompt_template
