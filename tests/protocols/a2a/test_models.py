"""Tests for A2A models."""

from uac.protocols.a2a.models import (
    A2AArtifact,
    A2AMessage,
    A2APart,
    A2ATaskParams,
    A2ATaskRequest,
    A2ATaskResponse,
    A2ATaskResult,
    A2ATaskStatus,
    AgentCard,
    AgentSkill,
)


class TestAgentSkill:
    def test_defaults(self) -> None:
        skill = AgentSkill(id="summarize", name="Summarize")
        assert skill.description == ""
        assert skill.tags == []

    def test_round_trip(self) -> None:
        skill = AgentSkill(
            id="translate", name="Translate", description="Translate text", tags=["nlp"]
        )
        data = skill.model_dump()
        restored = AgentSkill.model_validate(data)
        assert restored == skill


class TestAgentCard:
    def test_minimal(self) -> None:
        card = AgentCard(name="helper", url="https://agent.example.com")
        assert card.skills == []
        assert card.description == ""

    def test_with_skills(self) -> None:
        card = AgentCard(
            name="multi-agent",
            url="https://agent.example.com",
            skills=[
                AgentSkill(id="a", name="Alpha"),
                AgentSkill(id="b", name="Beta"),
            ],
        )
        assert len(card.skills) == 2

    def test_from_json(self) -> None:
        raw = {
            "name": "test-agent",
            "description": "A test agent",
            "url": "https://test.example.com",
            "skills": [{"id": "echo", "name": "Echo", "description": "Echo input"}],
        }
        card = AgentCard.model_validate(raw)
        assert card.name == "test-agent"
        assert card.skills[0].id == "echo"


class TestA2AMessage:
    def test_defaults(self) -> None:
        msg = A2AMessage()
        assert msg.role == "user"
        assert msg.parts == []

    def test_with_parts(self) -> None:
        msg = A2AMessage(
            role="agent", parts=[A2APart(type="text", text="Hello")]
        )
        assert msg.parts[0].text == "Hello"


class TestA2ATaskRequest:
    def test_defaults(self) -> None:
        req = A2ATaskRequest()
        assert req.jsonrpc == "2.0"
        assert req.method == "tasks/send"

    def test_round_trip(self) -> None:
        req = A2ATaskRequest(
            params=A2ATaskParams(
                id="task-1",
                message=A2AMessage(parts=[A2APart(text="Do something")]),
            )
        )
        data = req.model_dump()
        restored = A2ATaskRequest.model_validate(data)
        assert restored.params.id == "task-1"


class TestA2ATaskResponse:
    def test_success_response(self) -> None:
        resp = A2ATaskResponse(
            result=A2ATaskResult(
                id="task-1",
                status=A2ATaskStatus(state="completed"),
                artifacts=[A2AArtifact(parts=[A2APart(text="Result")])],
            )
        )
        assert resp.result is not None
        assert resp.result.artifacts[0].parts[0].text == "Result"
        assert resp.error is None

    def test_error_response(self) -> None:
        resp = A2ATaskResponse(error={"code": -32600, "message": "Invalid request"})
        assert resp.error is not None
        assert resp.result is None
