"""Tests for ReflexionMiddleware and output validators."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from uac.core.blackboard.models import ContextSlice, StateDelta, TraceEntry
from uac.core.interface.models import CanonicalMessage
from uac.core.orchestration.models import AgentManifest
from uac.core.orchestration.primitives import AgentNode
from uac.utils.reflexion import (
    JsonContentValidator,
    NonEmptyValidator,
    OutputValidator,
    ReflexionMiddleware,
    SchemaValidator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(name: str = "test-agent") -> AgentManifest:
    return AgentManifest(name=name, system_prompt_template="You are $name.")


def _make_client(response_text: str = "Hello") -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(return_value=CanonicalMessage.assistant(response_text))
    client.config = MagicMock()
    return client


def _make_context() -> ContextSlice:
    return ContextSlice(
        belief_state="testing",
        trace=[],
        artifacts={},
        pending_tasks=[],
    )


def _make_delta(text: str = "Hello", has_tool_calls: bool = False) -> StateDelta:
    return StateDelta(
        trace_entries=[
            TraceEntry(
                agent_id="test-agent",
                action="generate",
                data={"text": text, "has_tool_calls": has_tool_calls},
            )
        ],
        artifacts={"last_response": {"test-agent": text}},
    )


# ---------------------------------------------------------------------------
# NonEmptyValidator
# ---------------------------------------------------------------------------


class TestNonEmptyValidator:
    def test_valid_response(self) -> None:
        v = NonEmptyValidator()
        assert v.validate(_make_delta("Hello world")) == []

    def test_empty_response(self) -> None:
        v = NonEmptyValidator()
        errors = v.validate(_make_delta(""))
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_whitespace_only(self) -> None:
        v = NonEmptyValidator()
        errors = v.validate(_make_delta("   \n\t  "))
        assert len(errors) == 1

    def test_no_trace_entries(self) -> None:
        v = NonEmptyValidator()
        errors = v.validate(StateDelta())
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# JsonContentValidator
# ---------------------------------------------------------------------------


class TestJsonContentValidator:
    def test_valid_json(self) -> None:
        v = JsonContentValidator()
        assert v.validate(_make_delta('{"key": "value"}')) == []

    def test_valid_json_array(self) -> None:
        v = JsonContentValidator()
        assert v.validate(_make_delta('[1, 2, 3]')) == []

    def test_invalid_json(self) -> None:
        v = JsonContentValidator()
        errors = v.validate(_make_delta("not json at all"))
        assert len(errors) == 1
        assert "not valid JSON" in errors[0]

    def test_empty_text(self) -> None:
        v = JsonContentValidator()
        errors = v.validate(_make_delta(""))
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------


class TestSchemaValidator:
    def test_valid_keys(self) -> None:
        v = SchemaValidator(required_keys=["action", "target"])
        delta = _make_delta('{"action": "run", "target": "tests"}')
        assert v.validate(delta) == []

    def test_missing_keys(self) -> None:
        v = SchemaValidator(required_keys=["action", "target"])
        delta = _make_delta('{"action": "run"}')
        errors = v.validate(delta)
        assert len(errors) == 1
        assert "target" in errors[0]

    def test_not_an_object(self) -> None:
        v = SchemaValidator(required_keys=["key"])
        errors = v.validate(_make_delta('"just a string"'))
        assert len(errors) == 1
        assert "not an object" in errors[0]

    def test_invalid_json(self) -> None:
        v = SchemaValidator(required_keys=["key"])
        errors = v.validate(_make_delta("{broken"))
        assert len(errors) == 1
        assert "not valid JSON" in errors[0]


# ---------------------------------------------------------------------------
# OutputValidator protocol
# ---------------------------------------------------------------------------


class TestOutputValidatorProtocol:
    def test_custom_validator_is_protocol(self) -> None:
        class MyValidator:
            def validate(self, delta: StateDelta) -> list[str]:
                return []

        assert isinstance(MyValidator(), OutputValidator)


# ---------------------------------------------------------------------------
# ReflexionMiddleware
# ---------------------------------------------------------------------------


class TestReflexionMiddleware:
    async def test_passes_through_valid_output(self) -> None:
        agent = AgentNode(manifest=_make_manifest(), client=_make_client("valid"))
        middleware = ReflexionMiddleware(agent, validators=[NonEmptyValidator()])

        delta = await middleware.step(_make_context())

        assert len(delta.trace_entries) == 1
        assert delta.trace_entries[0].data["text"] == "valid"

    async def test_retries_on_validation_failure(self) -> None:
        # First response empty, second valid
        client = MagicMock()
        client.generate = AsyncMock(
            side_effect=[
                CanonicalMessage.assistant(""),
                CanonicalMessage.assistant("fixed"),
            ]
        )
        agent = AgentNode(manifest=_make_manifest(), client=client)
        middleware = ReflexionMiddleware(agent, validators=[NonEmptyValidator()])

        delta = await middleware.step(_make_context())

        assert client.generate.await_count == 2
        assert delta.trace_entries[0].data["text"] == "fixed"

    async def test_exhausts_retries(self) -> None:
        # All responses empty
        client = MagicMock()
        client.generate = AsyncMock(return_value=CanonicalMessage.assistant(""))
        agent = AgentNode(manifest=_make_manifest(), client=client)
        middleware = ReflexionMiddleware(
            agent, validators=[NonEmptyValidator()], max_retries=2
        )

        delta = await middleware.step(_make_context())

        # 1 initial + 2 retries = 3 calls
        assert client.generate.await_count == 3
        # Returns the last (still invalid) delta
        assert delta.trace_entries[0].data["text"] == ""

    async def test_injects_error_feedback(self) -> None:
        calls: list[CanonicalMessage] = []
        original_generate = AsyncMock(
            side_effect=[
                CanonicalMessage.assistant(""),
                CanonicalMessage.assistant("ok"),
            ]
        )

        client = MagicMock()
        client.generate = original_generate
        agent = AgentNode(manifest=_make_manifest(), client=client)
        middleware = ReflexionMiddleware(agent, validators=[NonEmptyValidator()])

        await middleware.step(_make_context())

        # Second call should have error feedback in the history
        # Check that generate was called twice
        assert client.generate.await_count == 2

    async def test_max_retries_zero(self) -> None:
        """With max_retries=0, no retries occur."""
        client = MagicMock()
        client.generate = AsyncMock(return_value=CanonicalMessage.assistant(""))
        agent = AgentNode(manifest=_make_manifest(), client=client)
        middleware = ReflexionMiddleware(
            agent, validators=[NonEmptyValidator()], max_retries=0
        )

        delta = await middleware.step(_make_context())

        assert client.generate.await_count == 1

    async def test_multiple_validators(self) -> None:
        """All validators must pass."""
        client = MagicMock()
        client.generate = AsyncMock(
            side_effect=[
                CanonicalMessage.assistant("not json"),
                CanonicalMessage.assistant('{"action": "run"}'),
            ]
        )
        agent = AgentNode(manifest=_make_manifest(), client=client)
        middleware = ReflexionMiddleware(
            agent,
            validators=[NonEmptyValidator(), JsonContentValidator()],
        )

        delta = await middleware.step(_make_context())

        assert client.generate.await_count == 2
        assert delta.trace_entries[0].data["text"] == '{"action": "run"}'

    async def test_name_delegates_to_agent(self) -> None:
        agent = AgentNode(manifest=_make_manifest("my-agent"), client=_make_client())
        middleware = ReflexionMiddleware(agent, validators=[])
        assert middleware.name == "my-agent"

    async def test_properties_delegate(self) -> None:
        agent = AgentNode(manifest=_make_manifest(), client=_make_client())
        middleware = ReflexionMiddleware(agent, validators=[])
        assert middleware.manifest is agent.manifest
        assert middleware.client is agent.client
        assert middleware.tools is agent.tools
