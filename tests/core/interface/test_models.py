"""Tests for the Canonical Message Schema (CMS) Pydantic models."""

import json

import pytest
from pydantic import ValidationError

from uac.core.interface.models import (
    AudioContent,
    CanonicalMessage,
    ContentPart,
    ConversationHistory,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResult,
)


class TestContentParts:
    def test_text_content(self) -> None:
        part = TextContent(text="hello")
        assert part.type == "text"
        assert part.text == "hello"

    def test_image_content_url(self) -> None:
        part = ImageContent(url="https://example.com/img.png", media_type="image/png")
        assert part.type == "image"
        assert part.url == "https://example.com/img.png"
        assert part.media_type == "image/png"
        assert part.data is None

    def test_image_content_base64(self) -> None:
        part = ImageContent(data="aGVsbG8=", media_type="image/jpeg")
        assert part.data == "aGVsbG8="
        assert part.url is None

    def test_audio_content(self) -> None:
        part = AudioContent(url="https://example.com/audio.wav", media_type="audio/wav")
        assert part.type == "audio"


class TestToolCall:
    def test_auto_id(self) -> None:
        tc = ToolCall(name="calculator", arguments={"expression": "2+2"})
        assert len(tc.id) == 12
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "2+2"}

    def test_explicit_id(self) -> None:
        tc = ToolCall(id="my-id", name="search", arguments={"query": "hello"})
        assert tc.id == "my-id"

    def test_default_empty_arguments(self) -> None:
        tc = ToolCall(name="noop")
        assert tc.arguments == {}

    def test_unique_ids(self) -> None:
        tc1 = ToolCall(name="a")
        tc2 = ToolCall(name="b")
        assert tc1.id != tc2.id


class TestToolResult:
    def test_from_text(self) -> None:
        result = ToolResult.from_text("call-1", "42")
        assert result.tool_call_id == "call-1"
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "42"

    def test_default_empty_content(self) -> None:
        result = ToolResult(tool_call_id="call-2")
        assert result.content == []


class TestCanonicalMessage:
    def test_system_factory(self) -> None:
        msg = CanonicalMessage.system("You are helpful.")
        assert msg.role == "system"
        assert msg.text == "You are helpful."

    def test_user_factory(self) -> None:
        msg = CanonicalMessage.user("Hello!")
        assert msg.role == "user"
        assert msg.text == "Hello!"

    def test_assistant_factory_text(self) -> None:
        msg = CanonicalMessage.assistant("Sure, I can help.")
        assert msg.role == "assistant"
        assert msg.text == "Sure, I can help."
        assert msg.tool_calls is None

    def test_assistant_factory_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc1", name="calc", arguments={"expr": "1+1"})
        msg = CanonicalMessage.assistant(tool_calls=[tc])
        assert msg.role == "assistant"
        assert msg.text == ""
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "calc"

    def test_tool_factory(self) -> None:
        result = ToolResult.from_text("tc1", "42")
        msg = CanonicalMessage.tool(result)
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc1"
        assert msg.text == "42"

    def test_metadata(self) -> None:
        msg = CanonicalMessage.user("hi", tokens=10, latency_ms=50)
        assert msg.metadata["tokens"] == 10
        assert msg.metadata["latency_ms"] == 50

    def test_empty_content(self) -> None:
        msg = CanonicalMessage(role="assistant")
        assert msg.content == []
        assert msg.text == ""

    def test_multimodal_content(self) -> None:
        parts: list[ContentPart] = [
            TextContent(text="Look at this:"),
            ImageContent(url="https://example.com/img.png"),
        ]
        msg = CanonicalMessage(role="user", content=parts)
        assert len(msg.content) == 2
        assert msg.text == "Look at this:"

    def test_multiple_tool_calls(self) -> None:
        calls = [
            ToolCall(id="a", name="search", arguments={"q": "foo"}),
            ToolCall(id="b", name="calc", arguments={"expr": "1+1"}),
        ]
        msg = CanonicalMessage.assistant(tool_calls=calls)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 2

    def test_invalid_role(self) -> None:
        with pytest.raises(ValidationError):
            CanonicalMessage(role="invalid")  # type: ignore[arg-type]


class TestConversationHistory:
    def test_empty(self) -> None:
        history = ConversationHistory()
        assert len(history) == 0

    def test_append(self) -> None:
        history = ConversationHistory()
        history.append(CanonicalMessage.system("Be helpful."))
        history.append(CanonicalMessage.user("Hello"))
        assert len(history) == 2

    def test_system_messages_filter(self) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("sys1"),
                CanonicalMessage.user("hello"),
                CanonicalMessage.system("sys2"),
            ]
        )
        assert len(history.system_messages) == 2
        assert len(history.non_system_messages) == 1

    def test_iteration(self) -> None:
        msgs = [CanonicalMessage.user("a"), CanonicalMessage.user("b")]
        history = ConversationHistory(messages=msgs)
        collected = list(history)
        assert len(collected) == 2
        assert collected[0].text == "a"


class TestSerialization:
    def test_message_round_trip(self) -> None:
        msg = CanonicalMessage.assistant(
            "Hello",
            tool_calls=[ToolCall(id="tc1", name="calc", arguments={"x": 1})],
            tokens=5,
        )
        data = msg.model_dump()
        restored = CanonicalMessage.model_validate(data)
        assert restored.text == "Hello"
        assert restored.tool_calls is not None
        assert restored.tool_calls[0].name == "calc"
        assert restored.metadata["tokens"] == 5

    def test_history_round_trip_json(self) -> None:
        history = ConversationHistory(
            messages=[
                CanonicalMessage.system("You are helpful."),
                CanonicalMessage.user("Hi"),
                CanonicalMessage.assistant("Hello!"),
            ]
        )
        json_str = history.model_dump_json()
        restored = ConversationHistory.model_validate_json(json_str)
        assert len(restored) == 3
        assert restored.messages[0].role == "system"

    def test_multimodal_round_trip(self) -> None:
        parts: list[ContentPart] = [
            TextContent(text="caption"),
            ImageContent(url="https://img.com/a.png", media_type="image/png"),
            AudioContent(data="base64data", media_type="audio/wav"),
        ]
        msg = CanonicalMessage(role="user", content=parts)
        data = msg.model_dump()
        restored = CanonicalMessage.model_validate(data)
        assert len(restored.content) == 3
        assert isinstance(restored.content[0], TextContent)
        assert isinstance(restored.content[1], ImageContent)
        assert isinstance(restored.content[2], AudioContent)

    def test_tool_flow_round_trip(self) -> None:
        """Full tool-call flow: assistant emits calls, tool responds."""
        tc = ToolCall(id="call-1", name="search", arguments={"query": "UAC"})
        assistant_msg = CanonicalMessage.assistant(
            "Let me search for that.", tool_calls=[tc]
        )
        result = ToolResult.from_text("call-1", "Found 3 results")
        tool_msg = CanonicalMessage.tool(result)

        history = ConversationHistory(messages=[assistant_msg, tool_msg])
        json_str = history.model_dump_json()
        parsed = json.loads(json_str)
        restored = ConversationHistory.model_validate(parsed)

        assert restored.messages[0].tool_calls is not None
        assert restored.messages[0].tool_calls[0].id == "call-1"
        assert restored.messages[1].tool_call_id == "call-1"
        assert restored.messages[1].text == "Found 3 results"
