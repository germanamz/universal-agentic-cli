"""Tests for provider-specific transpilers using recorded fixtures."""

from typing import Any

from uac.core.interface.models import (
    CanonicalMessage,
    ContentPart,
    ConversationHistory,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResult,
)
from uac.core.interface.transpilers.anthropic import AnthropicTranspiler
from uac.core.interface.transpilers.gemini import GeminiTranspiler
from uac.core.interface.transpilers.openai import OpenAITranspiler

# ---------------------------------------------------------------------------
# Fixtures: sample conversations
# ---------------------------------------------------------------------------


def _simple_history() -> ConversationHistory:
    return ConversationHistory(
        messages=[
            CanonicalMessage.system("You are helpful."),
            CanonicalMessage.user("Hello"),
            CanonicalMessage.assistant("Hi there!"),
        ]
    )


def _tool_call_history() -> ConversationHistory:
    tc = ToolCall(id="call-1", name="calculator", arguments={"expression": "2+2"})
    result = ToolResult.from_text("call-1", "4")
    return ConversationHistory(
        messages=[
            CanonicalMessage.system("You can use tools."),
            CanonicalMessage.user("What is 2+2?"),
            CanonicalMessage.assistant("Let me calculate.", tool_calls=[tc]),
            CanonicalMessage.tool(result),
            CanonicalMessage.assistant("The answer is 4."),
        ]
    )


def _multimodal_history() -> ConversationHistory:
    parts: list[ContentPart] = [
        TextContent(text="What's in this image?"),
        ImageContent(url="https://example.com/cat.png", media_type="image/png"),
    ]
    return ConversationHistory(
        messages=[
            CanonicalMessage.system("You can see images."),
            CanonicalMessage(role="user", content=parts),
        ]
    )


def _consecutive_user_history() -> ConversationHistory:
    """Two consecutive user messages — needs merging for Anthropic."""
    return ConversationHistory(
        messages=[
            CanonicalMessage.user("First message"),
            CanonicalMessage.user("Second message"),
            CanonicalMessage.assistant("Got both."),
        ]
    )


# ---------------------------------------------------------------------------
# OpenAI Transpiler Tests
# ---------------------------------------------------------------------------


class TestOpenAITranspiler:
    def setup_method(self) -> None:
        self.transpiler = OpenAITranspiler()

    def test_simple_to_provider(self) -> None:
        payload = self.transpiler.to_provider(_simple_history())
        messages = payload["messages"]
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2] == {"role": "assistant", "content": "Hi there!"}

    def test_tool_call_to_provider(self) -> None:
        payload = self.transpiler.to_provider(_tool_call_history())
        messages = payload["messages"]
        # assistant with tool_calls
        assistant_msg = messages[2]
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "calculator"
        # tool result
        tool_msg = messages[3]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call-1"
        assert tool_msg["content"] == "4"

    def test_multimodal_to_provider(self) -> None:
        payload = self.transpiler.to_provider(_multimodal_history())
        user_msg = payload["messages"][1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"

    def test_from_provider_text_response(self) -> None:
        response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        msg = self.transpiler.from_provider(response)
        assert msg.role == "assistant"
        assert msg.text == "Hello!"
        assert msg.metadata["usage"]["prompt_tokens"] == 10

    def test_from_provider_tool_calls(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc-1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "hello"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        msg = self.transpiler.from_provider(response)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"
        assert msg.tool_calls[0].arguments == {"query": "hello"}


# ---------------------------------------------------------------------------
# Anthropic Transpiler Tests
# ---------------------------------------------------------------------------


class TestAnthropicTranspiler:
    def setup_method(self) -> None:
        self.transpiler = AnthropicTranspiler()

    def test_system_extraction(self) -> None:
        payload = self.transpiler.to_provider(_simple_history())
        assert payload["system"] == "You are helpful."
        # System message should not be in the messages array
        for msg in payload["messages"]:
            assert msg["role"] != "system"

    def test_simple_messages(self) -> None:
        payload = self.transpiler.to_provider(_simple_history())
        messages = payload["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_consecutive_merge(self) -> None:
        payload = self.transpiler.to_provider(_consecutive_user_history())
        messages = payload["messages"]
        # Two user messages should be merged into one
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        # Merged content should be a list
        assert isinstance(messages[0]["content"], list)
        texts = [b["text"] for b in messages[0]["content"] if b["type"] == "text"]
        assert "First message" in texts
        assert "Second message" in texts

    def test_tool_result_as_user_message(self) -> None:
        payload = self.transpiler.to_provider(_tool_call_history())
        messages = payload["messages"]
        # Find the tool result — should be a user message
        tool_msg = next(
            m
            for m in messages
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m["content"]
            )
        )
        assert tool_msg["content"][0]["tool_use_id"] == "call-1"

    def test_tool_use_in_assistant(self) -> None:
        payload = self.transpiler.to_provider(_tool_call_history())
        messages = payload["messages"]
        # Find assistant message with tool_use
        assistant_with_tools = next(
            m
            for m in messages
            if m["role"] == "assistant"
            and isinstance(m["content"], list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in m["content"]
            )
        )
        tool_use = next(
            b for b in assistant_with_tools["content"] if b["type"] == "tool_use"
        )
        assert tool_use["name"] == "calculator"
        assert tool_use["id"] == "call-1"

    def test_from_provider_text(self) -> None:
        response: dict[str, Any] = {
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        msg = self.transpiler.from_provider(response)
        assert msg.text == "Hello!"
        assert msg.metadata["stop_reason"] == "end_turn"

    def test_from_provider_tool_use(self) -> None:
        response: dict[str, Any] = {
            "content": [
                {"type": "text", "text": "Let me search."},
                {
                    "type": "tool_use",
                    "id": "tu-1",
                    "name": "search",
                    "input": {"query": "hello"},
                },
            ],
            "stop_reason": "tool_use",
        }
        msg = self.transpiler.from_provider(response)
        assert msg.text == "Let me search."
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].name == "search"

    def test_multimodal_image(self) -> None:
        payload = self.transpiler.to_provider(_multimodal_history())
        user_msg = payload["messages"][0]
        assert isinstance(user_msg["content"], list)
        image_block = next(b for b in user_msg["content"] if b["type"] == "image")
        assert image_block["source"]["type"] == "url"


# ---------------------------------------------------------------------------
# Gemini Transpiler Tests
# ---------------------------------------------------------------------------


class TestGeminiTranspiler:
    def setup_method(self) -> None:
        self.transpiler = GeminiTranspiler()

    def test_system_instruction(self) -> None:
        payload = self.transpiler.to_provider(_simple_history())
        assert "system_instruction" in payload
        parts = payload["system_instruction"]["parts"]
        assert any(p["text"] == "You are helpful." for p in parts)

    def test_role_mapping(self) -> None:
        payload = self.transpiler.to_provider(_simple_history())
        contents = payload["contents"]
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"  # assistant -> model

    def test_function_call_structure(self) -> None:
        payload = self.transpiler.to_provider(_tool_call_history())
        contents = payload["contents"]
        # Find the assistant message with functionCall
        model_msg = next(
            c
            for c in contents
            if c["role"] == "model"
            and any("functionCall" in p for p in c["parts"])
        )
        fc_part = next(p for p in model_msg["parts"] if "functionCall" in p)
        assert fc_part["functionCall"]["name"] == "calculator"
        assert fc_part["functionCall"]["args"] == {"expression": "2+2"}

    def test_function_response(self) -> None:
        payload = self.transpiler.to_provider(_tool_call_history())
        contents = payload["contents"]
        # Tool results become user messages with functionResponse
        fn_resp_msg = next(
            c
            for c in contents
            if c["role"] == "user"
            and any("functionResponse" in p for p in c["parts"])
        )
        fr = fn_resp_msg["parts"][0]["functionResponse"]
        assert fr["response"]["content"] == "4"

    def test_from_provider_text(self) -> None:
        response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }
        msg = self.transpiler.from_provider(response)
        assert msg.role == "assistant"
        assert msg.text == "Hello!"
        assert msg.metadata["usage"]["promptTokenCount"] == 10

    def test_from_provider_function_call(self) -> None:
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"query": "hello"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        msg = self.transpiler.from_provider(response)
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].name == "search"
        assert msg.tool_calls[0].arguments == {"query": "hello"}

    def test_no_system_when_absent(self) -> None:
        history = ConversationHistory(
            messages=[CanonicalMessage.user("Hello")]
        )
        payload = self.transpiler.to_provider(history)
        assert "system_instruction" not in payload
