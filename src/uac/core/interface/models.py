"""Canonical Message Schema (CMS) — the internal message format for UAC.

The CMS is a superset of all provider capabilities, ensuring that orchestration
logic never touches provider-specific formats. Runtime transpilers convert
CMS messages to/from provider-specific payloads.
"""

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Content Parts — multimodal content building blocks
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    """Plain text content part."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content part (URL or inline base64)."""

    type: Literal["image"] = "image"
    url: str | None = None
    data: str | None = None
    media_type: str | None = None


class AudioContent(BaseModel):
    """Audio content part (URL or inline base64)."""

    type: Literal["audio"] = "audio"
    url: str | None = None
    data: str | None = None
    media_type: str | None = None


ContentPart = TextContent | ImageContent | AudioContent


# ---------------------------------------------------------------------------
# Tool Calling — structured tool invocations and results
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A tool invocation emitted by an assistant message."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    arguments: dict[str, Any] = {}


class ToolResult(BaseModel):
    """The result of executing a tool, returned as a tool-role message."""

    tool_call_id: str
    content: list[ContentPart] = []

    @classmethod
    def from_text(cls, tool_call_id: str, text: str) -> "ToolResult":
        """Create a ToolResult with a single text content part."""
        parts: list[ContentPart] = [TextContent(text=text)]
        return cls(tool_call_id=tool_call_id, content=parts)


# ---------------------------------------------------------------------------
# Canonical Message — the core message type
# ---------------------------------------------------------------------------


class CanonicalMessage(BaseModel):
    """A single message in the canonical format.

    Roles:
    - system: instruction/context messages
    - user: human input
    - assistant: LLM-generated messages (may include tool_calls)
    - tool: tool execution results (must include tool_call_id)
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: list[ContentPart] = []
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = {}

    @property
    def text(self) -> str:
        """Extract concatenated text from all TextContent parts."""
        return "".join(part.text for part in self.content if isinstance(part, TextContent))

    @classmethod
    def system(cls, text: str, **metadata: Any) -> "CanonicalMessage":
        """Create a system message."""
        parts: list[ContentPart] = [TextContent(text=text)]
        return cls(role="system", content=parts, metadata=metadata)

    @classmethod
    def user(cls, text: str, **metadata: Any) -> "CanonicalMessage":
        """Create a user message."""
        parts: list[ContentPart] = [TextContent(text=text)]
        return cls(role="user", content=parts, metadata=metadata)

    @classmethod
    def assistant(
        cls,
        text: str = "",
        tool_calls: list[ToolCall] | None = None,
        **metadata: Any,
    ) -> "CanonicalMessage":
        """Create an assistant message."""
        content: list[ContentPart] = [TextContent(text=text)] if text else []
        return cls(role="assistant", content=content, tool_calls=tool_calls, metadata=metadata)

    @classmethod
    def tool(cls, result: ToolResult, **metadata: Any) -> "CanonicalMessage":
        """Create a tool-result message."""
        return cls(
            role="tool",
            content=list(result.content),
            tool_call_id=result.tool_call_id,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Conversation History — ordered container of messages
# ---------------------------------------------------------------------------


class ConversationHistory(BaseModel):
    """An ordered sequence of canonical messages forming a conversation."""

    messages: list[CanonicalMessage] = []

    def append(self, message: CanonicalMessage) -> None:
        """Append a message to the history."""
        self.messages.append(message)

    @property
    def system_messages(self) -> list[CanonicalMessage]:
        """Return all system messages."""
        return [m for m in self.messages if m.role == "system"]

    @property
    def non_system_messages(self) -> list[CanonicalMessage]:
        """Return all non-system messages (for providers that separate system prompts)."""
        return [m for m in self.messages if m.role != "system"]

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):  # type: ignore[override]
        return iter(self.messages)
