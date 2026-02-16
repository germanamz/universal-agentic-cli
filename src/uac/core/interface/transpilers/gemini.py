"""Gemini transpiler — maps assistant->model role and tool calls to FunctionCall.

Key differences from CMS:
- Role "assistant" becomes "model".
- Tool calls use Gemini's FunctionCall/FunctionResponse structure.
- System instructions are passed via a separate "system_instruction" field.
"""

from typing import Any

from uac.core.interface.models import (
    AudioContent,
    CanonicalMessage,
    ContentPart,
    ConversationHistory,
    ImageContent,
    TextContent,
    ToolCall,
)


class GeminiTranspiler:
    """Converts between CMS and Gemini's generateContent format."""

    def to_provider(self, history: ConversationHistory) -> dict[str, Any]:
        """Convert CMS history to Gemini format.

        Returns {"system_instruction": ..., "contents": [...]} with role
        mapping and FunctionCall structure.
        """
        result: dict[str, Any] = {}

        # Extract system messages into system_instruction
        system_parts = [msg.text for msg in history.system_messages]
        if system_parts:
            result["system_instruction"] = {
                "parts": [{"text": text} for text in system_parts]
            }

        # Convert non-system messages
        contents: list[dict[str, Any]] = []
        for msg in history.non_system_messages:
            contents.append(self._message_to_gemini(msg))

        result["contents"] = contents
        return result

    def from_provider(self, response: dict[str, Any]) -> CanonicalMessage:
        """Convert a Gemini generateContent response to a CanonicalMessage."""
        candidate = response["candidates"][0]
        parts = candidate["content"]["parts"]

        content: list[ContentPart] = []
        tool_calls: list[ToolCall] | None = None

        for part in parts:
            if "text" in part:
                content.append(TextContent(text=part["text"]))
            elif "functionCall" in part:
                if tool_calls is None:
                    tool_calls = []
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        name=fc["name"],
                        arguments=fc.get("args", {}),
                    )
                )

        metadata: dict[str, Any] = {}
        if response.get("usageMetadata"):
            metadata["usage"] = response["usageMetadata"]
        metadata["finish_reason"] = candidate.get("finishReason")

        return CanonicalMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            metadata=metadata,
        )

    def _message_to_gemini(self, msg: CanonicalMessage) -> dict[str, Any]:
        """Convert a single CMS message to Gemini format."""
        # Role mapping: assistant -> model, tool -> user (with FunctionResponse)
        if msg.role == "tool":
            return {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": msg.tool_call_id or "",
                            "response": {"content": msg.text},
                        }
                    }
                ],
            }

        role = "model" if msg.role == "assistant" else msg.role
        parts: list[dict[str, Any]] = []

        # Convert content parts
        for part in msg.content:
            parts.append(self._content_part_to_gemini(part))

        # Convert tool calls to FunctionCall parts
        if msg.tool_calls:
            for tc in msg.tool_calls:
                parts.append(
                    {
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    }
                )

        return {"role": role, "parts": parts}

    def _content_part_to_gemini(self, part: ContentPart) -> dict[str, Any]:
        """Convert a ContentPart to Gemini's parts format."""
        if isinstance(part, TextContent):
            return {"text": part.text}
        if isinstance(part, ImageContent):
            if part.data:
                return {
                    "inline_data": {
                        "mime_type": part.media_type or "image/png",
                        "data": part.data,
                    }
                }
            return {"text": f"[Image: {part.url}]"}
        # AudioContent is the only remaining possibility
        audio: AudioContent = part
        if audio.data:
            return {
                "inline_data": {
                    "mime_type": audio.media_type or "audio/wav",
                    "data": audio.data,
                }
            }
        return {"text": f"[Audio: {audio.url}]"}
