"""OpenAI transpiler — CMS is closest to ChatML so this is the simplest mapping."""

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


class OpenAITranspiler:
    """Converts between CMS and OpenAI's chat completion format."""

    def to_provider(self, history: ConversationHistory) -> dict[str, Any]:
        """Convert CMS history to OpenAI messages format.

        Returns {"messages": [...]} where each message follows OpenAI's schema.
        """
        messages: list[dict[str, Any]] = []
        for msg in history:
            messages.append(self._message_to_openai(msg))
        return {"messages": messages}

    def from_provider(self, response: dict[str, Any]) -> CanonicalMessage:
        """Convert an OpenAI chat completion response to a CanonicalMessage."""
        choice = response["choices"][0]
        message = choice["message"]

        content: list[ContentPart] = []
        if message.get("content"):
            content = [TextContent(text=message["content"])]

        tool_calls: list[ToolCall] | None = None
        if message.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=_parse_arguments(tc["function"].get("arguments", "{}")),
                )
                for tc in message["tool_calls"]
            ]

        metadata: dict[str, Any] = {}
        if response.get("usage"):
            metadata["usage"] = response["usage"]
        metadata["finish_reason"] = choice.get("finish_reason")

        return CanonicalMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            metadata=metadata,
        )

    def _message_to_openai(self, msg: CanonicalMessage) -> dict[str, Any]:
        """Convert a single CMS message to OpenAI format."""
        result: dict[str, Any] = {"role": msg.role}

        if msg.role == "tool":
            result["tool_call_id"] = msg.tool_call_id
            result["content"] = msg.text
            return result

        # Handle multimodal content
        if len(msg.content) == 1 and isinstance(msg.content[0], TextContent):
            result["content"] = msg.content[0].text
        elif msg.content:
            result["content"] = [self._content_part_to_openai(p) for p in msg.content]
        else:
            result["content"] = None

        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _serialize_arguments(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]

        return result

    def _content_part_to_openai(self, part: ContentPart) -> dict[str, Any]:
        """Convert a ContentPart to OpenAI's content array format."""
        if isinstance(part, TextContent):
            return {"type": "text", "text": part.text}
        if isinstance(part, ImageContent):
            url = part.url
            if part.data and part.media_type:
                url = f"data:{part.media_type};base64,{part.data}"
            return {
                "type": "image_url",
                "image_url": {"url": url},
            }
        # AudioContent is the only remaining possibility
        audio: AudioContent = part
        return {
            "type": "input_audio",
            "input_audio": {
                "data": audio.data or "",
                "format": _audio_format(audio.media_type),
            },
        }


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Parse JSON string arguments from OpenAI response."""
    import json

    try:
        result: dict[str, Any] = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        result = {"raw": raw}
    return result


def _serialize_arguments(args: dict[str, Any]) -> str:
    """Serialize tool call arguments to JSON string for OpenAI."""
    import json

    return json.dumps(args)


def _audio_format(media_type: str | None) -> str:
    """Extract audio format from media type."""
    if media_type and "/" in media_type:
        return media_type.split("/")[1]
    return "wav"
