"""Anthropic transpiler — handles system message extraction and role alternation.

Key differences from CMS:
- System message is a separate top-level parameter, not in the messages array.
- Messages must strictly alternate between user and assistant roles.
- Consecutive same-role messages must be merged.
- Tool results are embedded as user messages with tool_result content blocks.
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


class AnthropicTranspiler:
    """Converts between CMS and Anthropic's messages API format."""

    def to_provider(self, history: ConversationHistory) -> dict[str, Any]:
        """Convert CMS history to Anthropic format.

        Returns {"system": "...", "messages": [...]} with system prompt
        extracted and consecutive same-role messages merged.
        """
        result: dict[str, Any] = {}

        # Extract system messages into top-level parameter
        system_parts = [msg.text for msg in history.system_messages]
        if system_parts:
            result["system"] = "\n\n".join(system_parts)

        # Convert non-system messages
        raw_messages: list[dict[str, Any]] = []
        for msg in history.non_system_messages:
            raw_messages.append(self._message_to_anthropic(msg))

        # Merge consecutive same-role messages
        result["messages"] = _merge_consecutive_roles(raw_messages)

        return result

    def from_provider(self, response: dict[str, Any]) -> CanonicalMessage:
        """Convert an Anthropic messages API response to a CanonicalMessage."""
        content_blocks = response.get("content", [])

        content: list[ContentPart] = []
        tool_calls: list[ToolCall] | None = None

        for block in content_blocks:
            if block["type"] == "text":
                content.append(TextContent(text=block["text"]))
            elif block["type"] == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )

        metadata: dict[str, Any] = {}
        if response.get("usage"):
            metadata["usage"] = response["usage"]
        metadata["stop_reason"] = response.get("stop_reason")

        return CanonicalMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            metadata=metadata,
        )

    def _message_to_anthropic(self, msg: CanonicalMessage) -> dict[str, Any]:
        """Convert a single CMS message to Anthropic format."""
        if msg.role == "tool":
            # Tool results become user messages with tool_result content blocks
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.text,
                    }
                ],
            }

        if msg.role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if msg.text:
                content_blocks.append({"type": "text", "text": msg.text})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
            return {"role": "assistant", "content": content_blocks}

        # User messages
        content = self._content_to_anthropic(msg.content)
        return {"role": "user", "content": content}

    def _content_to_anthropic(
        self, parts: list[ContentPart]
    ) -> str | list[dict[str, Any]]:
        """Convert content parts to Anthropic format.

        Returns a plain string for simple text, or content blocks for multimodal.
        """
        if len(parts) == 1 and isinstance(parts[0], TextContent):
            return parts[0].text

        blocks: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextContent):
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContent):
                source: dict[str, Any]
                if part.data:
                    source = {
                        "type": "base64",
                        "media_type": part.media_type or "image/png",
                        "data": part.data,
                    }
                else:
                    source = {"type": "url", "url": part.url}
                blocks.append({"type": "image", "source": source})
            else:
                # AudioContent — Anthropic doesn't natively support audio
                audio: AudioContent = part
                blocks.append(
                    {"type": "text", "text": f"[Audio: {audio.url or 'inline'}]"}
                )
        return blocks


def _merge_consecutive_roles(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge consecutive messages with the same role.

    Anthropic requires strict user/assistant alternation. When multiple
    consecutive messages share a role, their content is merged into one message.
    """
    if not messages:
        return []

    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            # Merge content into the previous message
            prev_content = merged[-1]["content"]
            new_content = msg["content"]
            merged[-1]["content"] = _merge_content(prev_content, new_content)
        else:
            merged.append(msg)
    return merged


def _merge_content(
    existing: str | list[dict[str, Any]], new: str | list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge two content values (str or list of blocks) into a single list."""
    result: list[dict[str, Any]] = []
    for item in (existing, new):
        if isinstance(item, str):
            result.append({"type": "text", "text": item})
        else:
            result.extend(item)
    return result
