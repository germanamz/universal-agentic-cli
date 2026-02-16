"""Unified Model Interface (UMI) & Transpilation."""

from uac.core.interface.client import ModelClient
from uac.core.interface.config import ModelConfig
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
from uac.core.interface.transpiler import Transpiler
from uac.core.polyfills.capabilities import CapabilityProfile, CapabilityRegistry
from uac.core.polyfills.strategy import (
    NativeStrategy,
    PromptedStrategy,
    ToolCallingStrategy,
)

__all__ = [
    "AudioContent",
    "CanonicalMessage",
    "CapabilityProfile",
    "CapabilityRegistry",
    "ContentPart",
    "ConversationHistory",
    "ImageContent",
    "ModelClient",
    "ModelConfig",
    "NativeStrategy",
    "PromptedStrategy",
    "TextContent",
    "ToolCall",
    "ToolCallingStrategy",
    "ToolResult",
    "Transpiler",
]
