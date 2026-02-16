"""Capability detection for LLM models.

Provides a structured profile of model capabilities and a registry
that maps model identifiers to their profiles. Used by the polyfill
layer to decide between native tool calling and ReAct prompting.
"""

from typing import Any

from pydantic import BaseModel

from uac.core.interface.config import ModelConfig


class CapabilityProfile(BaseModel):
    """Structured representation of a model's capabilities."""

    supports_native_tools: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_streaming: bool = True
    context_window: int = 4096

    @classmethod
    def from_capabilities_dict(cls, caps: dict[str, bool], **defaults: Any) -> "CapabilityProfile":
        """Build a profile from a flat capabilities dict.

        Keys recognised: ``native_tool_calling``, ``vision``, ``audio``,
        ``streaming``.  Unknown keys are silently ignored.
        """
        mapping: dict[str, str] = {
            "native_tool_calling": "supports_native_tools",
            "vision": "supports_vision",
            "audio": "supports_audio",
            "streaming": "supports_streaming",
        }
        kwargs: dict[str, Any] = dict(defaults)
        for src_key, dst_key in mapping.items():
            if src_key in caps:
                kwargs[dst_key] = caps[src_key]
        return cls(**kwargs)


class CapabilityRegistry:
    """Maps model identifiers to their capability profiles."""

    def __init__(self) -> None:
        self._models: dict[str, CapabilityProfile] = {}

    def register(self, model_id: str, profile: CapabilityProfile) -> None:
        """Register a profile for *model_id*."""
        self._models[model_id] = profile

    def resolve(self, config: ModelConfig) -> CapabilityProfile:
        """Resolve the capability profile for *config*.

        Lookup order:
        1. Full model string (e.g. ``openai/gpt-4o``)
        2. Model name only (e.g. ``gpt-4o``)
        3. Default profile (no native tools)

        If ``config.capabilities`` is non-empty its values override the
        resolved profile fields.
        """
        model = config.model
        name_only = model.split("/", 1)[1] if "/" in model else model

        profile = self._models.get(model) or self._models.get(name_only) or CapabilityProfile()

        # Apply user overrides from config.capabilities
        if config.capabilities:
            profile = CapabilityProfile.from_capabilities_dict(
                config.capabilities,
                **profile.model_dump(),
            )

        return profile
