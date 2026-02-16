"""Static model registry data.

Contains known model profiles and a helper to build a pre-loaded
``CapabilityRegistry``.
"""

from uac.core.polyfills.capabilities import CapabilityProfile, CapabilityRegistry

# ---------------------------------------------------------------------------
# Known model profiles
# ---------------------------------------------------------------------------

KNOWN_MODELS: dict[str, CapabilityProfile] = {
    # OpenAI
    "openai/gpt-4o": CapabilityProfile(
        supports_native_tools=True,
        supports_vision=True,
        context_window=128_000,
    ),
    "openai/gpt-4-turbo": CapabilityProfile(
        supports_native_tools=True,
        supports_vision=True,
        context_window=128_000,
    ),
    "openai/gpt-3.5-turbo": CapabilityProfile(
        supports_native_tools=True,
        context_window=16_385,
    ),
    # Anthropic
    "anthropic/claude-3-opus": CapabilityProfile(
        supports_native_tools=True,
        supports_vision=True,
        context_window=200_000,
    ),
    "anthropic/claude-3-sonnet": CapabilityProfile(
        supports_native_tools=True,
        supports_vision=True,
        context_window=200_000,
    ),
    "anthropic/claude-3-haiku": CapabilityProfile(
        supports_native_tools=True,
        supports_vision=True,
        context_window=200_000,
    ),
    # Gemini
    "gemini/gemini-1.5-pro": CapabilityProfile(
        supports_native_tools=True,
        supports_vision=True,
        supports_audio=True,
        context_window=1_000_000,
    ),
    "gemini/gemini-1.5-flash": CapabilityProfile(
        supports_native_tools=True,
        supports_vision=True,
        supports_audio=True,
        context_window=1_000_000,
    ),
    "gemini/gemini-2.0-flash": CapabilityProfile(
        supports_native_tools=True,
        supports_vision=True,
        supports_audio=True,
        context_window=1_000_000,
    ),
    # Local / open-weight (no native tool calling)
    "ollama/llama-3-8b": CapabilityProfile(
        supports_native_tools=False,
        context_window=8_192,
    ),
    "ollama/llama-3-70b": CapabilityProfile(
        supports_native_tools=False,
        context_window=8_192,
    ),
    "ollama/mistral-7b": CapabilityProfile(
        supports_native_tools=False,
        context_window=8_192,
    ),
    "ollama/mixtral-8x7b": CapabilityProfile(
        supports_native_tools=False,
        context_window=32_768,
    ),
}


def build_default_registry() -> CapabilityRegistry:
    """Return a ``CapabilityRegistry`` pre-loaded with known models."""
    registry = CapabilityRegistry()
    for model_id, profile in KNOWN_MODELS.items():
        registry.register(model_id, profile)
    return registry
