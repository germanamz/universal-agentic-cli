"""Tests for CapabilityProfile and CapabilityRegistry."""

from uac.core.interface.config import ModelConfig
from uac.core.polyfills.capabilities import CapabilityProfile, CapabilityRegistry
from uac.core.polyfills.registry_data import KNOWN_MODELS, build_default_registry


class TestCapabilityProfile:
    def test_defaults(self) -> None:
        p = CapabilityProfile()
        assert p.supports_native_tools is False
        assert p.supports_vision is False
        assert p.supports_audio is False
        assert p.supports_streaming is True
        assert p.context_window == 4096

    def test_from_capabilities_dict(self) -> None:
        caps = {"native_tool_calling": True, "vision": True}
        p = CapabilityProfile.from_capabilities_dict(caps)
        assert p.supports_native_tools is True
        assert p.supports_vision is True
        assert p.supports_audio is False

    def test_from_capabilities_dict_with_defaults(self) -> None:
        caps = {"vision": True}
        p = CapabilityProfile.from_capabilities_dict(caps, context_window=128_000)
        assert p.supports_vision is True
        assert p.context_window == 128_000

    def test_unknown_keys_ignored(self) -> None:
        caps = {"native_tool_calling": True, "quantum_computing": True}
        p = CapabilityProfile.from_capabilities_dict(caps)
        assert p.supports_native_tools is True


class TestCapabilityRegistry:
    def test_resolve_full_model_string(self) -> None:
        registry = CapabilityRegistry()
        profile = CapabilityProfile(supports_native_tools=True, context_window=128_000)
        registry.register("openai/gpt-4o", profile)

        config = ModelConfig(model="openai/gpt-4o")
        resolved = registry.resolve(config)
        assert resolved.supports_native_tools is True
        assert resolved.context_window == 128_000

    def test_resolve_name_only_fallback(self) -> None:
        registry = CapabilityRegistry()
        profile = CapabilityProfile(supports_native_tools=True)
        registry.register("gpt-4o", profile)

        config = ModelConfig(model="openai/gpt-4o")
        resolved = registry.resolve(config)
        assert resolved.supports_native_tools is True

    def test_resolve_unknown_model_defaults(self) -> None:
        registry = CapabilityRegistry()
        config = ModelConfig(model="some/unknown-model")
        resolved = registry.resolve(config)
        assert resolved.supports_native_tools is False
        assert resolved.context_window == 4096

    def test_config_capabilities_override(self) -> None:
        registry = CapabilityRegistry()
        profile = CapabilityProfile(supports_native_tools=False)
        registry.register("ollama/llama-3-8b", profile)

        config = ModelConfig(
            model="ollama/llama-3-8b",
            capabilities={"native_tool_calling": True},
        )
        resolved = registry.resolve(config)
        assert resolved.supports_native_tools is True

    def test_config_capabilities_partial_override(self) -> None:
        registry = CapabilityRegistry()
        profile = CapabilityProfile(
            supports_native_tools=True,
            supports_vision=True,
            context_window=128_000,
        )
        registry.register("openai/gpt-4o", profile)

        config = ModelConfig(
            model="openai/gpt-4o",
            capabilities={"vision": False},
        )
        resolved = registry.resolve(config)
        assert resolved.supports_native_tools is True
        assert resolved.supports_vision is False
        assert resolved.context_window == 128_000


class TestRegistryData:
    def test_known_models_not_empty(self) -> None:
        assert len(KNOWN_MODELS) > 0

    def test_gpt4o_is_native(self) -> None:
        assert KNOWN_MODELS["openai/gpt-4o"].supports_native_tools is True

    def test_llama_is_not_native(self) -> None:
        assert KNOWN_MODELS["ollama/llama-3-8b"].supports_native_tools is False

    def test_build_default_registry(self) -> None:
        registry = build_default_registry()
        config = ModelConfig(model="openai/gpt-4o")
        profile = registry.resolve(config)
        assert profile.supports_native_tools is True
