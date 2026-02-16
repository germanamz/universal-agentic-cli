"""Capability detection and cognitive polyfilling."""

from uac.core.polyfills.capabilities import CapabilityProfile, CapabilityRegistry
from uac.core.polyfills.react_injector import ReActInjector
from uac.core.polyfills.react_parser import ReActParser, ReActParseResult
from uac.core.polyfills.registry_data import build_default_registry
from uac.core.polyfills.strategy import (
    NativeStrategy,
    PromptedStrategy,
    ToolCallingStrategy,
)

__all__ = [
    "CapabilityProfile",
    "CapabilityRegistry",
    "NativeStrategy",
    "PromptedStrategy",
    "ReActInjector",
    "ReActParseResult",
    "ReActParser",
    "ToolCallingStrategy",
    "build_default_registry",
]
