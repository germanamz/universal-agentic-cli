"""Universal Agentic CLI — model-agnostic, multi-agent orchestration platform."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

if TYPE_CHECKING:
    from uac.sdk.workflow import WorkflowLoader as WorkflowLoader
    from uac.sdk.workflow import WorkflowRunner as WorkflowRunner

_SDK_EXPORTS = {
    "WorkflowRunner": "uac.sdk.workflow",
    "WorkflowLoader": "uac.sdk.workflow",
}


def __getattr__(name: str) -> object:
    module_path = _SDK_EXPORTS.get(name)
    if module_path is not None:
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, name)
    raise AttributeError(f"module 'uac' has no attribute {name!r}")
