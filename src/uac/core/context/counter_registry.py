"""Counter registry — auto-selects the best TokenCounter for a model config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from uac.core.context.counter import EstimatingCounter, TiktokenCounter, TokenCounter

if TYPE_CHECKING:
    from uac.core.interface.config import ModelConfig

# Providers whose tokenization is well-served by tiktoken.
_TIKTOKEN_PROVIDERS = frozenset({"openai", "azure", "azure_ai"})


def get_counter(config: ModelConfig) -> TokenCounter:
    """Return the most appropriate TokenCounter for the given model config.

    Uses tiktoken for OpenAI/Azure models and the estimating fallback for
    everything else (Anthropic, Gemini, local models, etc.).
    """
    if config.provider in _TIKTOKEN_PROVIDERS:
        # Strip provider prefix — tiktoken expects bare model names.
        model_name = config.model.split("/", 1)[-1] if "/" in config.model else config.model
        return TiktokenCounter(model_name)
    return EstimatingCounter()
