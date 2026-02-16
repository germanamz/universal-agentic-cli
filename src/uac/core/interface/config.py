"""Model configuration — provider, model name, capability flags."""

from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a specific model/provider combination.

    The ``model`` field uses LiteLLM's naming convention:
    ``provider/model_name`` (e.g. ``openai/gpt-4``, ``anthropic/claude-3-opus``).
    """

    model: str
    api_key: str | None = None
    api_base: str | None = None
    context_window: int | None = None
    capabilities: dict[str, bool] = Field(default_factory=lambda: dict[str, bool]())
    extra: dict[str, Any] = Field(default_factory=lambda: dict[str, Any]())

    @property
    def provider(self) -> str:
        """Extract the provider prefix from the model string."""
        if "/" in self.model:
            return self.model.split("/", 1)[0]
        return "openai"
