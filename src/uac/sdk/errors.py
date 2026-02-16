"""SDK error types."""

from __future__ import annotations


class WorkflowValidationError(Exception):
    """Raised when a workflow YAML fails parsing or validation."""
