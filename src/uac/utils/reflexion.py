"""Reflexion loops — self-correction middleware for agent outputs.

``ReflexionMiddleware`` wraps an :class:`AgentNode` and validates each
``step()`` output.  When validation fails it injects the error as
feedback into the context and retries, up to a configurable maximum.

Usage::

    from uac.utils.reflexion import ReflexionMiddleware, NonEmptyValidator

    wrapped = ReflexionMiddleware(agent, validators=[NonEmptyValidator()])
    delta = await wrapped.step(context)   # retries automatically on failure
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from opentelemetry import trace

from uac.core.blackboard.models import ContextSlice, StateDelta, TraceEntry

if TYPE_CHECKING:
    from uac.core.orchestration.primitives import AgentNode
from uac.utils.telemetry import (
    ATTR_AGENT_ID,
    ATTR_REFLEXION_ATTEMPT,
    ATTR_REFLEXION_MAX_RETRIES,
    get_tracer,
)

_tracer = get_tracer(__name__)


# ---------------------------------------------------------------------------
# Validator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OutputValidator(Protocol):
    """Validates a :class:`StateDelta` produced by an agent step.

    Return an empty list if valid, or a list of error messages otherwise.
    """

    def validate(self, delta: StateDelta) -> list[str]: ...


# ---------------------------------------------------------------------------
# Built-in validators
# ---------------------------------------------------------------------------


class NonEmptyValidator:
    """Rejects deltas whose response text is empty or whitespace-only."""

    def validate(self, delta: StateDelta) -> list[str]:
        for entry in delta.trace_entries:
            text = entry.data.get("text", "")
            if isinstance(text, str) and text.strip():
                return []
        return ["Agent produced an empty response."]


class JsonContentValidator:
    """Rejects deltas whose response text is not valid JSON.

    Only applies when the ``require_json`` flag is set (default ``True``).
    Useful for agents expected to produce structured output.
    """

    def validate(self, delta: StateDelta) -> list[str]:
        for entry in delta.trace_entries:
            text = entry.data.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                json.loads(text)
                return []
            except json.JSONDecodeError as exc:
                return [f"Response is not valid JSON: {exc}"]
        return ["No response text found to validate as JSON."]


class SchemaValidator:
    """Rejects deltas whose response text does not match a JSON Schema.

    Requires the response to be valid JSON and to match the provided schema
    dictionary.  Uses a minimal validation approach — checks for required
    top-level keys only to avoid heavy dependencies.
    """

    def __init__(self, required_keys: list[str]) -> None:
        self._required_keys = required_keys

    def validate(self, delta: StateDelta) -> list[str]:
        for entry in delta.trace_entries:
            text = entry.data.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                data: Any = json.loads(text)
            except json.JSONDecodeError as exc:
                return [f"Response is not valid JSON: {exc}"]
            if not isinstance(data, dict):
                return ["Response JSON is not an object."]
            missing = [k for k in self._required_keys if k not in data]
            if missing:
                return [f"Missing required keys: {', '.join(missing)}"]
            return []
        return ["No response text found to validate."]


# ---------------------------------------------------------------------------
# ReflexionMiddleware
# ---------------------------------------------------------------------------


class ReflexionMiddleware:
    """Wraps an :class:`AgentNode` with automatic retry on validation failure.

    On each call to :meth:`step`:

    1. Delegates to the underlying agent's ``step()``.
    2. Runs all validators on the resulting :class:`StateDelta`.
    3. If any validator fails, injects the error messages into the context
       and retries (up to *max_retries*).
    4. Each retry is logged as a child span.

    Parameters
    ----------
    agent:
        The agent to wrap.
    validators:
        One or more :class:`OutputValidator` instances.
    max_retries:
        Maximum number of retry attempts (default ``3``).
    """

    def __init__(
        self,
        agent: AgentNode,
        validators: list[OutputValidator],
        *,
        max_retries: int = 3,
    ) -> None:
        self.agent = agent
        self.validators = validators
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def manifest(self) -> Any:
        return self.agent.manifest

    @property
    def client(self) -> Any:
        return self.agent.client

    @property
    def tools(self) -> Any:
        return self.agent.tools

    async def step(self, context: ContextSlice) -> StateDelta:
        """Execute a step with automatic reflexion on validation failure."""
        with _tracer.start_as_current_span("reflexion.step") as span:
            span.set_attribute(ATTR_AGENT_ID, self.agent.name)
            span.set_attribute(ATTR_REFLEXION_MAX_RETRIES, self.max_retries)

            current_context = context
            last_delta: StateDelta | None = None

            for attempt in range(self.max_retries + 1):
                span.set_attribute(ATTR_REFLEXION_ATTEMPT, attempt)

                delta = await self.agent.step(current_context)
                last_delta = delta

                errors = self._validate(delta)
                if not errors:
                    if attempt > 0:
                        span.add_event(
                            "reflexion.succeeded",
                            {"attempt": attempt},
                        )
                    return delta

                # Log the retry
                span.add_event(
                    "reflexion.retry",
                    {
                        "attempt": attempt,
                        "errors": "; ".join(errors),
                    },
                )
                span.set_status(trace.StatusCode.OK)

                # Don't retry after the last attempt
                if attempt == self.max_retries:
                    span.add_event("reflexion.exhausted")
                    break

                # Inject error feedback into context for the next attempt
                error_feedback = (
                    f"Your previous response (attempt {attempt + 1}) had errors: "
                    + "; ".join(errors)
                    + ". Please fix these issues."
                )
                error_trace = TraceEntry(
                    agent_id=self.agent.name,
                    action="reflexion_error",
                    data={"errors": errors, "attempt": attempt + 1},
                )
                current_context = ContextSlice(
                    belief_state=context.belief_state + "\n\n" + error_feedback,
                    trace=[*context.trace, error_trace],
                    artifacts=context.artifacts,
                    pending_tasks=context.pending_tasks,
                )

            # Return the last delta even if validation failed
            assert last_delta is not None
            return last_delta

    def _validate(self, delta: StateDelta) -> list[str]:
        """Run all validators and collect error messages."""
        errors: list[str] = []
        for validator in self.validators:
            errors.extend(validator.validate(delta))
        return errors
