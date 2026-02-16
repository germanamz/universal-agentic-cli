"""OpenTelemetry tracing helpers for UAC.

Provides a thin wrapper around the OpenTelemetry API so the rest of the
codebase can call ``get_tracer()`` without caring whether the SDK is
installed.  When the SDK is *not* configured the API returns no-op
implementations — zero overhead in production unless explicitly opted in.

Usage::

    from uac.utils.telemetry import get_tracer

    _tracer = get_tracer(__name__)

    with _tracer.start_as_current_span("my.operation") as span:
        span.set_attribute("key", "value")

To activate real tracing, call :func:`configure_telemetry` once at startup
(requires the ``otel`` extra: ``pip install uac[otel]``).
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace

# ---------------------------------------------------------------------------
# Semantic attribute keys used throughout UAC instrumentation
# ---------------------------------------------------------------------------

ATTR_TOPOLOGY = "uac.topology"
ATTR_MAX_ITERATIONS = "uac.max_iterations"
ATTR_ITERATION = "uac.iteration"
ATTR_AGENT_ID = "uac.agent.id"
ATTR_MODEL = "uac.model"
ATTR_PROVIDER = "uac.provider"
ATTR_STRATEGY = "uac.strategy"
ATTR_TOKENS_PROMPT = "uac.tokens.prompt"
ATTR_TOKENS_COMPLETION = "uac.tokens.completion"
ATTR_TOKENS_TOTAL = "uac.tokens.total"
ATTR_FINISH_REASON = "uac.finish_reason"
ATTR_TOOL_NAME = "uac.tool.name"
ATTR_TOOL_PROVIDER = "uac.tool.provider"
ATTR_REFLEXION_ATTEMPT = "uac.reflexion.attempt"
ATTR_REFLEXION_MAX_RETRIES = "uac.reflexion.max_retries"

_INSTRUMENTATION_NAME = "uac"


def get_tracer(name: str | None = None) -> trace.Tracer:
    """Return a :class:`~opentelemetry.trace.Tracer` for *name*.

    If the OpenTelemetry SDK has not been configured the returned tracer
    is a no-op — all spans become no-ops with negligible overhead.
    """
    return trace.get_tracer(name or _INSTRUMENTATION_NAME)


def configure_telemetry(
    *,
    service_name: str = "uac",
    export_to_console: bool = True,
    otlp_endpoint: str | None = None,
) -> None:
    """Configure OpenTelemetry tracing (requires ``uac[otel]``).

    Parameters
    ----------
    service_name:
        The ``service.name`` resource attribute.
    export_to_console:
        If ``True``, export spans as JSON to stdout.
    otlp_endpoint:
        If set, export spans via OTLP/gRPC to this endpoint.

    Raises
    ------
    ImportError
        If the ``opentelemetry-sdk`` package is not installed.
    """
    # The SDK is an optional dependency — suppress type errors from unresolved imports.
    try:
        from opentelemetry.sdk.resources import Resource  # pyright: ignore[reportMissingImports,reportUnknownVariableType]
        from opentelemetry.sdk.trace import TracerProvider  # pyright: ignore[reportMissingImports,reportUnknownVariableType]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor  # pyright: ignore[reportMissingImports,reportUnknownVariableType]
    except ImportError as exc:
        msg = (
            "opentelemetry-sdk is required for configure_telemetry(). "
            "Install it with: pip install uac[otel]"
        )
        raise ImportError(msg) from exc

    resource = Resource.create({"service.name": service_name})  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
    provider = TracerProvider(resource=resource)  # pyright: ignore[reportUnknownVariableType]

    if export_to_console:
        _add_console_exporter(provider, SimpleSpanProcessor)

    if otlp_endpoint:
        _add_otlp_exporter(provider, BatchSpanProcessor, otlp_endpoint)

    trace.set_tracer_provider(provider)  # pyright: ignore[reportUnknownArgumentType]


def _add_console_exporter(provider: Any, processor_cls: Any) -> None:
    """Attach the console exporter (JSON to stdout)."""
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter  # pyright: ignore[reportMissingImports,reportUnknownVariableType]

    provider.add_span_processor(processor_cls(ConsoleSpanExporter()))


def _add_otlp_exporter(provider: Any, processor_cls: Any, endpoint: str) -> None:
    """Attach the OTLP gRPC exporter."""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # pyright: ignore[reportMissingImports,reportUnknownVariableType]
    except ImportError as exc:
        msg = (
            "opentelemetry-exporter-otlp is required for OTLP export. "
            "Install it with: pip install uac[otel]"
        )
        raise ImportError(msg) from exc

    provider.add_span_processor(processor_cls(OTLPSpanExporter(endpoint=endpoint)))
