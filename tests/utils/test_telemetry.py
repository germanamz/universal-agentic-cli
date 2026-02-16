"""Tests for OpenTelemetry tracing helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace

from uac.utils.telemetry import (
    ATTR_TOPOLOGY,
    _INSTRUMENTATION_NAME,
    configure_telemetry,
    get_tracer,
)


class TestGetTracer:
    def test_returns_tracer(self) -> None:
        tracer = get_tracer("test.module")
        assert isinstance(tracer, trace.Tracer)

    def test_default_name(self) -> None:
        tracer = get_tracer()
        assert isinstance(tracer, trace.Tracer)

    def test_noop_span(self) -> None:
        """Without SDK configured, spans should be no-ops."""
        tracer = get_tracer("test.noop")
        with tracer.start_as_current_span("test") as span:
            span.set_attribute("key", "value")
            # No error — the NoopSpan silently accepts attributes


class TestConfigureTelemetry:
    def test_raises_without_sdk(self) -> None:
        """configure_telemetry requires opentelemetry-sdk."""
        with patch.dict("sys.modules", {"opentelemetry.sdk.resources": None}):
            with pytest.raises(ImportError, match="opentelemetry-sdk"):
                configure_telemetry()

    def test_configures_with_console(self) -> None:
        """When SDK is available, should set up TracerProvider with console exporter."""
        try:
            from opentelemetry.sdk.trace import TracerProvider
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        # Reset to default provider after test
        original = trace.get_tracer_provider()
        try:
            configure_telemetry(service_name="test-svc", export_to_console=True)
            provider = trace.get_tracer_provider()
            # The proxy wraps the real one; check it's not the default NoopTracerProvider
            assert provider is not original or isinstance(provider, TracerProvider)
        finally:
            trace.set_tracer_provider(original)

    def test_otlp_raises_without_exporter(self) -> None:
        """OTLP export requires opentelemetry-exporter-otlp."""
        try:
            import opentelemetry.sdk.trace  # noqa: F401
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        with patch.dict(
            "sys.modules",
            {"opentelemetry.exporter.otlp.proto.grpc.trace_exporter": None},
        ):
            with pytest.raises(ImportError, match="opentelemetry-exporter-otlp"):
                configure_telemetry(
                    export_to_console=False,
                    otlp_endpoint="http://localhost:4317",
                )


class TestAttributeConstants:
    def test_constants_are_strings(self) -> None:
        assert isinstance(ATTR_TOPOLOGY, str)
        assert ATTR_TOPOLOGY.startswith("uac.")

    def test_instrumentation_name(self) -> None:
        assert _INSTRUMENTATION_NAME == "uac"
