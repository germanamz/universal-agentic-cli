"""Smoke test to verify the project scaffolding works."""

from __future__ import annotations


def test_import() -> None:
    import uac

    assert uac.__version__ == "0.1.0"


def test_cli_entrypoint() -> None:
    from uac.cli import main

    assert callable(main)


def test_sdk_imports() -> None:
    from uac.sdk import (
        AgentRef,
        GatekeeperSettings,
        TelemetrySettings,
        TopologyConfig,
        WorkflowLoader,
        WorkflowRunner,
        WorkflowSpec,
        WorkflowValidationError,
    )

    assert WorkflowRunner is not None
    assert WorkflowLoader is not None
    assert WorkflowSpec is not None
    assert AgentRef is not None
    assert TopologyConfig is not None
    assert GatekeeperSettings is not None
    assert TelemetrySettings is not None
    assert WorkflowValidationError is not None


def test_lazy_import_from_uac() -> None:
    import uac

    assert uac.WorkflowRunner is not None
    assert uac.WorkflowLoader is not None
