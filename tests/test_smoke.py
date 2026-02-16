"""Smoke test to verify the project scaffolding works."""

from __future__ import annotations


def test_import() -> None:
    import uac

    assert uac.__version__ == "0.1.0"


def test_cli_entrypoint() -> None:
    from uac.cli import main

    assert callable(main)
