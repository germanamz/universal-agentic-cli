"""``uac agents`` — list and inspect agent manifests."""

from __future__ import annotations

import json
from pathlib import Path

import click

from uac.cli_commands._output import console, print_agents_table


@click.group()
def agents() -> None:
    """Manage agent manifests."""


@agents.command("list")
@click.option(
    "--dir",
    "directory",
    default="agents",
    type=click.Path(exists=False),
    help="Directory containing agent manifests.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format.",
)
def list_agents(directory: str, fmt: str) -> None:
    """List all agent manifests in a directory."""
    from uac.core.orchestration.manifest import ManifestLoader

    dir_path = Path(directory)
    if not dir_path.is_dir():
        console.print(f"[yellow]Directory not found: {directory}[/yellow]")
        return

    loader = ManifestLoader(dir_path)
    manifests = loader.load_all()

    if not manifests:
        console.print("[yellow]No agent manifests found.[/yellow]")
        return

    if fmt == "json":
        data = {name: m.model_dump() for name, m in manifests.items()}
        console.print_json(json.dumps(data, default=str))
    else:
        print_agents_table(manifests)
