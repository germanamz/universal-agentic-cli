"""``uac inspect`` — inspect a blackboard snapshot file."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from uac.cli_commands._output import console, print_blackboard, print_blackboard_section
from uac.core.blackboard.blackboard import Blackboard


@click.command("inspect")
@click.argument("snapshot_file", type=click.Path(exists=True))
@click.option(
    "--section",
    type=click.Choice(["belief", "trace", "artifacts", "tasks"]),
    default=None,
    help="Show only a specific section.",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def inspect_cmd(snapshot_file: str, section: str | None, as_json: bool) -> None:
    """Inspect a blackboard snapshot file.

    SNAPSHOT_FILE is a JSON file produced by Blackboard.snapshot().
    """
    path = Path(snapshot_file)
    try:
        data = path.read_bytes()
        board = Blackboard.restore(data)
    except Exception as exc:
        console.print(f"[red]Error loading snapshot:[/red] {exc}")
        sys.exit(1)

    if section:
        print_blackboard_section(board, section, as_json=as_json)
    else:
        print_blackboard(board, as_json=as_json)
