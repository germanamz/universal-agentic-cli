"""CLI subcommand registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import click


def register_commands(cli: click.Group) -> None:
    """Register all subcommands on the CLI group."""
    from uac.cli_commands.agents import agents
    from uac.cli_commands.inspect import inspect_cmd
    from uac.cli_commands.run import run
    from uac.cli_commands.tools import tools

    cli.add_command(run)
    cli.add_command(agents)
    cli.add_command(tools)
    cli.add_command(inspect_cmd)
