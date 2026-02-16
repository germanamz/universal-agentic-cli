"""UAC CLI entrypoint."""

from __future__ import annotations

import click

from uac import __version__


@click.group()
@click.version_option(version=__version__, prog_name="uac")
def main() -> None:
    """UAC — Universal Agentic CLI."""


# Register subcommands
from uac.cli_commands import register_commands  # noqa: E402

register_commands(main)

if __name__ == "__main__":
    main()
