"""``uac tools`` — discover and inspect tools from MCP servers."""

from __future__ import annotations

import asyncio

import click

from uac.cli_commands._output import console, print_tools_table


@click.group()
def tools() -> None:
    """Discover and inspect tools."""


@tools.command("discover")
@click.argument("server")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "websocket"]),
    default="stdio",
    help="MCP server transport type.",
)
def discover(server: str, transport: str) -> None:
    """Discover tools from an MCP server.

    SERVER is the command (for stdio) or URL (for websocket) of the MCP server.
    """
    from uac.core.orchestration.models import MCPServerRef
    from uac.protocols.mcp.client import MCPClient

    if transport == "stdio":
        ref = MCPServerRef(name="cli-discover", transport="stdio", command=server)
    else:
        ref = MCPServerRef(name="cli-discover", transport="websocket", url=server)

    async def _discover() -> list[dict[str, object]]:
        async with MCPClient(ref) as client:
            return await client.discover_tools()

    try:
        tool_schemas = asyncio.run(_discover())
    except Exception as exc:
        console.print(f"[red]Discovery error:[/red] {exc}")
        return

    if not tool_schemas:
        console.print("[yellow]No tools discovered.[/yellow]")
        return

    print_tools_table(tool_schemas)
