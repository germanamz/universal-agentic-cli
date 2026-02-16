"""Shared CLI output formatters."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table

from uac.core.blackboard.blackboard import Blackboard  # noqa: TC001
from uac.core.orchestration.models import AgentManifest  # noqa: TC001

console = Console()


def print_blackboard(board: Blackboard, *, as_json: bool = False) -> None:
    """Pretty-print a Blackboard summary."""
    if as_json:
        console.print_json(board.model_dump_json())
        return

    console.print("\n[bold]Blackboard Summary[/bold]")
    console.print(f"  Belief state: {board.belief_state or '(empty)'}")
    console.print(f"  Trace entries: {len(board.execution_trace)}")
    console.print(f"  Artifacts: {len(board.artifacts)} key(s)")
    console.print(f"  Pending tasks: {len(board.pending_tasks)}")

    if board.artifacts:
        console.print("\n[bold]Artifacts:[/bold]")
        for key, val in board.artifacts.items():
            console.print(f"  {key}: {_truncate(str(val))}")

    if board.execution_trace:
        console.print("\n[bold]Recent Trace:[/bold]")
        for entry in board.execution_trace[-10:]:
            console.print(f"  [{entry.agent_id}] {entry.action}")


def print_blackboard_section(
    board: Blackboard, section: str, *, as_json: bool = False
) -> None:
    """Print a specific section of the Blackboard."""
    data: Any
    if section == "belief":
        data = board.belief_state
    elif section == "trace":
        data = [e.model_dump() for e in board.execution_trace]
    elif section == "artifacts":
        data = board.artifacts
    elif section == "tasks":
        data = [t.model_dump() for t in board.pending_tasks]
    else:
        console.print(f"[red]Unknown section: {section}[/red]")
        return

    if as_json:
        console.print_json(json.dumps(data, default=str))
    else:
        console.print(data)


def print_tools_table(tools: list[dict[str, Any]]) -> None:
    """Pretty-print a list of tool schemas as a table."""
    table = Table(title="Discovered Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for tool in tools:
        func = tool.get("function", {})
        table.add_row(
            func.get("name", "?"),
            _truncate(func.get("description", "")),
        )

    console.print(table)


def print_agents_table(manifests: dict[str, AgentManifest]) -> None:
    """Pretty-print agent manifests as a table."""
    table = Table(title="Agent Manifests")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Description")
    table.add_column("MCP Servers")

    for manifest in manifests.values():
        servers = ", ".join(s.name for s in manifest.mcp_servers) or "-"
        table.add_row(
            manifest.name,
            manifest.version,
            _truncate(manifest.description),
            servers,
        )

    console.print(table)


def _truncate(text: str, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
