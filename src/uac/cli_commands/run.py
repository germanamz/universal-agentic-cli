"""``uac run`` — execute a workflow from a YAML file."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from uac.cli_commands._output import console, print_blackboard


@click.command()
@click.argument("workflow", type=click.Path(exists=True))
@click.option("--goal", "-g", default=None, help="Override the goal for this run.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--telemetry", is_flag=True, help="Enable telemetry.")
@click.option("--dry-run", is_flag=True, help="Validate workflow only, do not execute.")
def run(
    workflow: str,
    goal: str | None,
    verbose: bool,
    telemetry: bool,
    dry_run: bool,
) -> None:
    """Execute a workflow defined in WORKFLOW yaml file."""
    from uac.sdk.workflow import WorkflowLoader, WorkflowRunner

    workflow_path = Path(workflow)
    loader = WorkflowLoader(workflow_path)

    try:
        spec = loader.load()
    except Exception as exc:
        console.print(f"[red]Validation error:[/red] {exc}")
        sys.exit(1)

    if telemetry:
        if spec.telemetry is None:
            from uac.sdk.models import TelemetrySettings

            spec.telemetry = TelemetrySettings(enabled=True)
        else:
            spec.telemetry.enabled = True

    if dry_run:
        console.print("[green]Workflow validated successfully.[/green]")
        console.print(f"  Name: {spec.name}")
        console.print(f"  Topology: {spec.topology.type}")
        console.print(f"  Agents: {', '.join(spec.agents)}")
        return

    effective_goal = goal or spec.name or "Execute workflow"

    if verbose:
        console.print(f"Running workflow: {spec.name}")
        console.print(f"Goal: {effective_goal}")

    runner = WorkflowRunner(spec, base_dir=workflow_path.parent)

    try:
        board = asyncio.run(runner.run(effective_goal))
    except Exception as exc:
        console.print(f"[red]Execution error:[/red] {exc}")
        sys.exit(1)

    print_blackboard(board)
