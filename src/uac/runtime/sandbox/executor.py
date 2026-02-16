"""SandboxExecutor protocol — the common interface for sandbox implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uac.runtime.sandbox.models import ExecutionRequest, SandboxResult


@runtime_checkable
class SandboxExecutor(Protocol):
    """Executes commands in an isolated environment.

    Implementations must provide ``execute()`` for running commands and
    ``cleanup()`` for releasing resources (e.g. removing containers).
    """

    async def execute(self, request: ExecutionRequest) -> SandboxResult:
        """Run a command in the sandbox and return the result."""
        ...

    async def cleanup(self) -> None:
        """Release any resources held by this executor."""
        ...
