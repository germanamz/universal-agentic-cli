"""Shared error types for the runtime safety layer."""


class RuntimeSafetyError(Exception):
    """Base error for all runtime safety failures."""


class SandboxError(RuntimeSafetyError):
    """A sandbox operation failed (creation, execution, or cleanup)."""

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        super().__init__(f"Sandbox error" + (f": {detail}" if detail else ""))


class SandboxTimeoutError(SandboxError):
    """Sandbox execution exceeded the configured timeout."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(f"Execution timed out after {timeout}s")


class ApprovalDeniedError(RuntimeSafetyError):
    """The gatekeeper denied execution of a tool call."""

    def __init__(self, tool_name: str, reason: str = "") -> None:
        self.tool_name = tool_name
        self.reason = reason
        msg = f"Approval denied for tool: {tool_name}"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg)


class ApprovalTimeoutError(RuntimeSafetyError):
    """The gatekeeper timed out waiting for user input."""

    def __init__(self, tool_name: str, timeout: float) -> None:
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(
            f"Approval timed out for tool: {tool_name} after {timeout}s"
        )
