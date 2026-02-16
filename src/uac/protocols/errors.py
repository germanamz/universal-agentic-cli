"""Shared error types for the protocol layer."""


class ProtocolError(Exception):
    """Base error for all protocol-layer failures."""


class ConnectionError(ProtocolError):
    """Failed to connect to an external service."""


class ToolNotFoundError(ProtocolError):
    """Requested tool does not exist in the provider's registry."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Tool not found: {name}")


class ToolExecutionError(ProtocolError):
    """A tool invocation failed at the provider side."""

    def __init__(self, name: str, detail: str = "") -> None:
        self.name = name
        self.detail = detail
        super().__init__(f"Tool execution failed: {name}" + (f" — {detail}" if detail else ""))
