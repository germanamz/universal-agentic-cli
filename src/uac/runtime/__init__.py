"""Runtime safety layer — sandbox isolation and gatekeeper approval."""

from uac.runtime.dispatcher import SafeDispatcher
from uac.runtime.errors import (
    ApprovalDeniedError,
    ApprovalTimeoutError,
    RuntimeSafetyError,
    SandboxError,
    SandboxTimeoutError,
)

__all__ = [
    "ApprovalDeniedError",
    "ApprovalTimeoutError",
    "RuntimeSafetyError",
    "SafeDispatcher",
    "SandboxError",
    "SandboxTimeoutError",
]
