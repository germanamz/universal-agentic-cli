"""Data models for the gatekeeper subsystem."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PolicyAction(str, Enum):
    """Action a policy rule prescribes for a tool."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class ToolPolicy(BaseModel):
    """A single policy rule matching tool names to an action."""

    pattern: str = Field(..., description="Tool name or glob pattern (e.g. 'file_*', '*').")
    action: PolicyAction = Field(..., description="What to do when this rule matches.")
    reason: str = Field(default="", description="Human-readable rationale for the rule.")


class GatekeeperConfig(BaseModel):
    """Configuration for the gatekeeper."""

    enabled: bool = Field(default=True, description="Master switch for gatekeeper checks.")
    default_action: PolicyAction = Field(
        default=PolicyAction.ASK,
        description="Action when no policy rule matches.",
    )
    safe_tools: list[str] = Field(
        default_factory=list,
        description="Tool names that are always allowed (fast-path bypass).",
    )
    policies: list[ToolPolicy] = Field(
        default_factory=list,
        description="Ordered policy rules (first match wins).",
    )
    approval_timeout: float = Field(
        default=300.0,
        description="Seconds to wait for user approval before auto-denying.",
    )


class ApprovalRequest(BaseModel):
    """A request for the gatekeeper to approve or deny a tool execution."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="", description="Why approval is being requested.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalResult(BaseModel):
    """The gatekeeper's decision on an approval request."""

    approved: bool
    reason: str = Field(default="")
