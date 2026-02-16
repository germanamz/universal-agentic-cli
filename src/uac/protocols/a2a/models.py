"""A2A models — Agent-to-Agent protocol data structures.

Based on the A2A specification: agents advertise capabilities via
``AgentCard`` at ``.well-known/agent.json``, and communicate via
JSON-RPC task messages.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Agent Discovery
# ---------------------------------------------------------------------------


class AgentSkill(BaseModel):
    """A single skill advertised by a remote agent."""

    id: str
    name: str
    description: str = ""
    tags: list[str] = []


class AgentCard(BaseModel):
    """Agent metadata served at ``.well-known/agent.json``."""

    name: str
    description: str = ""
    url: str
    skills: list[AgentSkill] = []


# ---------------------------------------------------------------------------
# Task Messages (JSON-RPC envelope)
# ---------------------------------------------------------------------------


class A2APart(BaseModel):
    """A content part within an A2A message."""

    type: str = "text"
    text: str = ""


class A2AMessage(BaseModel):
    """A message exchanged between agents."""

    role: Literal["user", "agent"] = "user"
    parts: list[A2APart] = []


class A2ATaskParams(BaseModel):
    """Parameters for the ``tasks/send`` JSON-RPC method."""

    id: str = ""
    message: A2AMessage = Field(default_factory=A2AMessage)


class A2ATaskRequest(BaseModel):
    """JSON-RPC request for ``tasks/send``."""

    jsonrpc: str = "2.0"
    method: str = "tasks/send"
    id: str | int = 1
    params: A2ATaskParams = Field(default_factory=A2ATaskParams)


# ---------------------------------------------------------------------------
# Task Response
# ---------------------------------------------------------------------------


class A2AArtifact(BaseModel):
    """An output artifact produced by a task."""

    parts: list[A2APart] = []
    name: str = ""


class A2ATaskStatus(BaseModel):
    """Status of a completed task."""

    state: str = "completed"
    message: A2AMessage | None = None


class A2ATaskResult(BaseModel):
    """The result payload inside a task response."""

    id: str = ""
    status: A2ATaskStatus = Field(default_factory=A2ATaskStatus)
    artifacts: list[A2AArtifact] = []


class A2ATaskResponse(BaseModel):
    """JSON-RPC response for ``tasks/send``."""

    jsonrpc: str = "2.0"
    id: str | int = 1
    result: A2ATaskResult | None = None
    error: dict[str, Any] | None = None
