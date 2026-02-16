"""A2A protocol — Agent-to-Agent discovery and delegation."""

from uac.protocols.a2a.client import A2AClient
from uac.protocols.a2a.mesh_adapter import A2AAgentNode
from uac.protocols.a2a.models import (
    A2AArtifact,
    A2AMessage,
    A2APart,
    A2ATaskParams,
    A2ATaskRequest,
    A2ATaskResponse,
    A2ATaskResult,
    A2ATaskStatus,
    AgentCard,
    AgentSkill,
)

__all__ = [
    "A2AAgentNode",
    "A2AArtifact",
    "A2AClient",
    "A2AMessage",
    "A2APart",
    "A2ATaskParams",
    "A2ATaskRequest",
    "A2ATaskResponse",
    "A2ATaskResult",
    "A2ATaskStatus",
    "AgentCard",
    "AgentSkill",
]
