"""A2AClient — discovers and invokes remote agents via the A2A protocol."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx

from uac.core.interface.models import ToolResult
from uac.protocols.a2a.models import (
    A2AMessage,
    A2APart,
    A2ATaskParams,
    A2ATaskRequest,
    A2ATaskResponse,
    AgentCard,
)
from uac.protocols.errors import ConnectionError, ToolExecutionError, ToolNotFoundError


class A2AClient:
    """Communicates with a remote A2A-compatible agent.

    Satisfies the :class:`~uac.protocols.provider.ToolProvider` protocol.

    Usage::

        async with A2AClient("https://agent.example.com") as client:
            tools = await client.discover_tools()
            result = await client.execute_tool("summarize", {"message": "..."})
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._card: AgentCard | None = None

    async def __aenter__(self) -> A2AClient:
        self._client = httpx.AsyncClient(base_url=self._base_url)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "A2AClient must be used as an async context manager"
            raise RuntimeError(msg)
        return self._client

    async def fetch_agent_card(self) -> AgentCard:
        """GET ``.well-known/agent.json`` and parse into an :class:`AgentCard`."""
        try:
            response = await self._http().get("/.well-known/agent.json")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ConnectionError(str(exc)) from exc
        self._card = AgentCard.model_validate(response.json())
        return self._card

    async def discover_tools(self) -> list[dict[str, Any]]:
        """Map each agent skill to an OpenAI-compatible function schema.

        Each skill becomes a function whose single parameter is ``message``
        (a string the caller wants the remote agent to act on).
        """
        if self._card is None:
            await self.fetch_agent_card()
        assert self._card is not None
        schemas: list[dict[str, Any]] = []
        for skill in self._card.skills:
            schemas.append({
                "type": "function",
                "function": {
                    "name": skill.id,
                    "description": skill.description or skill.name,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Input for the agent"},
                        },
                        "required": ["message"],
                    },
                },
            })
        return schemas

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Send a ``tasks/send`` JSON-RPC request to the remote agent."""
        # Validate the skill exists
        if self._card is not None:
            skill_ids = {s.id for s in self._card.skills}
            if name not in skill_ids:
                raise ToolNotFoundError(name)

        message_text = str(arguments.get("message", ""))
        request = A2ATaskRequest(
            params=A2ATaskParams(
                id=uuid4().hex[:12],
                message=A2AMessage(
                    role="user",
                    parts=[A2APart(type="text", text=message_text)],
                ),
            ),
        )

        try:
            response = await self._http().post("/", json=request.model_dump())
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolExecutionError(name, str(exc)) from exc

        task_response = A2ATaskResponse.model_validate(response.json())

        if task_response.error is not None:
            detail = task_response.error.get("message", str(task_response.error))
            raise ToolExecutionError(name, str(detail))

        # Extract text from the first artifact
        text = self._extract_text(task_response)
        return ToolResult.from_text(tool_call_id="", text=text)

    @staticmethod
    def _extract_text(response: A2ATaskResponse) -> str:
        """Extract text content from the task response."""
        if response.result is None:
            return ""
        # Try artifacts first
        for artifact in response.result.artifacts:
            for part in artifact.parts:
                if part.text:
                    return part.text
        # Fall back to status message
        if response.result.status.message:
            for part in response.result.status.message.parts:
                if part.text:
                    return part.text
        return ""
