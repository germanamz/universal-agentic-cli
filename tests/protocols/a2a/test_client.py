"""Tests for A2AClient with mocked httpx."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uac.protocols.a2a.client import A2AClient
from uac.protocols.errors import ConnectionError, ToolExecutionError, ToolNotFoundError


def _agent_card_json() -> dict:
    return {
        "name": "test-agent",
        "description": "A test agent",
        "url": "https://agent.example.com",
        "skills": [
            {"id": "summarize", "name": "Summarize", "description": "Summarize text"},
            {"id": "translate", "name": "Translate", "description": "Translate text"},
        ],
    }


def _task_response_json(text: str = "Done") -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "id": "task-1",
            "status": {"state": "completed"},
            "artifacts": [{"parts": [{"type": "text", "text": text}]}],
        },
    }


def _mock_httpx_client(
    card_json: dict | None = None,
    task_json: dict | None = None,
) -> MagicMock:
    client = AsyncMock()

    async def mock_get(url: str) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = card_json or _agent_card_json()
        resp.raise_for_status = MagicMock()
        return resp

    async def mock_post(url: str, json: dict | None = None) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = task_json or _task_response_json()
        resp.raise_for_status = MagicMock()
        return resp

    client.get = AsyncMock(side_effect=mock_get)
    client.post = AsyncMock(side_effect=mock_post)
    client.aclose = AsyncMock()
    return client


class TestA2AClientDiscovery:
    async def test_fetch_agent_card(self) -> None:
        mock = _mock_httpx_client()
        with patch("uac.protocols.a2a.client.httpx.AsyncClient", return_value=mock):
            async with A2AClient("https://agent.example.com") as client:
                card = await client.fetch_agent_card()
                assert card.name == "test-agent"
                assert len(card.skills) == 2

    async def test_discover_tools_returns_schemas(self) -> None:
        mock = _mock_httpx_client()
        with patch("uac.protocols.a2a.client.httpx.AsyncClient", return_value=mock):
            async with A2AClient("https://agent.example.com") as client:
                tools = await client.discover_tools()
                assert len(tools) == 2
                names = {t["function"]["name"] for t in tools}
                assert names == {"summarize", "translate"}
                # Each tool should have a "message" parameter
                for tool in tools:
                    assert "message" in tool["function"]["parameters"]["properties"]

    async def test_discover_fetches_card_if_needed(self) -> None:
        mock = _mock_httpx_client()
        with patch("uac.protocols.a2a.client.httpx.AsyncClient", return_value=mock):
            async with A2AClient("https://agent.example.com") as client:
                await client.discover_tools()
                mock.get.assert_awaited_once()

    async def test_connection_error(self) -> None:
        import httpx

        mock = AsyncMock()
        mock.get = AsyncMock(side_effect=httpx.HTTPError("unreachable"))
        mock.aclose = AsyncMock()

        with patch("uac.protocols.a2a.client.httpx.AsyncClient", return_value=mock):
            async with A2AClient("https://bad.example.com") as client:
                with pytest.raises(ConnectionError):
                    await client.fetch_agent_card()


class TestA2AClientExecution:
    async def test_execute_tool_success(self) -> None:
        mock = _mock_httpx_client(task_json=_task_response_json("Summary result"))
        with patch("uac.protocols.a2a.client.httpx.AsyncClient", return_value=mock):
            async with A2AClient("https://agent.example.com") as client:
                await client.discover_tools()
                result = await client.execute_tool("summarize", {"message": "Long text"})
                assert result.content[0].text == "Summary result"  # type: ignore[union-attr]

    async def test_execute_unknown_skill_raises(self) -> None:
        mock = _mock_httpx_client()
        with patch("uac.protocols.a2a.client.httpx.AsyncClient", return_value=mock):
            async with A2AClient("https://agent.example.com") as client:
                await client.discover_tools()
                with pytest.raises(ToolNotFoundError, match="nonexistent"):
                    await client.execute_tool("nonexistent", {"message": "x"})

    async def test_execute_error_response_raises(self) -> None:
        error_json = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": None,
            "error": {"code": -32600, "message": "Invalid request"},
        }
        mock = _mock_httpx_client(task_json=error_json)
        with patch("uac.protocols.a2a.client.httpx.AsyncClient", return_value=mock):
            async with A2AClient("https://agent.example.com") as client:
                await client.discover_tools()
                with pytest.raises(ToolExecutionError, match="summarize"):
                    await client.execute_tool("summarize", {"message": "x"})

    async def test_execute_http_error_raises(self) -> None:
        import httpx as httpx_lib

        mock = _mock_httpx_client()
        mock.post = AsyncMock(side_effect=httpx_lib.HTTPError("server error"))

        with patch("uac.protocols.a2a.client.httpx.AsyncClient", return_value=mock):
            async with A2AClient("https://agent.example.com") as client:
                await client.discover_tools()
                with pytest.raises(ToolExecutionError):
                    await client.execute_tool("summarize", {"message": "x"})
