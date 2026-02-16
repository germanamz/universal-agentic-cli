"""Tests for UTCPExecutor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uac.protocols.errors import ToolExecutionError, ToolNotFoundError
from uac.protocols.utcp.executor import UTCPExecutor
from uac.protocols.utcp.models import CLIToolDef, HTTPToolDef, UTCPParamMapping


def _http_tool(
    name: str = "get_weather",
    url: str = "https://api.example.com/weather/{city}",
    method: str = "GET",
) -> HTTPToolDef:
    return HTTPToolDef(
        name=name,
        url_template=url,
        method=method,
        params=[
            UTCPParamMapping(name="city", location="path", description="City name"),
            UTCPParamMapping(
                name="units", location="query", required=False, default="metric"
            ),
        ],
        description="Get weather for a city",
    )


def _cli_tool(name: str = "echo_msg", cmd: str = "echo {msg}") -> CLIToolDef:
    return CLIToolDef(
        name=name,
        command_template=cmd,
        params=[UTCPParamMapping(name="msg", location="arg", description="Message")],
        description="Echo a message",
    )


class TestUTCPExecutorDiscovery:
    async def test_discover_http_tool(self) -> None:
        executor = UTCPExecutor([_http_tool()])
        schemas = await executor.discover_tools()
        assert len(schemas) == 1
        func = schemas[0]["function"]
        assert func["name"] == "get_weather"
        assert "city" in func["parameters"]["properties"]

    async def test_discover_cli_tool(self) -> None:
        executor = UTCPExecutor([_cli_tool()])
        schemas = await executor.discover_tools()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "echo_msg"

    async def test_discover_multiple_tools(self) -> None:
        executor = UTCPExecutor([_http_tool(), _cli_tool()])
        schemas = await executor.discover_tools()
        assert len(schemas) == 2

    async def test_required_params_in_schema(self) -> None:
        executor = UTCPExecutor([_http_tool()])
        schemas = await executor.discover_tools()
        required = schemas[0]["function"]["parameters"]["required"]
        assert "city" in required
        assert "units" not in required


class TestUTCPExecutorHTTP:
    async def test_execute_http_success(self) -> None:
        mock_response = MagicMock()
        mock_response.text = '{"temp": 22}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        executor = UTCPExecutor([_http_tool()])
        with patch("uac.protocols.utcp.executor.httpx.AsyncClient", return_value=mock_client):
            result = await executor.execute_tool("get_weather", {"city": "London"})

        assert result.content[0].text == '{"temp": 22}'  # type: ignore[union-attr]
        mock_client.request.assert_awaited_once()
        call_kwargs = mock_client.request.call_args
        assert "London" in call_kwargs.kwargs["url"]

    async def test_execute_http_with_response_path(self) -> None:
        tool = HTTPToolDef(
            name="extract",
            url_template="https://api.example.com/data",
            response_path="data.value",
        )
        mock_response = MagicMock()
        mock_response.text = '{"data": {"value": "extracted"}}'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        executor = UTCPExecutor([tool])
        with patch("uac.protocols.utcp.executor.httpx.AsyncClient", return_value=mock_client):
            result = await executor.execute_tool("extract", {})

        assert result.content[0].text == "extracted"  # type: ignore[union-attr]

    async def test_execute_unknown_tool_raises(self) -> None:
        executor = UTCPExecutor([_http_tool()])
        with pytest.raises(ToolNotFoundError, match="nonexistent"):
            await executor.execute_tool("nonexistent", {})

    async def test_execute_http_error_raises(self) -> None:
        import httpx

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.HTTPError("connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        executor = UTCPExecutor([_http_tool()])
        with (
            patch("uac.protocols.utcp.executor.httpx.AsyncClient", return_value=mock_client),
            pytest.raises(ToolExecutionError, match="get_weather"),
        ):
            await executor.execute_tool("get_weather", {"city": "London"})


class TestUTCPExecutorCLI:
    async def test_execute_cli_success(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"hello world\n", b""))
        mock_proc.returncode = 0

        executor = UTCPExecutor([_cli_tool()])
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await executor.execute_tool("echo_msg", {"msg": "hello world"})

        assert result.content[0].text == "hello world"  # type: ignore[union-attr]

    async def test_execute_cli_nonzero_exit_raises(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error output"))
        mock_proc.returncode = 1

        executor = UTCPExecutor([_cli_tool()])
        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            pytest.raises(ToolExecutionError, match="echo_msg"),
        ):
            await executor.execute_tool("echo_msg", {"msg": "fail"})

    async def test_execute_cli_timeout_raises(self) -> None:

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())

        executor = UTCPExecutor([CLIToolDef(name="slow", command_template="sleep 100")])
        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("asyncio.wait_for", side_effect=TimeoutError()),
            pytest.raises(ToolExecutionError, match="timed out"),
        ):
            await executor.execute_tool("slow", {})

    async def test_cli_shell_quoting(self) -> None:
        """Verify arguments are shell-quoted to prevent injection."""
        tool = _cli_tool(name="echo", cmd="echo {msg}")
        result = UTCPExecutor._substitute_command(
            tool.command_template, {"msg": "hello; rm -rf /"}, tool
        )
        # shlex.quote wraps dangerous input in single quotes
        assert "rm -rf" not in result or "'" in result
