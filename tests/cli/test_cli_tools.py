"""Tests for ``uac tools`` CLI command."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from uac.cli import main


class TestToolsDiscover:
    def test_discover_tools(self) -> None:
        mock_schemas = [
            {
                "type": "function",
                "function": {"name": "read_file", "description": "Read a file"},
            }
        ]

        with patch("uac.protocols.mcp.client.MCPClient") as mock_client_cls:
            mock_instance = mock_client_cls.return_value
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.discover_tools = AsyncMock(return_value=mock_schemas)

            runner = CliRunner()
            result = runner.invoke(main, ["tools", "discover", "npx @mcp/fs"])

            assert result.exit_code == 0
            assert "read_file" in result.output

    def test_discover_no_tools(self) -> None:
        with patch("uac.protocols.mcp.client.MCPClient") as mock_client_cls:
            mock_instance = mock_client_cls.return_value
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.discover_tools = AsyncMock(return_value=[])

            runner = CliRunner()
            result = runner.invoke(main, ["tools", "discover", "npx @mcp/fs"])

            assert result.exit_code == 0
            assert "No tools discovered" in result.output

    def test_discover_error(self) -> None:
        with patch("uac.protocols.mcp.client.MCPClient") as mock_client_cls:
            mock_instance = mock_client_cls.return_value
            mock_instance.__aenter__ = AsyncMock(side_effect=RuntimeError("fail"))
            mock_instance.__aexit__ = AsyncMock(return_value=False)

            runner = CliRunner()
            result = runner.invoke(main, ["tools", "discover", "bad-server"])

            assert result.exit_code == 0
            assert "Discovery error" in result.output
