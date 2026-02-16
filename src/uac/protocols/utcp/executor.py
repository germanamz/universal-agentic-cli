"""UTCPExecutor — satisfies ToolProvider for HTTP and CLI tools."""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import Any

import httpx

from uac.core.interface.models import ToolResult
from uac.protocols.errors import ToolExecutionError, ToolNotFoundError
from uac.protocols.utcp.models import CLIToolDef, HTTPToolDef, UTCPToolDef


class UTCPExecutor:
    """Executes HTTP and CLI tools defined via UTCP manifests.

    Satisfies the :class:`~uac.protocols.provider.ToolProvider` protocol.
    """

    def __init__(self, tools: list[UTCPToolDef]) -> None:
        self._tools: dict[str, UTCPToolDef] = {t.name: t for t in tools}

    async def discover_tools(self) -> list[dict[str, Any]]:
        """Convert UTCP tool definitions to OpenAI function schemas."""
        schemas: list[dict[str, Any]] = []
        for tool in self._tools.values():
            properties: dict[str, Any] = {}
            required: list[str] = []
            for param in tool.params:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    required.append(param.name)
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return schemas

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name, dispatching to HTTP or CLI handler."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        if isinstance(tool, HTTPToolDef):
            return await self._execute_http(tool, arguments)
        return await self._execute_cli(tool, arguments)

    async def _execute_http(self, tool: HTTPToolDef, arguments: dict[str, Any]) -> ToolResult:
        """Execute an HTTP tool by substituting templates and making a request."""
        # Substitute URL template placeholders
        url = self._substitute_template(tool.url_template, arguments, tool)
        headers = dict(tool.headers)
        query_params: dict[str, str] = {}
        body: dict[str, Any] | None = None

        for param in tool.params:
            value = arguments.get(param.name, param.default)
            if value is None:
                continue
            if param.location == "query":
                query_params[param.name] = str(value)
            elif param.location == "header":
                headers[param.name] = str(value)
            elif param.location == "body":
                if body is None:
                    body = {}
                body[param.name] = value

        # Merge explicit body template
        if tool.body_template is not None:
            merged = dict(tool.body_template)
            if body:
                merged.update(body)
            body = merged

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=tool.method,
                    url=url,
                    headers=headers,
                    params=query_params,
                    json=body if body else None,
                )
                response.raise_for_status()
                text = response.text
        except httpx.HTTPError as exc:
            raise ToolExecutionError(tool.name, str(exc)) from exc

        # Extract nested response path if specified
        if tool.response_path:
            try:
                data = json.loads(text)
                for key in tool.response_path.split("."):
                    data = data[key]
                text = json.dumps(data) if not isinstance(data, str) else data
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # Return raw text if path extraction fails

        return ToolResult.from_text(tool_call_id="", text=text)

    async def _execute_cli(self, tool: CLIToolDef, arguments: dict[str, Any]) -> ToolResult:
        """Execute a CLI tool by substituting the command template."""
        cmd_str = self._substitute_command(tool.command_template, arguments, tool)
        parts = shlex.split(cmd_str)

        env = dict(tool.env) if tool.env else None

        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tool.cwd,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=tool.timeout)
        except TimeoutError as exc:
            raise ToolExecutionError(tool.name, "Command timed out") from exc
        except OSError as exc:
            raise ToolExecutionError(tool.name, str(exc)) from exc

        if proc.returncode != 0:
            detail = stderr.decode(errors="replace").strip() if stderr else "non-zero exit"
            raise ToolExecutionError(tool.name, detail)

        text = stdout.decode(errors="replace").strip() if stdout else ""
        return ToolResult.from_text(tool_call_id="", text=text)

    @staticmethod
    def _substitute_template(
        template: str, arguments: dict[str, Any], tool: HTTPToolDef
    ) -> str:
        """Replace ``{param}`` placeholders in a URL template with argument values."""
        result = template
        for param in tool.params:
            if param.location == "path":
                value = arguments.get(param.name, param.default)
                if value is not None:
                    result = result.replace(f"{{{param.name}}}", str(value))
        return result

    @staticmethod
    def _substitute_command(
        template: str, arguments: dict[str, Any], tool: CLIToolDef
    ) -> str:
        """Replace ``{param}`` placeholders in a command template with shell-quoted values."""
        result = template
        for param in tool.params:
            if param.location == "arg":
                value = arguments.get(param.name, param.default)
                if value is not None:
                    placeholder = f"{{{param.name}}}"
                    result = result.replace(placeholder, shlex.quote(str(value)))
        # Also do a generic pass for any remaining {key} patterns in arguments
        for key, value in arguments.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, shlex.quote(str(value)))
        return result
