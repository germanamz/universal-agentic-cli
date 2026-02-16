"""UTCP models — HTTP and CLI tool definitions for legacy integrations.

UTCP (Universal Tool Calling Protocol) wraps arbitrary HTTP endpoints and
CLI commands behind the :class:`~uac.protocols.provider.ToolProvider` interface.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class UTCPParamMapping(BaseModel):
    """Maps a single parameter to its location in the request."""

    name: str
    location: Literal["path", "query", "header", "body", "arg"]
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


class HTTPToolDef(BaseModel):
    """Definition for an HTTP-based tool."""

    kind: Literal["http"] = "http"
    name: str
    url_template: str
    method: str = "GET"
    headers: dict[str, str] = {}
    params: list[UTCPParamMapping] = []
    body_template: dict[str, Any] | None = None
    response_path: str | None = None
    description: str = ""


class CLIToolDef(BaseModel):
    """Definition for a CLI command-based tool."""

    kind: Literal["cli"] = "cli"
    name: str
    command_template: str
    params: list[UTCPParamMapping] = []
    timeout: float = 30.0
    cwd: str | None = None
    env: dict[str, str] = {}
    description: str = ""


UTCPToolDef = HTTPToolDef | CLIToolDef
