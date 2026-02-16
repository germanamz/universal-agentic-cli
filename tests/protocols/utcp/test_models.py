"""Tests for UTCP models."""

from uac.protocols.utcp.models import CLIToolDef, HTTPToolDef, UTCPParamMapping


class TestUTCPParamMapping:
    def test_defaults(self) -> None:
        param = UTCPParamMapping(name="q", location="query")
        assert param.type == "string"
        assert param.required is True
        assert param.default is None

    def test_custom_values(self) -> None:
        param = UTCPParamMapping(
            name="limit",
            location="query",
            type="integer",
            description="Max results",
            required=False,
            default=10,
        )
        assert param.name == "limit"
        assert param.type == "integer"
        assert param.default == 10

    def test_round_trip(self) -> None:
        param = UTCPParamMapping(name="id", location="path", description="Resource ID")
        data = param.model_dump()
        restored = UTCPParamMapping.model_validate(data)
        assert restored == param


class TestHTTPToolDef:
    def test_defaults(self) -> None:
        tool = HTTPToolDef(name="get_user", url_template="https://api.example.com/users/{id}")
        assert tool.kind == "http"
        assert tool.method == "GET"
        assert tool.headers == {}
        assert tool.params == []
        assert tool.body_template is None
        assert tool.response_path is None

    def test_full_definition(self) -> None:
        tool = HTTPToolDef(
            name="create_user",
            url_template="https://api.example.com/users",
            method="POST",
            headers={"Authorization": "Bearer token"},
            params=[UTCPParamMapping(name="name", location="body")],
            body_template={"role": "user"},
            response_path="data.id",
            description="Create a new user",
        )
        assert tool.method == "POST"
        assert tool.response_path == "data.id"
        assert len(tool.params) == 1

    def test_round_trip(self) -> None:
        tool = HTTPToolDef(
            name="test",
            url_template="https://example.com",
            description="Test tool",
        )
        data = tool.model_dump()
        restored = HTTPToolDef.model_validate(data)
        assert restored == tool


class TestCLIToolDef:
    def test_defaults(self) -> None:
        tool = CLIToolDef(name="list_files", command_template="ls {dir}")
        assert tool.kind == "cli"
        assert tool.timeout == 30.0
        assert tool.cwd is None
        assert tool.env == {}

    def test_custom_values(self) -> None:
        tool = CLIToolDef(
            name="grep_logs",
            command_template="grep {pattern} /var/log/app.log",
            params=[UTCPParamMapping(name="pattern", location="arg")],
            timeout=10.0,
            cwd="/var/log",
            env={"LANG": "C"},
            description="Search logs",
        )
        assert tool.timeout == 10.0
        assert tool.cwd == "/var/log"
        assert tool.env["LANG"] == "C"

    def test_round_trip(self) -> None:
        tool = CLIToolDef(name="echo", command_template="echo {msg}")
        data = tool.model_dump()
        restored = CLIToolDef.model_validate(data)
        assert restored == tool
