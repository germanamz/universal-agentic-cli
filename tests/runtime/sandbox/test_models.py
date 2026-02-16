"""Tests for sandbox data models."""

from uac.runtime.sandbox.models import ExecutionRequest, SandboxConfig, SandboxResult


class TestSandboxConfig:
    def test_defaults(self) -> None:
        cfg = SandboxConfig()
        assert cfg.timeout == 30.0
        assert cfg.memory_limit == "256m"
        assert cfg.cpu_limit == 1.0
        assert cfg.network_enabled is False
        assert cfg.image == "python:3.12-slim"
        assert cfg.read_only is True
        assert cfg.env == {}

    def test_custom_values(self) -> None:
        cfg = SandboxConfig(
            timeout=60.0,
            memory_limit="512m",
            network_enabled=True,
            image="node:20-slim",
        )
        assert cfg.timeout == 60.0
        assert cfg.memory_limit == "512m"
        assert cfg.network_enabled is True
        assert cfg.image == "node:20-slim"


class TestExecutionRequest:
    def test_minimal(self) -> None:
        req = ExecutionRequest(command=["echo", "hello"])
        assert req.command == ["echo", "hello"]
        assert req.stdin is None
        assert req.timeout is None
        assert req.env == {}

    def test_with_stdin(self) -> None:
        req = ExecutionRequest(command=["cat"], stdin="some input")
        assert req.stdin == "some input"


class TestSandboxResult:
    def test_success(self) -> None:
        result = SandboxResult(exit_code=0, stdout="hello\n")
        assert result.exit_code == 0
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.timed_out is False

    def test_timeout(self) -> None:
        result = SandboxResult(exit_code=137, timed_out=True)
        assert result.timed_out is True
