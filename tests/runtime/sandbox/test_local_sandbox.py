"""Tests for LocalSandbox."""

import warnings

import pytest

from uac.runtime.errors import SandboxError, SandboxTimeoutError
from uac.runtime.sandbox.local_sandbox import LocalSandbox, _WARNING_MSG
from uac.runtime.sandbox.models import ExecutionRequest, SandboxConfig


class TestLocalSandbox:
    def _make_sandbox(self, **kwargs) -> LocalSandbox:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LocalSandbox(SandboxConfig(**kwargs) if kwargs else None)

    def test_constructor_emits_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LocalSandbox()
            assert len(w) == 1
            assert _WARNING_MSG in str(w[0].message)

    async def test_execute_echo(self) -> None:
        sandbox = self._make_sandbox()
        req = ExecutionRequest(command=["echo", "hello"])
        result = await sandbox.execute(req)
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"

    async def test_execute_nonzero_exit(self) -> None:
        sandbox = self._make_sandbox()
        req = ExecutionRequest(command=["sh", "-c", "exit 42"])
        result = await sandbox.execute(req)
        assert result.exit_code == 42

    async def test_execute_with_stdin(self) -> None:
        sandbox = self._make_sandbox()
        req = ExecutionRequest(command=["cat"], stdin="piped input")
        result = await sandbox.execute(req)
        assert result.exit_code == 0
        assert result.stdout.strip() == "piped input"

    async def test_execute_timeout(self) -> None:
        sandbox = self._make_sandbox(timeout=0.1)
        req = ExecutionRequest(command=["sleep", "10"])
        with pytest.raises(SandboxTimeoutError) as exc_info:
            await sandbox.execute(req)
        assert exc_info.value.timeout == 0.1

    async def test_execute_per_request_timeout_override(self) -> None:
        sandbox = self._make_sandbox(timeout=60.0)
        req = ExecutionRequest(command=["sleep", "10"], timeout=0.1)
        with pytest.raises(SandboxTimeoutError):
            await sandbox.execute(req)

    async def test_execute_bad_command(self) -> None:
        sandbox = self._make_sandbox()
        req = ExecutionRequest(command=["nonexistent_command_xyz"])
        with pytest.raises(SandboxError):
            await sandbox.execute(req)

    async def test_cleanup_is_noop(self) -> None:
        sandbox = self._make_sandbox()
        await sandbox.cleanup()  # Should not raise

    async def test_env_vars_merged(self) -> None:
        sandbox = self._make_sandbox()
        req = ExecutionRequest(
            command=["sh", "-c", "echo $UAC_TEST_VAR"],
            env={"UAC_TEST_VAR": "hello_from_test"},
        )
        result = await sandbox.execute(req)
        assert result.stdout.strip() == "hello_from_test"
