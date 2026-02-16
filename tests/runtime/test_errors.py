"""Tests for runtime error hierarchy."""

import pytest

from uac.runtime.errors import (
    ApprovalDeniedError,
    ApprovalTimeoutError,
    RuntimeSafetyError,
    SandboxError,
    SandboxTimeoutError,
)


class TestErrorHierarchy:
    def test_sandbox_error_is_runtime_safety_error(self) -> None:
        assert issubclass(SandboxError, RuntimeSafetyError)

    def test_sandbox_timeout_error_is_sandbox_error(self) -> None:
        assert issubclass(SandboxTimeoutError, SandboxError)

    def test_approval_denied_is_runtime_safety_error(self) -> None:
        assert issubclass(ApprovalDeniedError, RuntimeSafetyError)

    def test_approval_timeout_is_runtime_safety_error(self) -> None:
        assert issubclass(ApprovalTimeoutError, RuntimeSafetyError)


class TestSandboxError:
    def test_message_with_detail(self) -> None:
        err = SandboxError("container crashed")
        assert "container crashed" in str(err)
        assert err.detail == "container crashed"

    def test_message_without_detail(self) -> None:
        err = SandboxError()
        assert "Sandbox error" in str(err)


class TestSandboxTimeoutError:
    def test_attributes(self) -> None:
        err = SandboxTimeoutError(30.0)
        assert err.timeout == 30.0
        assert "30.0s" in str(err)


class TestApprovalDeniedError:
    def test_with_reason(self) -> None:
        err = ApprovalDeniedError("rm_file", reason="too dangerous")
        assert err.tool_name == "rm_file"
        assert err.reason == "too dangerous"
        assert "rm_file" in str(err)
        assert "too dangerous" in str(err)

    def test_without_reason(self) -> None:
        err = ApprovalDeniedError("rm_file")
        assert "rm_file" in str(err)


class TestApprovalTimeoutError:
    def test_attributes(self) -> None:
        err = ApprovalTimeoutError("deploy", 60.0)
        assert err.tool_name == "deploy"
        assert err.timeout == 60.0
        assert "deploy" in str(err)
        assert "60.0s" in str(err)
