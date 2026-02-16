"""Tests for SandboxExecutor protocol conformance."""

from uac.runtime.sandbox.docker_sandbox import DockerSandbox
from uac.runtime.sandbox.executor import SandboxExecutor
from uac.runtime.sandbox.local_sandbox import LocalSandbox


class TestSandboxExecutorProtocol:
    def test_docker_sandbox_satisfies_protocol(self) -> None:
        assert isinstance(DockerSandbox(), SandboxExecutor)

    def test_local_sandbox_satisfies_protocol(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert isinstance(LocalSandbox(), SandboxExecutor)
