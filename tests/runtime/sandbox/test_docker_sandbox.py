"""Tests for DockerSandbox (docker CLI mocked)."""

from unittest.mock import AsyncMock, patch

import pytest

from uac.runtime.errors import SandboxError, SandboxTimeoutError
from uac.runtime.sandbox.docker_sandbox import DockerSandbox, _DockerOutput
from uac.runtime.sandbox.models import ExecutionRequest, SandboxConfig


def _mock_docker_output(stdout: str = "", stderr: str = "") -> _DockerOutput:
    return _DockerOutput(stdout=stdout, stderr=stderr)


class TestDockerSandbox:
    def _make_sandbox(self, **kwargs) -> DockerSandbox:
        return DockerSandbox(SandboxConfig(**kwargs) if kwargs else None)

    async def test_execute_success(self) -> None:
        sandbox = self._make_sandbox()

        with patch.object(
            DockerSandbox,
            "_run_docker",
            new_callable=AsyncMock,
        ) as mock_run:
            # create, start, wait, logs, rm
            mock_run.side_effect = [
                _mock_docker_output(),                       # docker create
                _mock_docker_output(),                       # docker start
                _mock_docker_output(stdout="0"),             # docker wait
                _mock_docker_output(stdout="hello world"),   # docker logs
                _mock_docker_output(),                       # docker rm
            ]

            req = ExecutionRequest(command=["echo", "hello world"])
            result = await sandbox.execute(req)

            assert result.exit_code == 0
            assert result.stdout == "hello world"
            assert mock_run.call_count == 5

    async def test_execute_timeout_kills_container(self) -> None:
        sandbox = self._make_sandbox(timeout=0.1)

        call_count = 0

        async def mock_run(cmd, *, ignore_errors=False, capture_stderr=False):
            nonlocal call_count
            call_count += 1
            if cmd[1] == "wait":
                # Simulate a long wait that will be cancelled
                import asyncio
                await asyncio.sleep(10)
            return _mock_docker_output()

        with patch.object(DockerSandbox, "_run_docker", side_effect=mock_run):
            req = ExecutionRequest(command=["sleep", "100"])
            with pytest.raises(SandboxTimeoutError):
                await sandbox.execute(req)

    async def test_build_create_command_defaults(self) -> None:
        sandbox = self._make_sandbox()
        req = ExecutionRequest(command=["echo", "hi"])
        cmd = sandbox._build_create_command("test-container", req)

        assert "docker" in cmd
        assert "create" in cmd
        assert "--name" in cmd
        assert "test-container" in cmd
        assert "--network" in cmd
        assert "none" in cmd
        assert "--read-only" in cmd
        assert "--memory" in cmd
        assert "256m" in cmd
        assert "echo" in cmd
        assert "hi" in cmd

    async def test_build_create_command_network_enabled(self) -> None:
        sandbox = self._make_sandbox(network_enabled=True)
        req = ExecutionRequest(command=["curl", "example.com"])
        cmd = sandbox._build_create_command("test-container", req)

        assert "--network" not in cmd

    async def test_build_create_command_env_vars(self) -> None:
        sandbox = self._make_sandbox()
        req = ExecutionRequest(command=["env"], env={"FOO": "bar"})
        cmd = sandbox._build_create_command("test-container", req)

        # Find the -e flag and check the value follows it
        env_indices = [i for i, v in enumerate(cmd) if v == "-e"]
        env_values = [cmd[i + 1] for i in env_indices]
        assert "FOO=bar" in env_values

    async def test_cleanup_removes_tracked_containers(self) -> None:
        sandbox = self._make_sandbox()
        sandbox._active_containers = {"container-a", "container-b"}

        with patch.object(
            DockerSandbox,
            "_run_docker",
            new_callable=AsyncMock,
            return_value=_mock_docker_output(),
        ) as mock_run:
            await sandbox.cleanup()

            assert len(sandbox._active_containers) == 0
            assert mock_run.call_count == 2

    async def test_container_removed_on_error(self) -> None:
        sandbox = self._make_sandbox()

        call_count = 0

        async def mock_run(cmd, *, ignore_errors=False, capture_stderr=False):
            nonlocal call_count
            call_count += 1
            if cmd[1] == "start":
                raise SandboxError("start failed")
            return _mock_docker_output()

        with patch.object(DockerSandbox, "_run_docker", side_effect=mock_run):
            req = ExecutionRequest(command=["echo"])
            with pytest.raises(SandboxError, match="start failed"):
                await sandbox.execute(req)

        # Verify container was cleaned up even on error
        assert len(sandbox._active_containers) == 0

    async def test_nonzero_exit_code(self) -> None:
        sandbox = self._make_sandbox()

        with patch.object(
            DockerSandbox,
            "_run_docker",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.side_effect = [
                _mock_docker_output(),                    # docker create
                _mock_docker_output(),                    # docker start
                _mock_docker_output(stdout="1"),          # docker wait (exit 1)
                _mock_docker_output(stderr="error msg"),  # docker logs
                _mock_docker_output(),                    # docker rm
            ]

            req = ExecutionRequest(command=["false"])
            result = await sandbox.execute(req)

            assert result.exit_code == 1
            assert result.stderr == "error msg"
