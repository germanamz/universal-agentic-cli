"""DockerSandbox — executes commands in ephemeral Docker containers.

Uses the ``docker`` CLI via subprocess (no docker-py dependency), matching
the subprocess pattern used in :mod:`uac.protocols.utcp.executor`.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from uac.runtime.errors import SandboxError, SandboxTimeoutError
from uac.runtime.sandbox.models import ExecutionRequest, SandboxConfig, SandboxResult

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Ephemeral Docker container sandbox.

    Satisfies the :class:`~uac.runtime.sandbox.executor.SandboxExecutor`
    protocol.

    Each ``execute()`` call:
    1. ``docker create`` with resource limits and network isolation.
    2. ``docker start`` the container.
    3. ``docker wait`` (with timeout) for it to finish.
    4. ``docker logs`` to capture output.
    5. ``docker rm -f`` in a ``finally`` block.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()
        self._active_containers: set[str] = set()

    async def execute(self, request: ExecutionRequest) -> SandboxResult:
        """Run a command inside an ephemeral Docker container."""
        container_name = f"uac-sandbox-{uuid.uuid4().hex[:12]}"
        timeout = request.timeout or self._config.timeout

        create_cmd = self._build_create_command(container_name, request)

        try:
            # 1. docker create
            await self._run_docker(create_cmd)
            self._active_containers.add(container_name)

            # 2. docker start
            await self._run_docker(["docker", "start", container_name])

            # 3. docker wait (with timeout)
            try:
                wait_result = await asyncio.wait_for(
                    self._run_docker(["docker", "wait", container_name]),
                    timeout=timeout,
                )
                exit_code = int(wait_result.stdout.strip()) if wait_result.stdout.strip() else 1
            except TimeoutError:
                # Kill the container on timeout
                await self._run_docker(
                    ["docker", "kill", container_name],
                    ignore_errors=True,
                )
                raise SandboxTimeoutError(timeout)

            # 4. docker logs
            logs = await self._run_docker(
                ["docker", "logs", container_name],
                capture_stderr=True,
            )

            return SandboxResult(
                exit_code=exit_code,
                stdout=logs.stdout,
                stderr=logs.stderr,
            )
        finally:
            # 5. Always remove the container
            await self._remove_container(container_name)

    async def cleanup(self) -> None:
        """Remove all tracked containers."""
        containers = list(self._active_containers)
        for name in containers:
            await self._remove_container(name)

    def _build_create_command(
        self,
        container_name: str,
        request: ExecutionRequest,
    ) -> list[str]:
        """Build the ``docker create`` command with resource limits."""
        cfg = self._config
        cmd: list[str] = [
            "docker", "create",
            "--name", container_name,
            "--memory", cfg.memory_limit,
            "--cpus", str(cfg.cpu_limit),
            "--workdir", cfg.workdir,
        ]

        if not cfg.network_enabled:
            cmd.extend(["--network", "none"])

        if cfg.read_only:
            cmd.append("--read-only")
            # /tmp needs to be writable for most tools
            cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])

        # Environment variables
        merged_env = {**cfg.env, **request.env}
        for key, value in merged_env.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Image + command
        cmd.append(cfg.image)
        cmd.extend(request.command)

        return cmd

    async def _remove_container(self, name: str) -> None:
        """Force-remove a container, swallowing errors."""
        await self._run_docker(["docker", "rm", "-f", name], ignore_errors=True)
        self._active_containers.discard(name)

    @staticmethod
    async def _run_docker(
        cmd: list[str],
        *,
        ignore_errors: bool = False,
        capture_stderr: bool = False,
    ) -> _DockerOutput:
        """Run a docker CLI command and return its output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
        except OSError as exc:
            if ignore_errors:
                return _DockerOutput()
            raise SandboxError(f"Failed to run docker: {exc}") from exc

        stdout = stdout_bytes.decode(errors="replace").strip() if stdout_bytes else ""
        stderr = stderr_bytes.decode(errors="replace").strip() if stderr_bytes else ""

        if proc.returncode != 0 and not ignore_errors:
            raise SandboxError(f"docker command failed (rc={proc.returncode}): {stderr or stdout}")

        return _DockerOutput(
            stdout=stdout,
            stderr=stderr if capture_stderr else "",
        )


class _DockerOutput:
    """Simple container for docker CLI output."""

    __slots__ = ("stdout", "stderr")

    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr
