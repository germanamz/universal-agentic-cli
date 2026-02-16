"""LocalSandbox — executes commands on the host with loud warnings.

This is a development/fallback executor that runs commands directly on the
host machine.  It is **not** sandboxed and emits prominent warnings every
time it is instantiated or used.
"""

from __future__ import annotations

import asyncio
import logging
import warnings

from uac.runtime.errors import SandboxError, SandboxTimeoutError
from uac.runtime.sandbox.models import ExecutionRequest, SandboxConfig, SandboxResult

logger = logging.getLogger(__name__)

_WARNING_MSG = (
    "LocalSandbox executes commands directly on the host with NO isolation. "
    "Use DockerSandbox for production workloads."
)


class LocalSandbox:
    """Host-local command executor (no isolation).

    Satisfies the :class:`~uac.runtime.sandbox.executor.SandboxExecutor`
    protocol but provides **zero** sandboxing.  Emits ``warnings.warn`` and
    ``logger.warning`` on construction and on every ``execute()`` call.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()
        warnings.warn(_WARNING_MSG, stacklevel=2)
        logger.warning(_WARNING_MSG)

    async def execute(self, request: ExecutionRequest) -> SandboxResult:
        """Run a command on the host."""
        logger.warning("LocalSandbox: executing %s on host (UNSANDBOXED)", request.command)

        timeout = request.timeout or self._config.timeout
        env = {**self._config.env, **request.env} or None

        try:
            proc = await asyncio.create_subprocess_exec(
                *request.command,
                stdin=asyncio.subprocess.PIPE if request.stdin else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdin_bytes = request.stdin.encode() if request.stdin else None
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()  # type: ignore[union-attr]
            raise SandboxTimeoutError(timeout)
        except OSError as exc:
            raise SandboxError(str(exc)) from exc

        return SandboxResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(errors="replace") if stdout else "",
            stderr=stderr.decode(errors="replace") if stderr else "",
        )

    async def cleanup(self) -> None:
        """No-op — nothing to clean up for host execution."""
