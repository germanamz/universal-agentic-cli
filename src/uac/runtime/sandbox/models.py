"""Data models for the sandbox subsystem."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SandboxConfig(BaseModel):
    """Configuration for a sandbox executor."""

    timeout: float = Field(default=30.0, description="Max execution time in seconds.")
    memory_limit: str = Field(default="256m", description="Memory limit (Docker format, e.g. '256m').")
    cpu_limit: float = Field(default=1.0, description="CPU quota (number of cores).")
    network_enabled: bool = Field(default=False, description="Allow network access inside the sandbox.")
    image: str = Field(default="python:3.12-slim", description="Docker image to use.")
    workdir: str = Field(default="/workspace", description="Working directory inside the container.")
    read_only: bool = Field(default=True, description="Mount the root filesystem as read-only.")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables to inject.")


class ExecutionRequest(BaseModel):
    """A request to execute a command inside a sandbox."""

    command: list[str] = Field(..., description="Command and arguments to execute.")
    stdin: str | None = Field(default=None, description="Optional stdin input.")
    timeout: float | None = Field(default=None, description="Per-request timeout override.")
    env: dict[str, str] = Field(default_factory=dict, description="Extra env vars for this request.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata.")


class SandboxResult(BaseModel):
    """Result of a sandboxed execution."""

    exit_code: int = Field(..., description="Process exit code.")
    stdout: str = Field(default="", description="Captured stdout.")
    stderr: str = Field(default="", description="Captured stderr.")
    timed_out: bool = Field(default=False, description="Whether the execution timed out.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata.")
