# Contributing Guide

## Dev Setup

```bash
# Clone the repository
git clone <repo-url>
cd universal-agentic-cli

# Install dependencies (requires uv)
uv sync

# Install with dev dependencies
uv sync --group dev
```

## Running Tests

```bash
# All tests
uv run pytest tests/

# With coverage
uv run pytest tests/ --cov=uac

# Specific test directory
uv run pytest tests/e2e/ -v

# Single test file
uv run pytest tests/e2e/test_pipeline_e2e.py -v
```

## Code Style

### Linting (Ruff)

```bash
# Check
uv run ruff check src/ tests/

# Auto-fix
uv run ruff check src/ tests/ --fix
```

Configuration is in `pyproject.toml`. Key rules: `E`, `W`, `F`, `I` (isort), `N` (naming), `UP` (pyupgrade), `B` (bugbear), `SIM`, `TCH`, `RUF`.

### Type Checking (Pyright)

```bash
uv run pyright src/ tests/
```

Strict mode is enabled. Python 3.11+ is required.

## Testing Conventions

### Async Tests

All async tests use `pytest-asyncio` with `asyncio_mode = "auto"` — no need for explicit `@pytest.mark.asyncio` decorators (though they are accepted).

### Mock Patterns

**Mocking LiteLLM** — patch `uac.core.interface.client.litellm`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

async def test_example():
    with patch("uac.core.interface.client.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)
        # ... test code
```

**Creating mock LLM responses** — use the E2E helper:

```python
from tests.e2e.conftest import make_mock_litellm_response

response = make_mock_litellm_response(
    content="Hello",
    model="openai/gpt-4o",
)
```

**Mocking MCP transports** — patch `MCPClient._create_transport`:

```python
from unittest.mock import patch
from uac.protocols.mcp.client import MCPClient

with patch.object(MCPClient, "_create_transport", return_value=mock_transport):
    async with MCPClient(ref) as client:
        # ... test code
```

**Creating AgentNodes for tests**:

```python
from uac.core.interface.models import CanonicalMessage
from uac.core.orchestration.models import AgentManifest
from uac.core.orchestration.primitives import AgentNode

manifest = AgentManifest(name="test", system_prompt_template="You are $name.")
client = MagicMock()
client.generate = AsyncMock(return_value=CanonicalMessage.assistant("response"))
node = AgentNode(manifest=manifest, client=client)
```

## Task Tracking

This project uses [Beads](https://github.com/steveyegge/beads) (`bd`) for task management:

```bash
# List ready tasks
bd ready

# Claim a task
bd update <id> --claim

# Create a task
bd create "Title" -p <priority>

# Add dependency
bd dep add <child> <parent>
```

Task IDs are hierarchical: `bd-a3f8` (epic), `bd-a3f8.1` (task), `bd-a3f8.1.1` (sub-task).
