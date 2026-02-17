# UAC — Universal Agentic CLI

A model-agnostic, multi-agent orchestration platform. UAC decouples orchestration logic from specific LLM providers and tool implementations, enabling hot-swappable models and plug-and-play tool integration via standard protocols.

## Features

- **Unified Model Interface** — Talk to OpenAI, Anthropic, Gemini, or local models through a single canonical message format with automatic runtime transpilation.
- **Cognitive Polyfilling** — Models without native tool calling get automatic ReAct prompt injection, enabling graceful degradation from GPT-4 to local 7B models.
- **Configurable Topologies** — Define agent workflows as Pipeline (sequential), Star (supervisor-directed), or Mesh (event-driven) via YAML.
- **MCP Tool Integration** — Discover and execute tools from any MCP-compatible server with automatic schema translation.
- **Blackboard Architecture** — Shared JSON state store for belief states, execution traces, and artifacts, decoupled from LLM context windows.
- **Runtime Safety** — Human-in-the-loop gatekeeper for high-risk actions with configurable approval policies.

## Quick Start

### Install

```bash
uv sync
```

### Define Agents

Create agent manifests in `agents/`:

```yaml
# agents/researcher.yaml
name: researcher
version: "1.0"
description: Researches topics and provides findings.
system_prompt_template: |
  You are $name, a research agent.
  Investigate the given topic thoroughly.
```

### Define a Workflow

```yaml
# workflows/research-pipeline.yaml
name: research-pipeline
topology:
  type: pipeline
  order: [researcher, drafter, reviewer]
model:
  model: openai/gpt-4o
  api_key: ${OPENAI_API_KEY}
agents:
  researcher:
    manifest: agents/researcher.yaml
  drafter:
    manifest: agents/drafter.yaml
  reviewer:
    manifest: agents/reviewer.yaml
```

### Run a Workflow

```bash
uac run workflows/research-pipeline.yaml --goal "Write a report on AI safety"
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `uac run <workflow.yaml> --goal "..."` | Execute a workflow |
| `uac agents list <dir>` | List agent manifests |
| `uac agents validate <file>` | Validate an agent manifest |
| `uac tools list` | List available tools |
| `uac inspect <workflow.yaml>` | Inspect workflow configuration |
| `uac --version` | Show version |

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Agent Manifests Reference](docs/agent-manifests.md)
- [Contributing Guide](docs/contributing.md)

## License

See [LICENSE](LICENSE) for details.
