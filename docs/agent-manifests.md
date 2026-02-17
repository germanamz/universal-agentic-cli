# Agent Manifests Reference

Agent manifests are YAML files that define an agent's identity, capabilities, and tool dependencies. They live in the `agents/` directory and are loaded at orchestration start-up.

## Schema

```yaml
# Required
name: string              # Unique agent identifier

# Optional
version: string           # Manifest version (default: "1.0")
description: string       # Human-readable description

model_requirements:
  min_context_window: int  # Minimum context window (default: 4096)
  capabilities: [string]   # Required capabilities (e.g., ["native_tool_calling"])
  preferred_model: string  # Preferred model identifier

system_prompt_template: string  # Prompt template with variable substitution

mcp_servers:               # MCP server dependencies
  - name: string           # Server identifier
    transport: stdio|websocket  # Transport type (default: stdio)
    command: string         # Command for stdio transport
    url: string             # URL for websocket transport
    env: {string: string}   # Environment variables

input_schema:              # Expected input format
  type: string
  description: string
  properties: {string: any}
  required: [string]

output_schema:             # Expected output format
  type: string
  description: string
  properties: {string: any}
  required: [string]

metadata: {string: any}   # Arbitrary key-value metadata
```

## Template Variables

System prompt templates support `$variable` substitution. Built-in variables:

| Variable | Value |
|----------|-------|
| `$name` | Agent's `name` field |
| `$description` | Agent's `description` field |
| `$version` | Agent's `version` field |

Custom variables can be passed via `prompt_variables` when constructing an `AgentNode`.

## Examples

### Basic Agent

```yaml
name: summariser
version: "1.0"
description: Summarises long documents into bullet points.
system_prompt_template: |
  You are $name. Your task is to read the provided document
  and produce a concise bullet-point summary.
```

### Agent with MCP Tools

```yaml
name: file-editor
version: "1.0"
description: Edits files using MCP filesystem tools.
model_requirements:
  min_context_window: 8192
  capabilities: [native_tool_calling]
system_prompt_template: |
  You are $name. You have access to filesystem tools.
  Use them to read, write, and modify files as instructed.
mcp_servers:
  - name: filesystem
    transport: stdio
    command: npx @modelcontextprotocol/server-filesystem /workspace
```

### Agent with Model Requirements

```yaml
name: vision-analyst
version: "1.0"
description: Analyses images and provides descriptions.
model_requirements:
  min_context_window: 128000
  capabilities: [native_tool_calling, vision]
  preferred_model: openai/gpt-4o
system_prompt_template: |
  You are $name, a vision analysis agent.
  Examine the provided images and describe what you see.
input_schema:
  type: object
  properties:
    image_url:
      type: string
  required: [image_url]
output_schema:
  type: object
  properties:
    description:
      type: string
    objects:
      type: array
```

## Workflow Integration

Agents are referenced in workflow YAML files:

```yaml
# workflow.yaml
name: document-pipeline
topology:
  type: pipeline
  order: [researcher, summariser, editor]
model:
  model: openai/gpt-4o
  api_key: ${OPENAI_API_KEY}
agents:
  researcher:
    manifest: agents/researcher.yaml
  summariser:
    manifest: agents/summariser.yaml
  editor:
    manifest: agents/editor.yaml
    model:  # Per-agent model override
      model: anthropic/claude-3-haiku
      api_key: ${ANTHROPIC_API_KEY}
```

Per-agent model overrides take precedence over the workflow-level `model` setting. If neither is specified, the agent's `preferred_model` from `model_requirements` is used, falling back to `openai/gpt-4o`.
