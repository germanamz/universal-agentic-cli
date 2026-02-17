# Architecture Overview

## Layer Diagram

```
+---------------------------------------------------------------+
|                        CLI (uac run)                          |
+---------------------------------------------------------------+
|                     SDK / WorkflowRunner                      |
+---------------------------------------------------------------+
|                   Orchestration Topologies                     |
|              Pipeline  |  Star  |  Mesh                       |
+---------------------------------------------------------------+
|              AgentNode (manifest + client + tools)             |
+-----------------+-------------------+-------------------------+
|  Unified Model  |   Blackboard      |   Protocol Layer        |
|  Interface      |   (shared state)  |   MCP / A2A / UTCP      |
+-----------------+-------------------+-------------------------+
|  Polyfills      |   Context Mgmt    |   Runtime Safety        |
|  (ReAct/Native) |   (token/pruning) |   (sandbox/gatekeeper)  |
+-----------------+-------------------+-------------------------+
|                     LiteLLM + Providers                       |
+---------------------------------------------------------------+
```

## Core Abstractions

### Unified Model Interface (UMI)

The UMI normalises communication across LLM providers. All internal code works with the **Canonical Message Schema (CMS)** — a superset of all provider message formats.

- `CanonicalMessage` — roles: system, user, assistant, tool
- `ConversationHistory` — ordered message container
- `ModelClient` — async wrapper around LiteLLM with CMS input/output
- `Transpiler` — converts CMS to/from provider formats (OpenAI, Anthropic, Gemini)

### Cognitive Polyfilling

The polyfill layer automatically detects model capabilities and selects the appropriate tool-calling strategy:

- **NativeStrategy** — passthrough for models with native function calling (GPT-4, Claude 3, Gemini)
- **PromptedStrategy** — injects ReAct (Thought/Action/Observation) prompts and parses structured output from free text for models without native tool calling (Llama, Mistral)

Strategy selection is automatic via `CapabilityRegistry` which maps model identifiers to capability profiles.

### Orchestration Topologies

Three configurable execution patterns, all extending the base `Orchestrator` class:

| Topology | Pattern | Use Case |
|----------|---------|----------|
| **Pipeline** | Sequential: A -> B -> C | Document processing, ETL |
| **Star** | Supervisor directs workers | Complex reasoning, delegation |
| **Mesh** | Event-driven pub/sub | Decentralised collaboration |

The orchestration loop (`Orchestrator.run()`):

1. Set initial goal on blackboard
2. `select_agent(iteration)` — topology decides who runs next
3. `agent.step(context_slice)` — agent generates a response
4. `blackboard.apply(delta)` — merge response into shared state
5. `is_done(iteration)` — check termination condition
6. Repeat until done or `max_iterations` reached

### Blackboard Architecture

The `Blackboard` is a shared mutable state store for multi-agent coordination:

- **belief_state** — current high-level goal/status string
- **execution_trace** — ordered list of `TraceEntry` records (agent_id, action, data)
- **artifacts** — key-value store for agent outputs, deep-merged across steps
- **pending_tasks** — priority queue of `TaskItem` objects

Agents interact with the blackboard via `StateDelta` objects (partial updates), never by direct mutation. The blackboard supports `snapshot()`/`restore()` for serialisation.

### Protocol Layer

- **MCP Client** — connects to Model Context Protocol servers, discovers tools via `tools/list`, executes via `tools/call`. Supports stdio and WebSocket transports.
- **ToolDispatcher** — maintains a name-to-provider routing table. Merges tools from multiple MCP servers into a single schema list.
- **ToolProvider** — protocol interface that MCP, A2A, and UTCP adapters implement.

### Runtime Safety

- **Gatekeeper** — intercepts tool calls and applies policy rules (allow/deny/ask). Supports configurable approval timeouts.
- **SafeDispatcher** — wraps `ToolDispatcher` with gatekeeper checks before execution.
- **Sandbox** — Docker/Wasm isolation for code execution (planned).

## Data Flow

```
WorkflowRunner.run(goal)
  |
  +-> Parse agent manifests (YAML)
  +-> Create ModelClient per agent (with capability detection)
  +-> Connect MCP servers, discover tools
  +-> Build AgentNode instances (manifest + client + tools)
  +-> Instantiate orchestrator (Pipeline/Star/Mesh)
  |
  +-> Orchestrator.run(goal)
        |
        +-> Set belief_state = goal
        |
        +-> Loop:
        |     select_agent(i) -> AgentNode
        |     blackboard.slice(agent_id) -> ContextSlice
        |     agent.step(context) -> StateDelta
        |     blackboard.apply(delta)
        |     is_done(i)?
        |
        +-> Return final Blackboard
```

## Topology Comparison

| Feature | Pipeline | Star | Mesh |
|---------|----------|------|------|
| Control flow | Fixed order | Supervisor-directed | Event-driven |
| Agent selection | Sequential index | Supervisor `Route:` directive | Subscription matching |
| Termination | All agents done | Supervisor says `DONE` | No pending events |
| Parallelism | None | Workers run serially | Concurrent subscribers |
| Best for | Deterministic workflows | Dynamic delegation | Collaborative tasks |
