# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Universal Agentic CLI (UAC) — a model-agnostic, multi-agent orchestration platform. The goal is to decouple orchestration logic from specific LLM providers and tool implementations, enabling hot-swappable models and plug-and-play tool integration via standard protocols (MCP, A2A, UTCP).

**Current state: pre-implementation (design/documentation only).** No source code, build system, or tests exist yet. The project is defined by four architectural documents:

- `PRODUCT.md` — comprehensive technical research and architectural vision (~45KB)
- `REQUIREMENTS.md` — functional and non-functional requirements
- `DOMAINS.md` — domain breakdown into 8 logical areas
- `STRUCTURE.md` — proposed directory layout and folder responsibilities

## Architecture

### Core Abstractions

1. **Unified Model Interface (UMI)** — normalizes communication across LLM providers. Defines a Canonical Message Schema (CMS) as the internal message format, with runtime transpilation to provider-specific formats (OpenAI, Anthropic, Gemini). Built on LiteLLM.

2. **Cognitive Polyfilling** — automatically injects ReAct (Thought/Action/Observation) prompts and regex-based output parsing for models lacking native function calling. Enables graceful degradation from GPT-4 to local 7B models.

3. **Orchestration Topologies** — three configurable patterns defined in YAML:
   - **Star**: hierarchical supervisor directing workers
   - **Pipeline**: sequential deterministic hand-offs
   - **Mesh**: decentralized event-driven A2A delegation

4. **Blackboard Architecture** — shared JSON/Redis state store for Belief States, Execution Traces, and Artifacts. Decouples state from LLM context windows. Enables time-travel debugging.

5. **Protocol Layer** — MCP client for tool discovery, A2A handshaking via `.well-known/agent.json`, and UTCP fallback for legacy REST/CLI tools.

6. **Runtime Safety** — Docker/Wasm sandboxing for code execution, Human-in-the-Loop (HITL) gatekeeper for high-risk actions.

7. **Context Management** — token monitoring with model-specific tokenizers, pruning via sliding window, recursive summarization, or RAG vector offloading.

8. **Observability** — OpenTelemetry tracing across agent hops, Reflexion loops for self-correction on malformed outputs.

### Proposed Directory Structure

```
uac-root/
├── agents/          # YAML Agent Manifests (persona, capabilities, tool deps)
├── bin/             # CLI entrypoints
├── configs/         # Provider keys, default topologies
├── core/
│   ├── blackboard/  # Shared state (JSON/Redis)
│   ├── context/     # Token logistics, pruning, summarization
│   ├── interface/   # UMI & transpilation
│   ├── orchestration/ # Topology engines
│   └── polyfills/   # ReAct injection & regex parsing
├── protocols/
│   ├── a2a/         # Agent-to-Agent protocol
│   ├── mcp/         # Model Context Protocol client
│   └── utcp/        # Universal Tool Calling Protocol
├── runtime/
│   ├── gatekeeper/  # HITL approval
│   └── sandbox/     # Docker/Wasm isolation
├── sdk/             # Custom tools/agents SDK
└── utils/           # OpenTelemetry, Reflexion loops
```

### Key Design Patterns

- **Adapter/Transpiler** — UMI translates CMS to provider formats at runtime
- **Polyfill/Strategy** — capability detection switches between native and prompted (ReAct) modes
- **Blackboard** — shared mutable state decoupled from any single agent
- **Topology Strategy** — orchestration logic is swappable via YAML config

### Design Constraints

- Orchestration must never depend on a specific provider API
- Agent Manifests are standalone YAML — agents are portable across deployments
- Models must be hot-swappable without code changes
- All tool execution must be sandboxed (Docker or Wasm)
- The system must support async/event-driven execution

## Task Management with Beads

This project uses [Beads](https://github.com/steveyegge/beads) (`bd`) for task tracking and parallelism. Beads is a Git-backed issue tracker with DAG-based dependencies designed for AI agents.

### Workflow

1. **Check for ready work** — Run `bd ready` to list tasks with no open blockers.
2. **Claim a task** — Run `bd update <id> --claim` to atomically assign yourself and mark it in-progress.
3. **Create tasks** — Run `bd create "Title" -p <priority>` (P0 = critical, P3 = low).
4. **Add dependencies** — Run `bd dep add <child> <parent>` to link blockers.
5. **Inspect a task** — Run `bd show <id>` to view details and audit trail.
6. **Close tasks** — Update status when work is complete.

### Parallelism

When multiple tasks are `ready` (no unresolved blockers), they can be worked on in parallel by separate agents or sessions. Use `bd ready` to identify independent work items and `bd dep add` to enforce ordering constraints. The DAG structure prevents agents from attempting blocked tasks.

### Key Conventions

- Always run `bd ready` before starting work to pick the highest-impact unblocked task.
- Use `--json` flag for programmatic/agent consumption (e.g., `bd ready --json`).
- Task IDs are hierarchical: `bd-a3f8` (epic), `bd-a3f8.1` (task), `bd-a3f8.1.1` (sub-task).
- Beads data lives in `.beads/` and is committed to Git — do not manually edit these files.
