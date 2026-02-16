# Execution Plan: Universal Agentic CLI (UAC)

## Guiding Principles

- Build from the inside out: foundational abstractions first, integration points last.
- Each phase produces a testable, self-contained deliverable.
- No phase should depend on a later phase.
- Prefer working end-to-end slices over broad horizontal layers — get a single model talking through the UMI before wiring up three topologies.

---

## Phase 0 — Project Scaffolding

**Goal:** Establish the repo structure, build system, and development tooling so every subsequent phase lands in a consistent environment.

**Deliverables:**
- Python package skeleton matching `STRUCTURE.md` (pyproject.toml, src layout under `uac/`)
- Dependency management (Poetry or uv)
- Linting/formatting (ruff) and type checking (pyright or mypy)
- Test harness (pytest, pytest-asyncio)
- CI pipeline stub (GitHub Actions: lint + test)
- Empty `__init__.py` files for every package in the tree

**Depends on:** Nothing.

---

## Phase 1 — Canonical Message Schema & Unified Model Interface

**Goal:** Define the internal message types and build the adapter layer so the rest of the system never touches provider-specific formats.

**Covers:** FR1.1, FR1.2, FR1.3 · Domain 1 (UMI & Translation)

### 1a — Canonical Message Schema (CMS)

- Define Pydantic models: `CanonicalMessage`, `ContentPart` (Text, Image, Audio), `ToolCall`, `ToolResult`.
- Define `ConversationHistory` as an ordered container of `CanonicalMessage`.
- Unit tests: round-trip serialization, schema validation for edge cases (empty content, multiple tool calls).

### 1b — Provider Transpilers

- Implement transpiler interface: `Transpiler.to_provider(history) -> ProviderPayload` and `Transpiler.from_provider(response) -> CanonicalMessage`.
- Concrete transpilers for **OpenAI**, **Anthropic**, and **Gemini**:
  - OpenAI: direct mapping (CMS is closest to ChatML).
  - Anthropic: extract system message to top-level param, merge consecutive same-role messages.
  - Gemini: map `assistant` → `model`, convert tool calls to FunctionCall structure.
- Unit tests with recorded fixtures (no live API calls).

### 1c — LiteLLM Integration

- Wrap LiteLLM behind a `ModelClient` abstraction (`generate(messages, tools?, config) -> CanonicalMessage`).
- `ModelConfig` dataclass: provider, model name, API key ref, context window size, capability flags.
- Integration test: call a real provider with a trivial prompt (gated behind an env flag).

**Depends on:** Phase 0.

---

## Phase 2 — Capability Detection & Polyfilling

**Goal:** Enable tool use on any model, whether it supports native function calling or not.

**Covers:** FR2.1, FR2.2, FR2.3 · Domain 2 (Polyfilling)

### 2a — Capability Registry

- `CapabilityProfile` dataclass: `supports_native_tools`, `supports_vision`, `context_window`, etc.
- Static registry (JSON/YAML) mapping known model identifiers to profiles.
- Fallback heuristic: if model is unknown, assume no native tools.

### 2b — ReAct Polyfill

- `ReActInjector`: given a list of tool definitions, produces the ReAct system prompt block (Thought/Action/Action Input/Observation format).
- `ReActParser`: regex-based parser that scans raw text output, extracts `Action` and `Action Input`, and returns a `ToolCall`.
- Strategy interface: `ToolCallingStrategy` with two implementations — `NativeStrategy` (pass tool schemas to the API) and `PromptedStrategy` (inject ReAct, parse output).
- The `ModelClient` from Phase 1c selects strategy based on the capability profile.

### 2c — End-to-End Tool Calling Smoke Test

- Define a trivial tool (e.g., `calculator(expression: str) -> float`).
- Test that the full path works for both a native-capable model stub and a prompted-mode model stub.

**Depends on:** Phase 1.

---

## Phase 3 — Blackboard & State Management

**Goal:** Build the shared state store so agents can read/write structured data independent of LLM context.

**Covers:** FR3.2 · Domain 5 (State Management)

### 3a — Blackboard Core

- `Blackboard` class backed by an in-memory dict (JSON-serializable).
- Sections: `belief_state` (str), `execution_trace` (list of `TraceEntry`), `artifacts` (key-value), `pending_tasks` (priority queue).
- Reducer API: `blackboard.apply(delta: StateDelta) -> Blackboard`. Deltas are partial updates merged via deep-merge rules.
- Snapshot/restore for time-travel: `blackboard.snapshot() -> bytes`, `Blackboard.restore(bytes)`.

### 3b — Redis Backend (Optional / Pluggable)

- `BlackboardBackend` protocol with `InMemoryBackend` and `RedisBackend` implementations.
- Configuration-driven selection.

### 3c — Context Slicing

- `ContextSlicer`: given a Blackboard and an agent's manifest, produce a minimal `ContextSlice` (filtered artifacts, relevant trace entries) to reduce token usage.

**Depends on:** Phase 0 (no dependency on UMI).

---

## Phase 4 — Agent Manifests & Orchestration Engine

**Goal:** Parse agent definitions and implement the three topology engines that drive multi-agent workflows.

**Covers:** FR3.1, NFR1.1, NFR1.2 · Domain 3 (Orchestration)

### 4a — Agent Manifest Parser

- Pydantic model for the Agent Manifest YAML schema (name, version, description, model_requirements, system_prompt_template, mcp_servers, input/output schemas).
- Loader: read YAML from `agents/` directory, validate, return typed objects.
- Jinja2 rendering for `system_prompt_template`.

### 4b — Orchestration Primitives

- `AgentNode`: wraps a manifest + `ModelClient` + tools. Exposes `async step(context_slice) -> StateDelta`.
- `Orchestrator` base class: `async run(goal: str) -> BlackboardSnapshot`.
- Execution loop: select agent → slice context → call agent → apply delta → check termination.

### 4c — Topology: Pipeline (Sequential)

- `PipelineOrchestrator`: agents are an ordered list; output of agent N feeds agent N+1.
- YAML config: `topology: pipeline`, `agents: [a, b, c]`.

### 4d — Topology: Star (Supervisor)

- `StarOrchestrator`: a supervisor agent inspects the blackboard and emits a routing command (`Route: <agent_name>`).
- Parse supervisor output for routing directives.
- Termination condition: supervisor emits `DONE` or max iterations reached.

### 4e — Topology: Mesh (Event-Driven)

- `MeshOrchestrator`: agents subscribe to topic patterns on an internal event bus.
- Messages are broadcast; agents that match the topic activate.
- Built on `asyncio.Queue` or a lightweight pub/sub.

**Depends on:** Phase 1 (ModelClient), Phase 2 (tool calling), Phase 3 (Blackboard).

---

## Phase 5 — Context Optimization

**Goal:** Prevent context overflow and manage token budgets across heterogeneous models.

**Covers:** FR5.1, FR5.2 · Domain 6 (Context Optimization)

### 5a — Token Counting

- `TokenCounter` protocol with implementations wrapping `tiktoken` (OpenAI models) and HuggingFace tokenizers.
- Auto-select tokenizer based on `ModelConfig`.

### 5b — Pruning Strategies

- `ContextPruner` with pluggable strategies:
  - **SlidingWindow**: drop oldest messages, always preserve system prompt.
  - **Summarizer**: call a cheap model to compress old messages into a narrative block.
  - **VectorOffload**: move old messages to a vector store, replace with retrieval stub.
- Middleware integration: `ContextManager` wraps `ModelClient.generate`, transparently pruning before each call.

**Depends on:** Phase 1 (ModelClient, CMS).

---

## Phase 6 — Protocol Layer (MCP, A2A, UTCP)

**Goal:** Connect the orchestrator to external tools and remote agents via standard protocols.

**Covers:** FR4.1, FR4.2, FR4.3 · Domain 4 (Connectivity)

### 6a — MCP Client

- Implement JSON-RPC client over stdio and WebSocket transports.
- Discovery: connect to an MCP server, perform capability handshake, enumerate available tools.
- Map discovered tools into the internal tool schema (so they work with both native and polyfilled strategies).

### 6b — A2A Discovery & Delegation

- Fetch and parse `.well-known/agent.json` (Agent Cards).
- `A2AClient`: send a JSON-RPC task request to a remote agent, receive results.
- Integrate with Mesh topology: remote agents appear as nodes.

### 6c — UTCP Fallback

- Schema for defining REST/CLI tools declaratively (URL template, HTTP method, headers, body mapping).
- `UTCPExecutor`: constructs and fires raw HTTP requests or shell commands from the schema.

**Depends on:** Phase 2 (tool schema), Phase 4 (orchestration for A2A integration).

---

## Phase 7 — Runtime Safety

**Goal:** Sandbox tool execution and add human approval gates.

**Covers:** NFR2.1, NFR2.2 · Domain 7 (Security)

### 7a — Execution Sandbox

- `SandboxExecutor` protocol.
- `DockerSandbox`: spin up an ephemeral container, execute a command, capture stdout/stderr, tear down.
- `WasmSandbox` (stretch): compile tool logic to Wasm, run via Wasmtime.
- Fallback `LocalSandbox` for development (with big warnings).

### 7b — HITL Gatekeeper

- `Gatekeeper` middleware: intercept tool calls tagged as side-effects (file write, network request, code execution).
- CLI prompt: display the pending action, wait for user `y/n`.
- Configurable allowlist of "safe" tool categories that skip approval.

**Depends on:** Phase 4 (orchestration loop calls tools through gatekeeper).

---

## Phase 8 — Observability & Self-Correction

**Goal:** Make multi-agent workflows debuggable and resilient to transient model failures.

**Covers:** NFR3.1, NFR3.2 · Domain 8 (Observability)

### 8a — OpenTelemetry Integration

- Instrument the orchestration loop: span per agent step, attributes for model, tokens used, latency.
- Instrument tool calls: span per execution with input/output.
- Export to stdout (JSON) and optionally to an OTLP collector.

### 8b — Reflexion Loops

- `ReflexionMiddleware`: wraps `AgentNode.step()`.
  - If the output fails validation (malformed JSON, schema mismatch), inject the error as a user message and retry.
  - Configurable max retries (default 3).
  - Log each retry as a child span.

**Depends on:** Phase 4 (orchestration), Phase 1 (ModelClient).

---

## Phase 9 — CLI & SDK

**Goal:** Expose everything through a user-friendly command-line interface and an importable Python SDK.

### 9a — CLI Entrypoint (`bin/`)

- `uac run <workflow.yaml>` — load topology + manifests, execute, stream output.
- `uac agents list` — enumerate available agent manifests.
- `uac tools discover <mcp-server-url>` — connect to an MCP server, list tools.
- `uac inspect <snapshot-file>` — pretty-print a Blackboard snapshot.
- Built on `click` or `typer`.

### 9b — Python SDK (`sdk/`)

- Programmatic API mirroring the CLI:
  ```python
  from uac import Orchestrator, load_manifest
  orch = Orchestrator.from_yaml("workflow.yaml")
  result = await orch.run("Analyze this stock")
  ```
- Public types re-exported for downstream consumers.

**Depends on:** All previous phases.

---

## Phase 10 — Integration Testing & Documentation

### 10a — End-to-End Scenarios

- **Scenario 1 (Pipeline):** Three-agent pipeline (Research → Draft → Review) with a real model (or recorded fixtures).
- **Scenario 2 (Star):** Supervisor delegates between a coder and a tester agent.
- **Scenario 3 (Hot-swap):** Run the same workflow with OpenAI, then swap to a local Ollama model — verify polyfill activates and output is structurally equivalent.
- **Scenario 4 (MCP):** Connect to a community MCP server, discover tools, use them in a workflow.

### 10b — Documentation

- README with quickstart.
- Architecture guide (distilled from PRODUCT.md).
- Agent Manifest authoring guide.
- Contributing guide.

**Depends on:** All previous phases.

---

## Dependency Graph (Summary)

```
Phase 0 ─────────────────────────────────────┐
  │                                           │
  ├──► Phase 1 (UMI) ──► Phase 2 (Polyfill)  │
  │         │                   │             │
  │         │                   │             │
  │         ▼                   ▼             │
  │    Phase 5 (Context)   Phase 4 (Orch) ◄──┤◄── Phase 3 (Blackboard)
  │                             │             │
  │                             ▼             │
  │                     Phase 6 (Protocols)   │
  │                             │             │
  │                             ▼             │
  │                     Phase 7 (Runtime)     │
  │                             │             │
  │                             ▼             │
  │                     Phase 8 (Observability)
  │                             │
  │                             ▼
  └────────────────────► Phase 9 (CLI & SDK)
                                │
                                ▼
                        Phase 10 (E2E & Docs)
```

## Parallelization Opportunities

- **Phase 1 and Phase 3** can run in parallel (Blackboard has no UMI dependency).
- **Phase 5 (Context)** can run in parallel with **Phase 4 (Orchestration)** once Phase 1 is done.
- **Phase 7 (Runtime)** and **Phase 8 (Observability)** can run in parallel.
- Within Phase 4, Pipeline/Star/Mesh topologies can be developed concurrently once 4b is complete.
- Within Phase 6, MCP/A2A/UTCP are largely independent of each other.

## Risk Areas

| Risk | Impact | Mitigation |
|------|--------|------------|
| ReAct polyfill unreliable on small models | Core value prop fails | Extensive prompt iteration; fallback to structured XML format; test across model families early |
| LiteLLM gaps or breaking changes | Provider support blocked | Pin version; wrap behind `ModelClient` abstraction so LiteLLM is swappable |
| MCP ecosystem immaturity | Limited tool ecosystem | UTCP fallback covers REST/CLI tools; MCP is additive, not blocking |
| Context pruning loses critical info | Agent reasoning degrades | Summarization preserves key facts; vector offload keeps everything retrievable |
| Docker sandbox latency | Slow tool execution | Wasm as lightweight alternative; local sandbox for dev; lazy container reuse |
