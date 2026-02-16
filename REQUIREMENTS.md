# Requirements Specification: Universal Agentic CLI (UAC)

## 1. Functional Requirements (FR)

### FR1: Unified Model Interface (UMI)

- **FR1.1: Canonical Message Schema (CMS):** The tool must implement an internal Superset Schema to normalize different provider formats (OpenAI, Anthropic, Gemini) into a single `CanonicalMessage` object.
- **FR1.2: Runtime Transpilation:** The system must dynamically convert CMS history into provider-specific formats (e.g., merging consecutive messages for Anthropic or mapping tool calls to Gemini’s Protobuf structure).
- **FR1.3: LiteLLM Integration:** Use LiteLLM as the foundational adapter layer to support 100+ LLM providers via a single API interface.

### FR2: Capability Normalization & Polyfilling

- **FR2.1: Native/Prompted Mode Detection:** The tool must detect if a model supports native function calling.
- **FR2.2: ReAct Polyfill:** For models lacking native tool support, the system must automatically inject a ReAct (Thought/Action/Observation) system prompt and utilize Regex-based output parsing to execute tools.
- **FR2.3: Graceful Degradation:** Ensure workflows designed for high-tier models (GPT-4) can execute on smaller, local models (Llama-3) by switching orchestration logic to the polyfill.

### FR3: Orchestration & Topology Management

- **FR3.1: Configurable Topologies:** Support multiple agent interaction patterns via YAML configuration:
  - **Star:** Hierarchical Supervisor directing workers.
  - **Pipeline:** Sequential deterministic hand-offs.
  - **Mesh:** Decentralized A2A delegation.
- **FR3.2: Shared Blackboard Architecture:** Implement a centralized JSON/Redis state store for "Belief States," "Execution Traces," and "Artifacts" to decouple state from LLM context.

### FR4: Protocol Support (Connectivity)

- **FR4.1: MCP Client:** Implement a Model Context Protocol (MCP) client to discover and invoke tools from external MCP servers.
- **FR4.2: A2A Handshaking:** Support Agent-to-Agent protocol for discovering remote agents via "Agent Cards" (`.well-known/agent.json`).
- **FR4.3: UTCP Fallback:** Provide a Universal Tool Calling Protocol (UTCP) fallback for manual REST/CLI tool definitions.

### FR5: Context & Memory Management

- **FR5.1: Token Monitoring:** Calculate token load per request using model-specific tokenizers.
- **FR5.2: Pruning Strategies:** Automatically apply Sliding Window, Recursive Summarization, or Vector Offloading (RAG) when context limits are reached.

---

## 2. Non-Functional Requirements (NFR)

### NFR1: Portability & Modularity

- **NFR1.1: Agent Manifests:** Agents must be defined in standalone YAML files, specifying persona, required capabilities (e.g., `min_context_window`), and tool dependencies.
- **NFR1.2: Plug-and-Play:** Users must be able to "hot-swap" models within a workflow without modifying the underlying agent logic or tool definitions.

### NFR2: Security & Isolation

- **NFR2.1: Sandboxed Execution:** Tool execution (especially code execution) must occur in ephemeral Docker containers or WebAssembly (Wasm) runtimes.
- **NFR2.2: Human-in-the-Loop (HITL):** The CLI must support a "Gatekeeper" mode to intercept and require manual approval for high-risk tool actions (e.g., file system writes).

### NFR3: Reliability & Observability

- **NFR3.1: Reflexion Loops:** Implement automatic self-correction; if a model provides malformed JSON or a tool error, the tool must feed the error back to the model for a retry.
- **NFR3.2: OpenTelemetry Tracing:** Provide deep-link traces of the "thought process" and state transitions across multi-agent hops for debugging.

### NFR4: Performance

- **NFR4.1: Asynchronous Execution:** The orchestrator must support asynchronous messaging and event-driven updates to prevent blocking during long-running agent tasks.
