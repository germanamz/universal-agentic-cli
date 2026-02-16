## Project Domain Breakdown: Universal Agentic CLI (UAC)

### 1. Unified Model Interface (UMI) & Translation

This domain handles the "normalization problem." It is responsible for ensuring the orchestrator can talk to any LLM regardless of provider-specific API quirks.

- **Key Components:** Canonical Message Schema (CMS), Runtime Transpilers (OpenAI/Anthropic/Gemini mapping), and LiteLLM integration.
- **Focus:** Protocol translation, role mapping (e.g., "assistant" to "model"), and message history restructuring.

### 2. Capability Polyfilling & Model Adaptation

This domain focuses on "Cognitive Polyfilling"—upgrading the functional abilities of weaker or local models to match high-tier models.

- **Key Components:** Native/Prompted mode detection, ReAct (Thought/Action/Observation) injection, and Regex-based output parsing.
- **Focus:** Enabling tool-use on models that do not natively support function calling.

### 3. Orchestration & Topology Management

The "brain" of the system that dictates how multiple agents interact and how the workflow progresses.

- **Key Components:** Topology engines (Star, Pipeline, Mesh/Network), Supervisor logic, and Agent Manifest (YAML) parsing.
- **Focus:** Control flow, delegation logic, and managing agent-to-agent (A2A) handshaking.

### 4. Connectivity & Protocol Standards

This domain manages the "Connective Tissue" of the ecosystem, allowing the CLI to interact with external tools and other agents.

- **Key Components:** MCP (Model Context Protocol) Client, A2A Handshaking (.well-known/agent.json), and UTCP (Universal Tool Calling Protocol) fallbacks.
- **Focus:** External tool discovery, standardized integration, and decentralized delegation.

### 5. State Management & Memory (Blackboard Architecture)

Responsible for maintaining a "Source of Truth" that is decoupled from the LLM’s ephemeral context window.

- **Key Components:** Shared Blackboard (JSON/Redis), Belief States, Execution Traces, and Artifact storage.
- **Focus:** Persistence, "time-travel" debugging, and cross-agent context sharing.

### 6. Context Optimization & Token Logistics

A specialized domain focused on the physical constraints of LLM processing (context windows and costs).

- **Key Components:** Universal Tokenizers, Sliding Window logic, Recursive Summarization, and RAG (Vector Offloading).
- **Focus:** Preventing context overflows and managing token efficiency across different model tiers.

### 7. Security, Sandboxing & Oversight

The safety layer ensures that autonomous actions (especially code execution) do not compromise the host system.

- **Key Components:** Docker/Wasm Execution Environments, Human-in-the-Loop (HITL) Gatekeeper, and "Side-effect" interception.
- **Focus:** Environment isolation, manual approval workflows, and risk mitigation.

### 8. Observability & Reliability (Reflexion)

The domain dedicated to monitoring and self-healing.

- **Key Components:** OpenTelemetry Tracing, Reflexion Loops (self-correction), and error-feedback loops.
- **Focus:** Debugging multi-agent hops, validating structured outputs, and automatic retries.
