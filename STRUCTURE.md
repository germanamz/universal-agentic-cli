# Project Structure: Universal Agentic CLI (UAC)

## Overview

The UAC is organized into a modular monorepo-style architecture. The primary design goal is to ensure that the **Orchestration** logic is entirely decoupled from specific **Provider APIs** and **Tool Implementations**. This allows for "hot-swapping" models and "plug-and-play" tool integration via MCP and A2A protocols.

---

## Directory Tree

```text
uac-root/
├── agents/                 # YAML-based Agent Manifests & Persona definitions
├── bin/                    # CLI Entrypoints (UAC executable)
├── configs/                # Global configurations (Provider keys, default topologies)
├── core/                   # The "Substrate" - Logic that powers the agents
│   ├── blackboard/         # Shared State Management (Blackboard Architecture)
│   ├── context/            # Token logistics, pruning, and summarization
│   ├── interface/          # Unified Model Interface (UMI) & Transpilation
│   ├── orchestration/      # Topology Engines (Star, Pipeline, Mesh)
│   └── polyfills/          # Cognitive Polyfilling (ReAct & Regex Parsing)
├── protocols/              # Standards-based connectivity
│   ├── a2a/                # Agent-to-Agent handshaking & Discovery
│   ├── mcp/                # Model Context Protocol Client implementation
│   └── utcp/               # Universal Tool Calling Protocol fallbacks
├── runtime/                # Security & Execution Environments
│   ├── gatekeeper/         # Human-in-the-Loop (HITL) approval logic`
│   └── sandbox/            # Docker and Wasm isolation layers
├── sdk/                    # Internal library for creating custom UAC tools/agents
└── utils/                  # Observability (OpenTelemetry) and Reflexion logic
```

## Detailed Folder Roles

### 1. `/agents`

Contains the Agent Manifests (.yaml). This is where the user defines an agent's "Identity."

Role: Defines the persona, system prompt templates, required capabilities (min_context), and specific tool dependencies for a particular agent role.

### 2. `/core/interface` (UMI)

The Unified Model Interface. This is the most critical layer for model-agnosticism.

Role: Implements the CanonicalMessageSchema (CMS). It uses LiteLLM to handle raw API communication but performs "Runtime Transpilation" to ensure history and roles are formatted correctly for Anthropic vs. OpenAI vs. Gemini.

### 3. `/core/polyfills`

Handles Cognitive Polyfilling.

Role: Detects if a model lacks native tool use. If so, it injects the ReAct (Thought/Action/Observation) framework into the system prompt and provides the Regex parsers to extract tool calls from raw text.

### 4. `/core/blackboard`

The Source of Truth.

Role: Implements the Shared Blackboard. It manages "Belief States" and "Execution Traces" in a JSON/Redis store, ensuring state exists independently of the LLM's ephemeral context window.

### 5. `/core/orchestration`

The Topology Engine.

Role: Manages the "Control Flow." It reads the topology configuration (Star, Pipeline, or Mesh) and determines which agent is activated next based on the state of the Blackboard or Supervisor commands.

### 6. `/protocols`

The Connective Tissue.

Role:

- mcp/: Acts as an MCP Client to fetch tools from external servers.

- a2a/: Manages decentralized delegation by looking for .well-known/agent.json files on remote servers.

### 7. `/runtime`

The Safety Layer.

Role:

- sandbox/: Manages the lifecycle of ephemeral Docker containers or Wasm instances where code execution occurs.

- gatekeeper/: Intercepts "side-effect" actions and pauses the orchestrator for manual human approval.

### 8. `/utils`

Observability & Reliability.

Role: Implements Reflexion Loops (self-healing on malformed outputs) and OpenTelemetry hooks to provide a visual trace of agent handoffs and tool executions.
