# HOOMAN Notes (Personal Reference)

## Parallelization Map

What can run at the same time, and what's blocking what.

### After Phase 0 (scaffolding) is done

Start **two tracks simultaneously**:

```
Track A: Phase 1 (UMI)        ← the critical path, everything downstream needs this
Track B: Phase 3 (Blackboard) ← no dependency on UMI, purely a state store
```

### After Phase 1 (UMI) is done

Three things unlock at once:

```
Track A: Phase 2 (Polyfill)   ← needs UMI's ModelClient
Track B: Phase 5 (Context)    ← needs UMI's CMS + ModelConfig for tokenizer selection
Track C: Phase 3 continues    ← if not already done
```

### After Phases 2 + 3 are done

Phase 4 (Orchestration) can start — it needs ModelClient, tool calling, and Blackboard.

### After Phase 4 (Orchestration) is done

Two more tracks open:

```
Track A: Phase 6 (Protocols)  ← MCP/A2A/UTCP, all three are independent of each other
Track B: Phase 7 (Runtime)    ← sandbox + gatekeeper hook into the orchestration loop
Track C: Phase 8 (Observability) ← instruments the orchestration loop, parallel with Phase 7
```

### Within-phase parallelism

- **Phase 4:** once 4b (primitives) lands, Pipeline / Star / Mesh topologies (4c/4d/4e) are independent
- **Phase 6:** MCP client, A2A discovery, and UTCP fallback share no code — work on all three at once

### Visual timeline (ideal with 2 people)

```
Week  Person A                        Person B
────  ──────────────────────────────  ──────────────────────────────
 1    Phase 0 (scaffolding)           Phase 0 (scaffolding)
 2    Phase 1a–1b (CMS, transpilers)  Phase 3a (Blackboard core)
 3    Phase 1c (LiteLLM)              Phase 3b–3c (Redis, slicing)
 4    Phase 2a–2b (polyfill)          Phase 5a–5b (context/tokens)
 5    Phase 4a–4b (manifests, prims)  Phase 5 wrap-up / Phase 4 help
 6    Phase 4c (Pipeline)             Phase 4d (Star)
 7    Phase 4e (Mesh)                 Phase 6a (MCP client)
 8    Phase 6b (A2A)                  Phase 6c (UTCP)
 9    Phase 7 (Runtime safety)        Phase 8 (Observability)
10    Phase 9 (CLI & SDK)             Phase 9 (CLI & SDK)
11    Phase 10 (E2E + docs)           Phase 10 (E2E + docs)
```

Solo? Follow the critical path: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7+8 → 9 → 10.

---

## Critical Path

The longest chain that determines minimum total time:

```
Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 6 → Phase 7 → Phase 9 → Phase 10
```

Everything else (3, 5, 8) can slot in around this without extending the timeline.

---

## What to Build First for a Demo

If you want to show something working as early as possible:

1. **Phase 0 + Phase 1** — gets you a working `ModelClient` that talks to any provider
2. **Phase 2** — now you can show the same tool-using workflow running on GPT-4 AND a local Ollama model
3. **Phase 4 (Pipeline only)** + a simple Blackboard — three agents in sequence, visible state passing

That's a compelling demo: "same workflow, any model, observable state" — and it only requires Phases 0–4 (pipeline subset).

---

## Interfaces to Nail Down Early

These are the contracts that everything else builds on. Get them right before writing implementations:

| Interface | Why it matters |
|-----------|---------------|
| `CanonicalMessage` + `ContentPart` | Every component passes these around. Wrong shape = cascading rewrites |
| `Transpiler` protocol | Adding a new provider should be one file. If the protocol is awkward you'll feel it 10x over |
| `ToolCallingStrategy` | Must cleanly abstract native vs ReAct. If polyfill leaks into orchestration code, the whole point is lost |
| `BlackboardBackend` protocol | In-memory for dev, Redis for prod. If the protocol is too narrow, Redis won't fit |
| `AgentNode.step()` signature | The orchestrator only talks to agents through this. It's the API boundary between "brain" and "body" |

---

## Decisions to Make Before Coding

- **Package manager:** Poetry or uv? (uv is faster, Poetry is more established)
- **Async framework:** plain asyncio or trio? (asyncio — wider ecosystem, LiteLLM already uses it)
- **Config format:** YAML everywhere or TOML for project config + YAML for manifests?
- **Testing strategy:** recorded fixtures (VCR-style) for LLM calls, or pure mocks? (VCR for integration, mocks for unit)
- **Minimum Python version:** 3.11+ (for TaskGroup, ExceptionGroup) or 3.10?

---

## Gotchas to Watch For

- **LiteLLM version pinning:** it moves fast and breaks things. Pin it. Test upgrades explicitly.
- **Anthropic message alternation:** their API rejects consecutive same-role messages. The transpiler MUST merge these. Easy to forget during testing since OpenAI doesn't care.
- **ReAct parsing fragility:** small models will break the format constantly. Build the parser to be forgiving (partial matches, whitespace tolerance, multiple formats). Test with intentionally messy output.
- **Blackboard deep-merge conflicts:** two agents writing to the same artifact key simultaneously. Decide on a conflict strategy early (last-write-wins? error? merge?).
- **Docker cold start:** first sandbox invocation pulls an image and is slow. Pre-pull in a setup step or use a warm pool.
- **Token counting mismatch:** tiktoken is exact for OpenAI but approximate for other providers. Don't over-optimize — leave a 10% buffer.

---

## Test Strategy Cheat Sheet

| Layer | Test type | What to mock |
|-------|-----------|-------------|
| CMS / Transpilers | Unit | Nothing — pure data transforms |
| ModelClient | Unit + Integration | LiteLLM (unit), real API (integration, gated) |
| ReAct Parser | Unit | Nothing — feed it raw strings |
| Blackboard | Unit | Nothing — in-memory by default |
| Orchestration | Unit | ModelClient (return canned responses) |
| MCP Client | Integration | Spin up a local MCP server fixture |
| Sandbox | Integration | Docker daemon must be available |
| CLI | E2E | Full stack with mocked ModelClient |
