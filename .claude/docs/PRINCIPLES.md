# OpenEnv Design Principles

This document encodes the shared alignment between contributors on what OpenEnv optimizes for, what we trade off, and key decisions we've made.

## Core Principles (from RFC 000)

1. **Minimize lifecycle deltas**: Training → Evals → Production should use identical interfaces
2. **Minimize human-agent divergence**: Tools that work for humans should work for agents
3. **Be hands-on**: Provide ready-to-use implementations, not just specs
4. **Design for LLMs**: Optimize for context efficiency, in-distribution behavior

## What We Optimize For

- **Simple Gymnasium-style API** (`reset`, `step`, `state`) - familiar to RL practitioners
- **Container isolation** for reproducibility and security
- **Type safety** with generics and Pydantic across the wire
- **Production-readiness** from day one - training and production use same interfaces

## What We Trade Off

- **Flexibility for simplicity**: One canonical way to build environments
- **Performance for isolation**: Docker overhead is acceptable for reproducibility
- **Cutting-edge for stability**: FastAPI over experimental frameworks

## Key Decisions Made

These decisions are documented in RFCs and should not be changed without a new RFC:

| Decision | Rationale | RFC |
|----------|-----------|-----|
| **Rewards inside environment** | Domain knowledge encapsulated in env, not external | 002 |
| **Agents cannot reset** | Prevents learning that consequences are reversible | 001 |
| **MCP as universal standard** | All agent-environment tool interaction via MCP | 003 |
| **WebSocket for step loop** | Lower latency than HTTP per-step | 002 |
| **Two-interface model** | WebSocket for orchestration, MCP for agent tools | 001 |
| **One env = one trajectory** | Batching via environment stacking, not multiplexing | 004 |

**One env = one trajectory**: Environments do not support multiplexed trajectories. To generate batches, stack multiple environment instances. Helpers like `EnvPool` orchestrate batch collection across the stack. Multiplexing is left to future work.

## When to Revisit These Principles

- If a principle blocks a valid use case, open an RFC discussion
- If production experience contradicts a trade-off, document and propose changes
- Pre-1.0: Breaking changes acceptable with documentation
- Post-1.0: Semantic versioning strictly enforced
