# Memory seed for C01

Prior decision: MemoryScheduler owns lifecycle scheduling, load-aware cadence, hierarchy movement, and scheduler persistence only. Policy arbitration stays in the orchestrator/policy layer. Semantic compression, world-model graph construction, and cognitive drive reports may be invoked by the scheduler, but their payloads must stay report-like and must not call CLI/MCP/API entrypoints directly.

Compatibility rule: CLI, MCP, and API must continue to route through TurnOrchestrator and must preserve existing TurnResult fields: injected_context, execution_trace, retrieved_memories, selected_memories, and context_budget.

Regression warning: do not reintroduce Reasonix-specific deployment or token UI behavior; this branch is Codex-only.
