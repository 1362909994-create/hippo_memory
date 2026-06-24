from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from hippocampus_memory.db import Database
from hippocampus_memory.orchestrator.memory_scheduler import (
    MemoryWorldModel,
    SchedulerReport,
    SemanticMemoryModel,
)
from hippocampus_memory.orchestrator.turn_orchestrator import TurnResult
from tests.utils.system_simulator import SimulationStepResult


@dataclass(slots=True)
class TraceRecord:
    kind: str
    payload: dict[str, Any]


@dataclass(slots=True)
class MemoryTraceLogger:
    records: list[TraceRecord] = field(default_factory=list)

    def record_turn(self, result: TurnResult, *, label: str = "turn") -> None:
        self.records.append(
            TraceRecord(
                label,
                {
                    "trace_path": [event.node_id for event in result.trace],
                    "decisions": [event.decision for event in result.trace],
                    "selected_count": len(result.turn_context.selected_memories),
                    "retrieved_count": len(result.turn_context.retrieved_memories),
                    "context_budget_keys": sorted(result.turn_context.context_budget),
                },
            )
        )

    def record_scheduler(self, report: SchedulerReport, *, label: str = "scheduler") -> None:
        self.records.append(
            TraceRecord(
                label,
                {
                    "state_version": report.state_version,
                    "action_types": [action.action_type for action in report.lifecycle_actions],
                    "global_objective": report.global_objective,
                    "semantic_report_keys": sorted(report.semantic_report),
                },
            )
        )

    def record_step(self, step: SimulationStepResult) -> None:
        self.records.append(
            TraceRecord(
                step.event.kind,
                {
                    "index": step.index,
                    "duration_ms": step.duration_ms,
                    "error": step.error,
                    "output": step.output,
                },
            )
        )

    def summary(self) -> dict[str, Any]:
        durations = [float(record.payload.get("duration_ms", 0.0)) for record in self.records]
        failures = self.failure_trace()
        return {
            "record_count": len(self.records),
            "kind_counts": dict(Counter(record.kind for record in self.records)),
            "performance_metrics": {
                "total_ms": sum(durations),
                "max_ms": max(durations, default=0.0),
                "avg_ms": sum(durations) / max(1, len(durations)),
            },
            "failure_count": len(failures),
        }

    def failure_trace(self) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for record in self.records:
            if record.payload.get("error"):
                failures.append({"kind": record.kind, **record.payload})
        return failures

    def memory_graph_snapshot(self, db: Database, *, project: str) -> dict[str, Any]:
        memories = db.list_memories(project=project, include_archived=True, limit=1000)
        profiles = {memory.id: SemanticMemoryModel().profile(memory) for memory in memories}
        graph = MemoryWorldModel().build(memories=memories, profiles=profiles)
        return {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "entity_count": len(graph["entities"]),
            "concept_count": len(graph["concepts"]),
            "memory_count": len(memories),
        }
