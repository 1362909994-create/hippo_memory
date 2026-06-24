from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

from hippocampus_memory.consolidator import Consolidator
from hippocampus_memory.db import Database
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.models import MemoryRecord
from hippocampus_memory.orchestrator.memory_scheduler import (
    MemoryScheduler,
    SemanticCompressionEngine,
    SemanticMemoryModel,
)
from hippocampus_memory.orchestrator.turn_orchestrator import (
    DecisionPolicyEngine,
    TurnOrchestrator,
    TurnTraceEvent,
)


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": self.payload}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationEvent:
        return cls(kind=str(data["kind"]), payload=dict(data.get("payload", {})))


@dataclass(slots=True)
class SimulationStepResult:
    index: int
    event: SimulationEvent
    duration_ms: float
    output: dict[str, Any]
    error: str | None = None

    def normalized(self) -> dict[str, Any]:
        output = dict(self.output)
        output.pop("memory_id", None)
        output.pop("duration_ms", None)
        return {"kind": self.event.kind, "output": output, "error": self.error}


class SystemSimulationHarness:
    def __init__(
        self,
        db: Database,
        *,
        project: str = "system-sim",
        state_dir: str | Path | None = None,
    ) -> None:
        self.db = db
        self.project = project
        base = Path(state_dir) if state_dir is not None else Path(db.path).parent
        self.policy_engine = DecisionPolicyEngine(base / f"{project}.policy.json")
        self.scheduler = MemoryScheduler(db, state_path=base / f"{project}.scheduler.json")
        self.orchestrator = TurnOrchestrator(
            db,
            policy_engine=self.policy_engine,
            memory_scheduler=self.scheduler,
        )
        self.writer = MemoryWriter(db)
        self.steps: list[SimulationStepResult] = []

    def run_events(self, events: list[SimulationEvent]) -> list[SimulationStepResult]:
        return [self.run_event(index, event) for index, event in enumerate(events)]

    def run_event(self, index: int, event: SimulationEvent) -> SimulationStepResult:
        start = perf_counter()
        error: str | None = None
        try:
            output = self._dispatch(index, event)
        except Exception as exc:  # pragma: no cover - logged for failure trace assertions
            output = {}
            error = f"{type(exc).__name__}: {exc}"
        duration_ms = (perf_counter() - start) * 1000.0
        result = SimulationStepResult(index, event, duration_ms, output, error)
        self.steps.append(result)
        return result

    def _dispatch(self, index: int, event: SimulationEvent) -> dict[str, Any]:
        if event.kind == "write":
            return self._write(index, event.payload)
        if event.kind == "recall":
            return self._recall(event.payload)
        if event.kind == "auto_store":
            return self._auto_store(event.payload)
        if event.kind == "scheduler":
            return self._scheduler(event.payload)
        if event.kind == "consolidate":
            return self._consolidate()
        if event.kind == "compress":
            return self._compress()
        if event.kind == "policy_feedback":
            return self._policy_feedback(event.payload)
        if event.kind == "age":
            return self._age(event.payload)
        raise ValueError(f"unsupported simulation event: {event.kind}")

    def _write(self, index: int, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.writer.write(
            project=payload.get("project", self.project),
            memory_type=payload.get("memory_type", "technical_fact"),
            content=payload.get("content", f"Synthetic memory {index} about scheduler policy."),
            entities=list(payload.get("entities", ["scheduler policy"])),
            tags=list(payload.get("tags", ["simulation"])),
            confidence=float(payload.get("confidence", 0.75)),
            importance=float(payload.get("importance", 0.6)),
            ttl_days=payload.get("ttl_days"),
        )
        return {"memory_id": result.memory_id, "created": result.created, **self.snapshot()}

    def _recall(self, payload: dict[str, Any]) -> dict[str, Any]:
        turn = self.orchestrator.run_turn(
            str(payload.get("query", "continue scheduler policy work")),
            context={
                "project": payload.get("project", self.project),
                "writeback": False,
                "max_tokens": int(payload.get("max_tokens", 900)),
            },
            mode="preview",
        )
        return {
            "trace_path": [event.node_id for event in turn.trace],
            "selected_count": len(turn.turn_context.selected_memories),
            "retrieved_count": len(turn.turn_context.retrieved_memories),
            "injected_context_len": len(turn.injected_context),
            **self.snapshot(),
        }

    def _auto_store(self, payload: dict[str, Any]) -> dict[str, Any]:
        turn = self.orchestrator.run_turn(
            str(payload.get("text", "Decision: keep scheduler reports deterministic.")),
            context={
                "operation": "memory_auto_store",
                "project": payload.get("project", self.project),
                "store_mode": payload.get("store_mode", "auto"),
                "writeback": False,
            },
            mode=str(payload.get("mode", "write")),
        )
        return {
            "trace_path": [event.node_id for event in turn.trace],
            "memory_writeback": turn.memory_writeback,
            **self.snapshot(),
        }

    def _scheduler(self, payload: dict[str, Any]) -> dict[str, Any]:
        report = self.scheduler.run_cycle(
            project=payload.get("project", self.project),
            system_load=payload.get("system_load", {}),
        )
        return {
            "state_version": report.state_version,
            "action_types": [action.action_type for action in report.lifecycle_actions],
            "blocked_transitions": len(report.stability_report.get("blocked_transitions", [])),
            "semantic_keys": sorted(report.semantic_report.keys()),
            **self.snapshot(),
        }

    def _consolidate(self) -> dict[str, Any]:
        result = Consolidator(self.db).consolidate(project=self.project)
        return {"consolidation": result, **self.snapshot(include_archived=True)}

    def _compress(self) -> dict[str, Any]:
        memories = self.db.list_memories(project=self.project, include_archived=True, limit=1000)
        result = SemanticCompressionEngine().compress(memories, model=SemanticMemoryModel())
        return {
            "merged_meanings": len(result["merged_meanings"]),
            "semantic_redundancy": result["semantic_redundancy"],
            **self.snapshot(include_archived=True),
        }

    def _policy_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("signal", "successful_recall"))
        if kind == "fallback_usage":
            trace = [
                TurnTraceEvent(
                    "fallback_lightweight",
                    "fallback",
                    {},
                    {"fallback_reason": "simulation"},
                    "writeback",
                )
            ]
        elif kind == "skip_memory_correctness":
            trace = [TurnTraceEvent("skip_memory", "skip", {}, {}, "fallback_lightweight")]
        else:
            trace = [TurnTraceEvent("execute", "executed", {}, {"selected_count": 1}, "writeback")]
        signals = self.policy_engine.update_from_trace(self.project, trace)
        return {
            "signals": signals,
            "should_recall_weight": self.policy_engine.edge_weight(
                self.project, "should_recall", "recall"
            ),
            **self.snapshot(),
        }

    def _age(self, payload: dict[str, Any]) -> dict[str, Any]:
        memories = self.db.list_memories(project=self.project, include_archived=True, limit=1000)
        if not memories:
            return self.snapshot()
        target = memories[int(payload.get("index", 0)) % len(memories)]
        days = int(payload.get("days", 90))
        usage_count = int(payload.get("usage_count", target.usage_count))
        created_at = (datetime.now(UTC) - timedelta(days=days)).replace(microsecond=0).isoformat()
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET created_at = ?, updated_at = ?, last_used_at = NULL, usage_count = ?
                WHERE id = ?
                """,
                (created_at, created_at, usage_count, target.id),
            )
        return {"aged_content": target.content, "days": days, **self.snapshot()}

    def snapshot(self, *, include_archived: bool = False) -> dict[str, Any]:
        memories = self.db.list_memories(
            project=self.project,
            include_archived=include_archived,
            include_private=True,
            include_sensitive=True,
            limit=10000,
        )
        return memory_snapshot(memories)

    @staticmethod
    def generate_events(count: int, *, project: str = "system-sim") -> list[SimulationEvent]:
        events: list[SimulationEvent] = []
        for index in range(count):
            if index % 50 == 0:
                events.append(
                    SimulationEvent(
                        "write",
                        {
                            "project": project,
                            "memory_type": "decision",
                            "content": f"Decision: enable policy cache scenario {index}.",
                            "entities": ["policy cache"],
                            "importance": 0.85,
                            "confidence": 0.9,
                        },
                    )
                )
                events.append(
                    SimulationEvent(
                        "write",
                        {
                            "project": project,
                            "memory_type": "decision",
                            "content": f"Do not enable policy cache scenario {index}.",
                            "entities": ["policy cache"],
                            "importance": 0.7,
                            "confidence": 0.8,
                        },
                    )
                )
            elif index % 13 == 0:
                events.append(SimulationEvent("scheduler", {"project": project}))
            elif index % 11 == 0:
                events.append(SimulationEvent("compress", {"project": project}))
            elif index % 7 == 0:
                events.append(
                    SimulationEvent(
                        "recall", {"project": project, "query": "continue policy cache"}
                    )
                )
            elif index % 5 == 0:
                events.append(SimulationEvent("policy_feedback", {"signal": "fallback_usage"}))
            elif index % 3 == 0:
                events.append(
                    SimulationEvent(
                        "age",
                        {"project": project, "index": index, "days": 80, "usage_count": 0},
                    )
                )
            else:
                events.append(
                    SimulationEvent(
                        "write",
                        {
                            "project": project,
                            "memory_type": "technical_fact",
                            "content": f"Synthetic fact {index} supports deterministic replay.",
                            "entities": ["deterministic replay"],
                            "importance": 0.45 + ((index % 5) * 0.08),
                            "confidence": 0.55 + ((index % 4) * 0.07),
                        },
                    )
                )
        events.append(SimulationEvent("consolidate", {"project": project}))
        return events[:count]


def memory_snapshot(memories: list[MemoryRecord]) -> dict[str, Any]:
    status_counts = Counter(memory.status for memory in memories)
    type_counts = Counter(memory.memory_type for memory in memories)
    contents = sorted((memory.memory_type, memory.status, memory.content) for memory in memories)
    return {
        "memory_count": len(memories),
        "status_counts": dict(sorted(status_counts.items())),
        "type_counts": dict(sorted(type_counts.items())),
        "contents": contents,
    }
