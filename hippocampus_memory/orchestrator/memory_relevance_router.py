from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from hippocampus_memory.models import SearchResult
from hippocampus_memory.orchestrator.task_intent import TaskIntent, classify_task_intent


@dataclass(frozen=True, slots=True)
class MemoryRelevanceReport:
    detected_task_intent: str
    intent_confidence: float
    intent_signals: list[str]
    scores_before: dict[str, float]
    scores_after: dict[str, float]
    boosted_memories: list[str]
    suppressed_memories: list[str]
    adjustments: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_task_intent": self.detected_task_intent,
            "intent_confidence": self.intent_confidence,
            "intent_signals": self.intent_signals,
            "memory_relevance_scores_before": self.scores_before,
            "memory_relevance_scores_after": self.scores_after,
            "boosted_memories": self.boosted_memories,
            "suppressed_memories": self.suppressed_memories,
            "adjustments": self.adjustments,
        }


@dataclass(frozen=True, slots=True)
class MemoryRelevanceRoutingResult:
    memories: list[SearchResult]
    report: MemoryRelevanceReport


class MemoryRelevanceRouter:
    def rerank(
        self,
        task_input: str,
        memories: list[SearchResult],
    ) -> MemoryRelevanceRoutingResult:
        decision = classify_task_intent(task_input)
        scores_before = {memory.memory_id: memory.score for memory in memories}
        adjusted: list[SearchResult] = []
        boosted: list[str] = []
        suppressed: list[str] = []
        adjustments: dict[str, dict[str, Any]] = {}

        for memory in memories:
            delta, reasons = self._adjustment(memory, decision.intent)
            after = _clamp_score(memory.score + delta)
            if delta > 0:
                boosted.append(memory.memory_id)
            elif delta < 0:
                suppressed.append(memory.memory_id)
            adjusted.append(self._copy_with_score(memory, after, delta, reasons, decision.intent))
            adjustments[memory.memory_id] = {
                "before": memory.score,
                "after": after,
                "delta": delta,
                "reasons": reasons,
            }

        adjusted.sort(
            key=lambda memory: (memory.score, memory.importance, memory.confidence),
            reverse=True,
        )
        return MemoryRelevanceRoutingResult(
            memories=adjusted,
            report=MemoryRelevanceReport(
                detected_task_intent=decision.intent,
                intent_confidence=decision.confidence,
                intent_signals=decision.signals,
                scores_before=scores_before,
                scores_after={memory.memory_id: memory.score for memory in adjusted},
                boosted_memories=boosted,
                suppressed_memories=suppressed,
                adjustments=adjustments,
            ),
        )

    def _adjustment(self, memory: SearchResult, intent: TaskIntent) -> tuple[float, list[str]]:
        haystack = _memory_text(memory)
        delta = 0.0
        reasons: list[str] = []

        if intent == "architecture_refactor":
            if _has_any(
                haystack,
                (
                    "scheduler",
                    "memoryscheduler",
                    "policy",
                    "policyarbiter",
                    "orchestrator",
                    "semantic",
                    "world model",
                    "world_model",
                    "architecture",
                    "refactor",
                    "coupling",
                    "decouple",
                ),
            ):
                delta += 0.4
                reasons.append("architecture_layer_match")
            if memory.memory_type in {"decision", "constraint", "project_context"}:
                delta += 0.08
                reasons.append("architecture_memory_type")
            if _is_unrelated_history_noise(haystack):
                delta -= 0.55
                reasons.append("suppressed_unrelated_history_noise")
        elif intent == "bug_fix":
            if memory.memory_type in {"failure", "task_state"} or _has_any(
                haystack,
                ("error", "failed", "failure", "bug", "trace", "regression", "crash"),
            ):
                delta += 0.35
                reasons.append("bug_fix_failure_trace_match")
            if _is_architecture_runtime_memory(haystack) and not _has_any(
                haystack,
                ("error", "failed", "failure", "bug", "trace", "regression", "crash"),
            ):
                delta -= 0.55
                reasons.append("suppressed_architecture_runtime_for_bug_fix")
            if _is_unrelated_history_noise(haystack) and not _has_any(
                haystack,
                ("bug", "error", "trace"),
            ):
                delta -= 0.25
                reasons.append("suppressed_unrelated_history_noise")
        elif intent == "debugging":
            if _has_any(haystack, ("debug", "trace", "diagnostic", "log", "root cause", "error")):
                delta += 0.3
                reasons.append("debugging_trace_match")
        elif intent == "code_understanding":
            if _has_any(
                haystack,
                ("semantic", "world model", "world_model", "graph", "profile", "code map"),
            ):
                delta += 0.35
                reasons.append("understanding_semantic_world_match")
            if _is_unrelated_history_noise(haystack):
                delta -= 0.25
                reasons.append("suppressed_unrelated_history_noise")
        elif intent == "general_query":
            if _is_architecture_runtime_memory(haystack):
                delta -= 0.55
                reasons.append("suppressed_architecture_runtime_for_general_query")
            if _is_unrelated_history_noise(haystack):
                delta -= 0.55
                reasons.append("suppressed_unrelated_history_noise")

        return delta, reasons or ["neutral"]

    def _copy_with_score(
        self,
        memory: SearchResult,
        score: float,
        delta: float,
        reasons: list[str],
        intent: TaskIntent,
    ) -> SearchResult:
        score_details = dict(memory.score_details)
        score_details["task_relevance_delta"] = delta
        score_details["task_relevance_score"] = score
        reason_suffix = (
            f"; task_relevance intent={intent} delta={delta:+.2f} "
            f"reasons={','.join(reasons)}"
        )
        return replace(
            memory,
            score=score,
            matched_reason=f"{memory.matched_reason}{reason_suffix}",
            score_details=score_details,
        )


def _memory_text(memory: SearchResult) -> str:
    parts = [
        memory.content,
        memory.summary or "",
        memory.memory_type,
        " ".join(memory.tags),
        " ".join(memory.entities),
        memory.matched_reason,
    ]
    return " ".join(parts).casefold()


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _is_unrelated_history_noise(text: str) -> bool:
    return _has_any(
        text,
        (
            "prompt-only",
            "context_auto",
            "hippo_memory_context_auto",
            "token savings",
            "status bar",
            "status-bar",
            "token ui",
            "token-ui",
            "ui history",
        ),
    )


def _is_architecture_runtime_memory(text: str) -> bool:
    return _has_any(
        text,
        (
            "architecture_runtime_profile",
            "architecture runtime profile",
            "architecture runtime memory",
            "memoryscheduler",
            "turnorchestrator",
            "policyarbiter",
            "semanticmemorymodel",
            "memoryworldmodel",
            "cli/mcp/api",
            "entry point",
            "world_model",
            "world model",
            "architectural decision",
            "scheduler policy",
            "orchestrator semantic",
        ),
    )


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))
