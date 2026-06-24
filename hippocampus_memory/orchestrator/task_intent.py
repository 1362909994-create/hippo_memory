from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TaskIntent = Literal[
    "architecture_refactor",
    "bug_fix",
    "debugging",
    "code_understanding",
    "general_query",
]


@dataclass(frozen=True, slots=True)
class TaskIntentDecision:
    intent: TaskIntent
    confidence: float
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "signals": self.signals,
        }


def classify_task_intent(text: str) -> TaskIntentDecision:
    lowered = f" {text.casefold()} "
    scores: dict[TaskIntent, float] = {
        "architecture_refactor": 0.0,
        "bug_fix": 0.0,
        "debugging": 0.0,
        "code_understanding": 0.0,
        "general_query": 0.1,
    }
    signals: dict[TaskIntent, list[str]] = {intent: [] for intent in scores}

    _score_keywords(
        lowered,
        scores,
        signals,
        "architecture_refactor",
        {
            "architecture": 0.35,
            "architectural": 0.35,
            "refactor": 0.35,
            "decouple": 0.35,
            "coupling": 0.3,
            "separation": 0.25,
            "orchestrator": 0.25,
            "scheduler": 0.25,
            "policy": 0.25,
            "semantic": 0.2,
            "world model": 0.2,
            "runtime": 0.15,
            "module": 0.15,
        },
    )
    _score_keywords(
        lowered,
        scores,
        signals,
        "bug_fix",
        {
            "bug": 0.35,
            "fix": 0.3,
            "failing": 0.3,
            "regression": 0.3,
            "broken": 0.25,
            "crash": 0.25,
            "error": 0.2,
            "leak": 0.3,
            "leaking": 0.3,
            "avoid leaking": 0.35,
            "private": 0.22,
            "sensitive": 0.22,
            "security": 0.25,
        },
    )
    _score_keywords(
        lowered,
        scores,
        signals,
        "debugging",
        {
            "debug": 0.35,
            "trace": 0.25,
            "investigate": 0.3,
            "why": 0.2,
            "root cause": 0.3,
            "diagnose": 0.3,
            "log": 0.15,
        },
    )
    _score_keywords(
        lowered,
        scores,
        signals,
        "code_understanding",
        {
            "explain": 0.35,
            "understand": 0.35,
            "how does": 0.3,
            "overview": 0.25,
            "code map": 0.25,
            "semantic graph": 0.25,
            "world model": 0.2,
        },
    )

    if scores["architecture_refactor"] > 0.0 and "refactor" in lowered:
        scores["architecture_refactor"] += 0.2
    if scores["bug_fix"] > 0.0 and scores["debugging"] > 0.0:
        if "fix" in lowered or "bug" in lowered:
            scores["bug_fix"] += 0.1
        else:
            scores["debugging"] += 0.1

    intent = max(scores, key=scores.get)
    if scores[intent] <= 0.1:
        intent = "general_query"
    confidence = min(1.0, max(0.2, scores[intent]))
    return TaskIntentDecision(intent=intent, confidence=confidence, signals=signals[intent])


def _score_keywords(
    text: str,
    scores: dict[TaskIntent, float],
    signals: dict[TaskIntent, list[str]],
    intent: TaskIntent,
    weights: dict[str, float],
) -> None:
    for keyword, weight in weights.items():
        if keyword in text:
            scores[intent] += weight
            signals[intent].append(keyword)
