from __future__ import annotations

from datetime import UTC, datetime

from hippocampus_memory.models import MemoryRecord, MemoryStatus, MemoryType, MemoryVisibility
from hippocampus_memory.utils import clamp


def rank_memory(
    memory: MemoryRecord,
    *,
    keyword_score: float = 0.0,
    semantic_score: float = 0.0,
    project: str | None = None,
    include_private: bool = False,
    include_sensitive: bool = False,
) -> tuple[float, str]:
    project_boost = 1.0 if project and memory.project == project else 0.0
    recency_score = _recency(memory.updated_at or memory.created_at)
    usage_score = min(1.0, memory.usage_count / 10.0)
    type_boost = _type_boost(memory.memory_type)
    status_penalty = {
        MemoryStatus.ACTIVE: 0.0,
        MemoryStatus.OUTDATED: 0.3,
        MemoryStatus.ARCHIVED: 0.5,
        MemoryStatus.DELETED: 99.0,
    }.get(memory.status, 0.0)
    visibility_penalty = _visibility_penalty(
        memory,
        project=project,
        include_private=include_private,
        include_sensitive=include_sensitive,
    )

    score = (
        0.35 * semantic_score
        + 0.25 * keyword_score
        + 0.15 * project_boost
        + 0.10 * memory.importance
        + 0.05 * memory.confidence
        + 0.05 * recency_score
        + 0.05 * usage_score
        + type_boost
        - status_penalty
        - visibility_penalty
    )
    reasons = []
    if keyword_score:
        reasons.append(f"keyword={keyword_score:.2f}")
    if semantic_score:
        reasons.append(f"semantic={semantic_score:.2f}")
    if project_boost:
        reasons.append("project_match")
    if memory.importance >= 0.75:
        reasons.append("high_importance")
    if type_boost > 0:
        reasons.append(f"type={memory.memory_type}")
    return clamp(score, 0.0, 1.0), ", ".join(reasons) or "ranked_candidate"


def _type_boost(memory_type: str) -> float:
    return {
        MemoryType.CONSTRAINT.value: 0.12,
        MemoryType.DECISION.value: 0.10,
        MemoryType.FAILURE.value: 0.09,
        MemoryType.TASK_STATE.value: 0.08,
        MemoryType.USER_PREFERENCE.value: 0.06,
        MemoryType.PROJECT_CONTEXT.value: 0.05,
        MemoryType.TECHNICAL_FACT.value: 0.03,
        MemoryType.SOURCE_CHUNK.value: -0.03,
    }.get(memory_type, 0.0)


def _visibility_penalty(
    memory: MemoryRecord,
    *,
    project: str | None,
    include_private: bool,
    include_sensitive: bool,
) -> float:
    if memory.visibility == MemoryVisibility.GLOBAL:
        return 0.0
    if memory.visibility == MemoryVisibility.PROJECT:
        return 0.0 if project and memory.project == project else 0.2
    if memory.visibility == MemoryVisibility.PRIVATE:
        return 0.0 if include_private else 0.4
    if memory.visibility == MemoryVisibility.SENSITIVE:
        return 0.0 if include_sensitive else 0.8
    return 0.0


def _recency(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
    except ValueError:
        return 0.0
    age_days = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400)
    return 1.0 / (1.0 + age_days / 30.0)
