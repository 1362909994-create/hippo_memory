from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from hippocampus_memory.db import Database
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.models import MemoryRecord, MemoryVisibility
from hippocampus_memory.sensitive import is_sensitive_text
from hippocampus_memory.utils import normalize_text, text_similarity

POLICY_VERSION = "auto-memory-v1"

WRITE_SCORE_THRESHOLD = 0.74
QUEUE_SCORE_THRESHOLD = 0.56
NEAR_EXISTING_MEMORY_SIMILARITY = 0.90


@dataclass(slots=True)
class MemoryAdmissionDecision:
    content: str
    memory_type: str
    confidence: float
    importance: float
    action: str
    reason: str
    visibility: str = MemoryVisibility.PROJECT.value
    ttl_days: int | None = None
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_memory_admission(
    text: str,
    *,
    project: str | None = None,
    max_candidates: int = 12,
    allow_sensitive: bool = False,
    write_score_threshold: float = WRITE_SCORE_THRESHOLD,
    queue_score_threshold: float = QUEUE_SCORE_THRESHOLD,
) -> list[MemoryAdmissionDecision]:
    del project
    decisions: list[MemoryAdmissionDecision] = []
    seen: list[str] = []
    for segment in _candidate_segments(text):
        normalized = normalize_text(segment)
        if not _is_worth_considering(normalized):
            continue
        if _near_existing(normalized, seen):
            continue
        seen.append(normalized)
        decision = _classify_segment(normalized)
        if decision is None:
            continue
        _enrich_architecture_runtime_memory(decision)
        if is_sensitive_text(normalized):
            decision.visibility = MemoryVisibility.SENSITIVE.value
            decision.reason += "; sensitive content detected"
            if not allow_sensitive and decision.action == "write":
                decision.action = "queue"
        if decision.score >= write_score_threshold and decision.action != "skip":
            if decision.visibility == MemoryVisibility.SENSITIVE.value and not allow_sensitive:
                decision.action = "queue"
            else:
                decision.action = "write"
        elif decision.score >= queue_score_threshold and decision.action != "skip":
            decision.action = "queue"
        else:
            decision.action = "skip"
        decisions.append(decision)

    decisions.sort(key=lambda item: (_action_rank(item.action), item.score), reverse=True)
    return decisions[: max(1, min(100, max_candidates))]


def auto_store_memories(
    db: Database,
    text: str,
    *,
    project: str | None = None,
    source: str = "auto_store",
    mode: str = "auto",
    max_candidates: int = 12,
    allow_sensitive: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    mode = _normalize_mode(mode)
    decisions = plan_memory_admission(
        text,
        project=project,
        max_candidates=max_candidates,
        allow_sensitive=allow_sensitive,
    )
    writer = MemoryWriter(db)
    existing_memories = db.list_memories(project=project, limit=500)
    items: list[dict[str, Any]] = []
    written = 0
    queued = 0
    skipped = 0
    duplicates = 0
    previewed = 0
    for decision in decisions:
        target_action = _target_action(decision, mode, allow_sensitive=allow_sensitive)
        metadata = _metadata_for_decision(decision)
        item = {
            "decision": decision.to_dict(),
            "outcome": "preview" if dry_run else target_action,
            "memory_id": None,
            "candidate_id": None,
            "duplicate_memory_id": None,
        }
        near_duplicate_id = _near_existing_memory_id(decision, existing_memories)
        if target_action in {"write", "queue"} and near_duplicate_id is not None:
            duplicates += 1
            item["outcome"] = "near_duplicate"
            item["duplicate_memory_id"] = near_duplicate_id
            items.append(item)
            continue
        if dry_run:
            previewed += 1
            items.append(item)
            continue
        if target_action == "preview":
            previewed += 1
            items.append(item)
            continue
        if target_action == "write":
            result = writer.write(
                content=decision.content,
                memory_type=decision.memory_type,
                project=project,
                entities=decision.entities,
                tags=decision.tags,
                source=source,
                confidence=decision.confidence,
                importance=decision.importance,
                visibility=decision.visibility,
                ttl_days=decision.ttl_days,
                metadata=metadata,
            )
            item["memory_id"] = result.memory_id
            if result.duplicate:
                duplicates += 1
                item["outcome"] = "duplicate"
            elif result.created:
                written += 1
        elif target_action == "queue":
            candidate_id = db.insert_candidate(
                project=project,
                content=decision.content,
                memory_type=decision.memory_type,
                confidence=decision.confidence,
                importance=decision.importance,
                source=source,
                metadata=metadata,
            )
            item["candidate_id"] = candidate_id
            queued += 1
        else:
            skipped += 1
        items.append(item)
    return {
        "policy_version": POLICY_VERSION,
        "project": project,
        "mode": mode,
        "dry_run": dry_run,
        "written": written,
        "queued": queued,
        "skipped": skipped,
        "duplicates": duplicates,
        "previewed": previewed,
        "items": items,
    }


def _classify_segment(text: str) -> MemoryAdmissionDecision | None:
    lowered = text.casefold()
    if _has_any(lowered, _DO_NOT_STORE):
        return MemoryAdmissionDecision(
            content=text,
            memory_type="technical_fact",
            confidence=0.2,
            importance=0.1,
            action="skip",
            reason="explicit do-not-store language",
            score=0.15,
        )
    for rule in _RULES:
        if _has_any(lowered, rule["terms"]):
            confidence = float(rule["confidence"])
            importance = float(rule["importance"])
            score = _score(text, confidence, importance, rule["memory_type"])
            return MemoryAdmissionDecision(
                content=text,
                memory_type=str(rule["memory_type"]),
                confidence=confidence,
                importance=importance,
                action="queue",
                reason=str(rule["reason"]),
                visibility=MemoryVisibility.PROJECT.value,
                ttl_days=rule.get("ttl_days"),
                entities=_extract_entities(text),
                tags=_tags_for(str(rule["memory_type"])),
                score=score,
            )
    if _looks_like_technical_fact(text):
        confidence = 0.68
        importance = 0.62
        return MemoryAdmissionDecision(
            content=text,
            memory_type="technical_fact",
            confidence=confidence,
            importance=importance,
            action="queue",
            reason="technical fact or implementation detail",
            visibility=MemoryVisibility.PROJECT.value,
            entities=_extract_entities(text),
            tags=_tags_for("technical_fact"),
            score=_score(text, confidence, importance, "technical_fact"),
        )
    return None


def _candidate_segments(text: str) -> list[str]:
    segments: list[str] = []
    for raw in text.splitlines():
        line = _clean_candidate_line(raw)
        if not line:
            continue
        if len(line) <= 280:
            segments.append(line)
            continue
        segments.extend(_split_sentences(line))
    return segments


def _clean_candidate_line(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line.strip())
    line = re.sub(
        r"^(?:user|assistant|ai|codex|system|用户|助手)\s*[:：]\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _split_sentences(line: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？；;])\s+", line)
    output: list[str] = []
    for part in parts:
        cleaned = normalize_text(part)
        if len(cleaned) > 360:
            cleaned = cleaned[:360].rstrip()
        if cleaned:
            output.append(cleaned)
    return output


def _is_worth_considering(text: str) -> bool:
    if _meaningful_length(text) < 8:
        return False
    lowered = text.casefold()
    if lowered in _LOW_SIGNAL_EXACT:
        return False
    if lowered.startswith(("exit code:", "wall time:", "output:", "traceback ")):
        return False
    if re.fullmatch(r"[=\-_*`~]{4,}", text):
        return False
    return True


def _near_existing(text: str, seen: list[str]) -> bool:
    for existing in seen:
        if text_similarity(text, existing) >= 0.94:
            return True
    return False


def _looks_like_technical_fact(text: str) -> bool:
    lowered = text.casefold()
    if re.search(r"\b[\w.-]+\.(?:py|ts|tsx|js|json|toml|md|ps1)\b", lowered):
        return True
    if re.search(r"\b[A-Z]:\\", text):
        return True
    if re.search(r"\b(?:api|sqlite|fastapi|mcp|cli|schema|pytest|ruff|token)\b", lowered):
        return True
    if re.search(r"\b(?:api[_-]?key|secret|password|token)\s*[:=]", lowered):
        return True
    if lowered.startswith("technical fact:"):
        return True
    if re.search(r"\b(?:uses|implemented|configured|defaults?|stores?|indexes?)\b", lowered):
        return True
    if re.search(r"\b(?:used by|stored in|indexed by)\b", lowered):
        return True
    return any(term in text for term in ("实现", "接口", "函数", "数据库", "索引", "默认"))


def _score(text: str, confidence: float, importance: float, memory_type: str) -> float:
    score = 0.52 * confidence + 0.42 * importance + _type_bonus(memory_type)
    length = _meaningful_length(text)
    if length > 220:
        score -= 0.05
    if length < 16:
        score -= 0.04
    if re.search(r"\b(?:maybe|possibly|guess|可能|大概|也许)\b", text.casefold()):
        score -= 0.08
    return max(0.0, min(1.0, score))


def _type_bonus(memory_type: str) -> float:
    return {
        "constraint": 0.07,
        "decision": 0.06,
        "failure": 0.06,
        "task_state": 0.05,
        "user_preference": 0.05,
        "project_context": 0.03,
        "technical_fact": 0.02,
    }.get(memory_type, 0.0)


def _extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    entities.extend(match.strip("`'\"") for match in re.findall(r"`([^`]{2,80})`", text))
    entities.extend(re.findall(r"\b[A-Z][A-Za-z0-9_]{2,40}\b", text))
    entities.extend(re.findall(r"\b[\w.-]+\.(?:py|ts|tsx|js|json|toml|md|ps1)\b", text))
    entities.extend(re.findall(r"\b[A-Z]:\\[^\s,;]+", text))
    return _dedupe([entity[:80] for entity in entities])[:8]


def _tags_for(memory_type: str) -> list[str]:
    return _dedupe(["auto", memory_type])


def _enrich_architecture_runtime_memory(decision: MemoryAdmissionDecision) -> None:
    lowered = decision.content.casefold()
    if not _looks_like_architecture_runtime_memory(lowered):
        return
    tags = ["architecture", "memory_os"]
    entities: list[str] = []
    for tag, needles, canonical_entities in _ARCHITECTURE_RUNTIME_SIGNALS:
        if _has_any(lowered, needles):
            tags.append(tag)
            entities.extend(canonical_entities)
    decision.tags = _dedupe([*decision.tags, *tags])
    decision.entities = _dedupe([*decision.entities, *entities])[:12]
    if "architecture runtime" not in decision.reason:
        decision.reason += "; architecture runtime memory"
    decision.metadata["architecture_runtime_profile"] = _architecture_runtime_profile(
        lowered,
        decision,
    )
    decision.importance = max(decision.importance, 0.82)
    decision.score = max(
        decision.score,
        _score(decision.content, decision.confidence, decision.importance, decision.memory_type),
    )


def _looks_like_architecture_runtime_memory(lowered: str) -> bool:
    if not _has_any(lowered, _ARCHITECTURE_RUNTIME_TERMS):
        return False
    return _has_any(lowered, _ARCHITECTURE_DECISION_TERMS)


def _architecture_runtime_profile(
    lowered: str,
    decision: MemoryAdmissionDecision,
) -> dict[str, Any]:
    return {
        "layers": [tag for tag in decision.tags if tag in _ARCHITECTURE_LAYER_TAGS],
        "interfaces": [
            entity for entity in decision.entities if entity in _ARCHITECTURE_INTERFACE_ENTITIES
        ],
        "boundary_signals": _architecture_boundary_signals(lowered),
        "canonical_entities": [
            entity
            for entity in decision.entities
            if entity not in _ARCHITECTURE_INTERFACE_ENTITIES
        ],
    }


def _architecture_boundary_signals(lowered: str) -> list[str]:
    signals: list[str] = []
    for name, needles in _ARCHITECTURE_BOUNDARY_SIGNALS:
        if _has_any(lowered, needles):
            signals.append(name)
    return _dedupe(signals)


def _metadata_for_decision(decision: MemoryAdmissionDecision) -> dict[str, Any]:
    metadata = {
        "policy_version": POLICY_VERSION,
        "admission_reason": decision.reason,
        "admission_score": decision.score,
        "planned_action": decision.action,
        "visibility": decision.visibility,
        "entities": decision.entities,
        "tags": decision.tags,
        "ttl_days": decision.ttl_days,
    }
    metadata.update(decision.metadata)
    return metadata


def _target_action(
    decision: MemoryAdmissionDecision,
    mode: str,
    *,
    allow_sensitive: bool,
) -> str:
    if decision.action == "skip":
        return "skip"
    if mode == "preview":
        return "preview"
    if mode == "queue":
        return "queue"
    if mode == "write":
        if decision.visibility == MemoryVisibility.SENSITIVE.value and not allow_sensitive:
            return "queue"
        return "write"
    return decision.action


def _near_existing_memory_id(
    decision: MemoryAdmissionDecision,
    existing_memories: list[MemoryRecord],
) -> str | None:
    for memory in existing_memories:
        if memory.memory_type != decision.memory_type:
            continue
        if text_similarity(decision.content, memory.content) >= NEAR_EXISTING_MEMORY_SIMILARITY:
            return memory.id
    return None


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip().casefold()
    if normalized in {"auto", "write", "queue", "preview"}:
        return normalized
    raise ValueError("mode must be one of: auto, write, queue, preview")


def _action_rank(action: str) -> int:
    return {"write": 3, "queue": 2, "preview": 1, "skip": 0}.get(action, 0)


def _meaningful_length(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]", text))


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.casefold() in text for term in terms)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


_LOW_SIGNAL_EXACT = {
    "ok",
    "okay",
    "thanks",
    "thank you",
    "好的",
    "可以",
    "继续",
    "明白",
    "收到",
}

_DO_NOT_STORE = (
    "do not remember",
    "don't remember",
    "do not store",
    "不要记住",
    "不要保存",
    "别记",
)

_RULES: tuple[dict[str, Any], ...] = (
    {
        "memory_type": "user_preference",
        "confidence": 0.86,
        "importance": 0.82,
        "reason": "stable user preference",
        "terms": (
            "i prefer",
            "always use",
            "from now on",
            "remember that i",
            "以后都",
            "以后不要",
            "我希望",
            "我不想",
            "偏好",
        ),
    },
    {
        "memory_type": "constraint",
        "confidence": 0.86,
        "importance": 0.82,
        "reason": "explicit project or workflow constraint",
        "terms": (
            "must",
            "must not",
            "cannot",
            "never",
            "do not",
            "avoid",
            "required",
            "constraint",
            "必须",
            "不能",
            "不要",
            "不允许",
            "避免",
            "约束",
        ),
    },
    {
        "memory_type": "decision",
        "confidence": 0.82,
        "importance": 0.78,
        "reason": "explicit decision or chosen approach",
        "terms": (
            "decided",
            "decision",
            "we will use",
            "chosen",
            "adopt",
            "settled on",
            "决定",
            "选择",
            "采用",
            "结论",
            "方案是",
        ),
    },
    {
        "memory_type": "failure",
        "confidence": 0.8,
        "importance": 0.78,
        "reason": "failed attempt or do-not-repeat lesson",
        "ttl_days": 180,
        "terms": (
            "failed",
            "does not work",
            "did not work",
            "timeout",
            "permission denied",
            "error was",
            "失败",
            "不行",
            "无效",
            "报错",
            "超时",
            "权限失败",
        ),
    },
    {
        "memory_type": "task_state",
        "confidence": 0.76,
        "importance": 0.74,
        "reason": "current state or next step",
        "ttl_days": 45,
        "terms": (
            "current task",
            "current goal",
            "next step",
            "todo",
            "done:",
            "completed",
            "blocked",
            "continue from",
            "当前",
            "下一步",
            "已完成",
            "做到",
            "继续",
            "还差",
            "阻塞",
        ),
    },
    {
        "memory_type": "project_context",
        "confidence": 0.72,
        "importance": 0.68,
        "reason": "durable project context",
        "terms": (
            "project is",
            "goal is",
            "architecture",
            "workflow",
            "designed to",
            "项目是",
            "目标是",
            "架构",
            "工作流",
            "设计初衷",
        ),
    },
)

_ARCHITECTURE_RUNTIME_TERMS = (
    "architecture",
    "architectural",
    "orchestrator",
    "turnorchestrator",
    "memoryscheduler",
    "scheduler",
    "policy",
    "policyarbiter",
    "semantic",
    "world model",
    "world-model",
    "world_model",
    "cognitive",
    "cli/mcp/api",
    "mcp",
)

_ARCHITECTURE_DECISION_TERMS = (
    "decision",
    "constraint",
    "must",
    "must not",
    "should",
    "should not",
    "remain",
    "owns",
    "boundary",
    "boundaries",
    "decouple",
    "decoupled",
    "separate",
    "separated",
    "entry point",
    "single entry",
)

_ARCHITECTURE_RUNTIME_SIGNALS: tuple[tuple[str, tuple[str, ...], list[str]], ...] = (
    ("orchestrator", ("orchestrator", "turnorchestrator"), ["TurnOrchestrator"]),
    ("scheduler", ("scheduler", "memoryscheduler"), ["MemoryScheduler"]),
    ("policy", ("policy", "policyarbiter"), ["PolicyArbiter"]),
    ("semantic", ("semantic",), ["SemanticMemoryModel"]),
    ("world_model", ("world model", "world-model", "world_model"), ["MemoryWorldModel"]),
    ("cognitive", ("cognitive",), ["CognitiveDriveEngine"]),
    ("cli", ("cli",), ["CLI"]),
    ("mcp", ("mcp",), ["MCP"]),
    ("api", ("api",), ["API"]),
)

_ARCHITECTURE_LAYER_TAGS = {
    "orchestrator",
    "scheduler",
    "policy",
    "semantic",
    "world_model",
    "cognitive",
}

_ARCHITECTURE_INTERFACE_ENTITIES = {"CLI", "MCP", "API"}

_ARCHITECTURE_BOUNDARY_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("entry_point", ("entry point", "single entry", "cli/mcp/api")),
    ("ownership", ("owns", "ownership", "owner", "responsibility")),
    ("decoupling", ("decouple", "decoupled", "separate", "separated")),
    ("communication_contract", ("communicate", "through reports", "report", "interface")),
    ("compatibility", ("compatibility", "compatible", "do not break", "without breaking")),
)
