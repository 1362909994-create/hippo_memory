from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryType(StrEnum):
    USER_PREFERENCE = "user_preference"
    PROJECT_CONTEXT = "project_context"
    DECISION = "decision"
    FAILURE = "failure"
    CONSTRAINT = "constraint"
    TECHNICAL_FACT = "technical_fact"
    TASK_STATE = "task_state"
    SOURCE_CHUNK = "source_chunk"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    OUTDATED = "outdated"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemoryVisibility(StrEnum):
    GLOBAL = "global"
    PROJECT = "project"
    PRIVATE = "private"
    SENSITIVE = "sensitive"


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"


@dataclass(slots=True)
class MemoryRecord:
    id: str
    content: str
    memory_type: str
    project: str | None = None
    summary: str | None = None
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: str | None = None
    source_ref: str | None = None
    confidence: float = 0.8
    importance: float = 0.5
    status: str = MemoryStatus.ACTIVE
    visibility: str = MemoryVisibility.PROJECT
    created_at: str | None = None
    updated_at: str | None = None
    last_used_at: str | None = None
    usage_count: int = 0
    ttl_days: int | None = None
    expires_at: str | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    memory_id: str
    content: str
    summary: str | None
    memory_type: str
    project: str | None
    importance: float
    confidence: float
    status: str
    visibility: str
    score: float
    matched_reason: str
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    score_details: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class WriteResult:
    memory_id: str
    created: bool
    duplicate: bool = False
