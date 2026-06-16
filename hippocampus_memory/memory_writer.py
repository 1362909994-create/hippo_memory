from __future__ import annotations

from typing import Any

from hippocampus_memory.config import Settings
from hippocampus_memory.db import Database
from hippocampus_memory.embedding import EmbeddingBackend, create_embedding_backend
from hippocampus_memory.models import (
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    MemoryVisibility,
    WriteResult,
)
from hippocampus_memory.sensitive import is_sensitive_text
from hippocampus_memory.utils import (
    clamp,
    content_hash,
    iso_after_days,
    normalize_text,
    stable_id,
    utc_now,
)
from hippocampus_memory.vector_store import create_vector_store


class MemoryWriter:
    def __init__(
        self,
        db: Database,
        embedding_backend: EmbeddingBackend | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or db.settings
        self.embedding_backend = embedding_backend or create_embedding_backend(self.settings)
        self.vector_store = create_vector_store(db, self.settings)

    def write(
        self,
        content: str,
        memory_type: str,
        project: str | None = None,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        source_ref: str | None = None,
        confidence: float = 0.8,
        importance: float = 0.5,
        status: str = MemoryStatus.ACTIVE,
        visibility: str | None = None,
        ttl_days: int | None = None,
        metadata: dict[str, Any] | None = None,
        summary: str | None = None,
    ) -> WriteResult:
        normalized = normalize_text(content)
        if not normalized:
            raise ValueError("content must not be empty")
        memory_type = MemoryType(memory_type).value
        status = MemoryStatus(status).value
        if visibility is None and is_sensitive_text(normalized):
            visibility = MemoryVisibility.SENSITIVE
        visibility = MemoryVisibility(visibility or ("project" if project else "global")).value
        confidence = clamp(confidence)
        importance = clamp(importance)
        if confidence < 0 or confidence > 1 or importance < 0 or importance > 1:
            raise ValueError("confidence and importance must be in [0, 1]")

        digest = content_hash(normalized)
        duplicate = self.db.find_duplicate(digest, project, memory_type)
        if duplicate:
            return WriteResult(memory_id=duplicate, created=False, duplicate=True)

        now = utc_now()
        memory = MemoryRecord(
            id=stable_id("mem"),
            content=normalized,
            summary=summary,
            memory_type=memory_type,
            project=project,
            entities=entities or [],
            tags=tags or [],
            source=source,
            source_ref=source_ref,
            confidence=confidence,
            importance=importance,
            status=status,
            visibility=visibility,
            created_at=now,
            updated_at=now,
            ttl_days=ttl_days,
            expires_at=iso_after_days(ttl_days),
            content_hash=digest,
            metadata=metadata or {},
        )
        self.db.insert_memory(memory)
        if status != MemoryStatus.DELETED:
            self.vector_store.upsert(memory.id, self.embedding_backend.embed(memory.content))
        return WriteResult(memory_id=memory.id, created=True, duplicate=False)
