from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MemoryWriteRequest(BaseModel):
    content: str
    memory_type: str = Field(alias="type")
    project: str | None = None
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    source_ref: str | None = None
    confidence: float = 0.8
    importance: float = 0.5
    status: str = "active"
    visibility: str | None = None
    ttl_days: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class MemoryWriteResponse(BaseModel):
    memory_id: str
    created: bool
    duplicate: bool = False


class MemorySearchRequest(BaseModel):
    query: str
    project: str | None = None
    memory_types: list[str] | None = None
    visibility_scope: list[str] | None = None
    entities: list[str] | None = None
    tags: list[str] | None = None
    top_k: int = 10
    include_archived: bool = False
    include_private: bool = False
    include_sensitive: bool = False
    search_mode: Literal["hybrid", "keyword", "semantic"] = "hybrid"
    dedupe_results: bool = True


class MemorySearchItem(BaseModel):
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
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class MemorySearchResponse(BaseModel):
    results: list[MemorySearchItem]
    injected_context: str | None = None
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_memories: list[MemorySearchItem] = Field(default_factory=list)
    selected_memories: list[MemorySearchItem] = Field(default_factory=list)
    context_budget: dict[str, Any] = Field(default_factory=dict)


class MemoryPackRequest(BaseModel):
    query: str
    project: str | None = None
    min_tokens: int = 300
    max_tokens: int = 1500
    source_chunk_limit: int = 2
    compact: bool = False
    exclude_memory_ids: list[str] | None = None
    session_dedupe: bool = False


class MemoryPackResponse(BaseModel):
    pack: str
    injected_context: str | None = None
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_memories: list[MemorySearchItem] = Field(default_factory=list)
    selected_memories: list[MemorySearchItem] = Field(default_factory=list)
    context_budget: dict[str, Any] = Field(default_factory=dict)


class TextPackResponse(BaseModel):
    text: str


class AutoStoreRequest(BaseModel):
    text: str
    project: str | None = None
    source: str = "auto_store"
    mode: Literal["auto", "write", "queue", "preview"] = "auto"
    max_candidates: int = 12
    allow_sensitive: bool = False
    dry_run: bool = False


class AutoContextRequest(BaseModel):
    intent: str
    project: str | None = None
    session_key: str = "default"
    max_tokens: int = 3500
    include_code_map: bool = True


class ConsolidateRequest(BaseModel):
    project: str | None = None


class ForgetRequest(BaseModel):
    memory_id: str | None = None
    project: str | None = None
    hard: bool = False


class ProjectIndexRequest(BaseModel):
    path: str
    project: str


class CodeMapRequest(BaseModel):
    project: str
    query: str | None = None
    max_files: int = 12


class CodeSymbolRequest(BaseModel):
    project: str
    query: str | None = None
    limit: int = 20


class CodeReferenceRequest(BaseModel):
    project: str
    symbol: str
    limit: int = 50


class CodeIntelligenceRequest(BaseModel):
    project: str
    intent: str
    limit: int = 8


class CodeDiagnosticsRequest(BaseModel):
    project: str
    path: str | None = None
    checker: str | None = None
    refresh: bool = False
    limit: int = 100


class ImpactRequest(BaseModel):
    project: str
    intent: str
    max_tokens: int = 1200


class CandidateAcceptRequest(BaseModel):
    candidate_id: str


class ConflictResolveRequest(BaseModel):
    conflict_id: str
    resolution: str | None = None
    status: str = "resolved"


class SessionSummarizeRequest(BaseModel):
    path: str | None = None
    text: str | None = None
    project: str | None = None
    write: bool = False
    confirm_write: bool = False
    use_llm: bool = False


class HealthResponse(BaseModel):
    status: str
    db_path: str
