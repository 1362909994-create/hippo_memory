from __future__ import annotations

from hippocampus_memory.config import Settings
from hippocampus_memory.db import Database
from hippocampus_memory.embedding import (
    EmbeddingBackend,
    HashEmbeddingBackend,
    create_embedding_backend,
)
from hippocampus_memory.models import MemoryRecord, SearchMode, SearchResult
from hippocampus_memory.ranker import rank_memory
from hippocampus_memory.utils import normalize_text, text_similarity, tokenize
from hippocampus_memory.vector_store import create_vector_store


class Retriever:
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

    def search(
        self,
        query: str,
        project: str | None = None,
        memory_types: list[str] | None = None,
        visibility_scope: list[str] | None = None,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
        include_archived: bool = False,
        include_private: bool = False,
        include_sensitive: bool = False,
        search_mode: str = SearchMode.HYBRID,
        dedupe_results: bool = True,
    ) -> list[SearchResult]:
        mode = SearchMode(search_mode)
        top_k = max(1, min(100, top_k))
        allowed = self.db.list_memories(
            project=project,
            include_archived=include_archived,
            include_private=include_private,
            include_sensitive=include_sensitive,
            limit=500,
        )
        allowed_by_id = {memory.id: memory for memory in allowed}
        if memory_types:
            allowed_by_id = {
                mid: memory
                for mid, memory in allowed_by_id.items()
                if memory.memory_type in memory_types
            }
        if visibility_scope:
            allowed_by_id = {
                mid: memory
                for mid, memory in allowed_by_id.items()
                if memory.visibility in visibility_scope
            }
        if entities:
            allowed_by_id = {
                mid: memory
                for mid, memory in allowed_by_id.items()
                if _contains_all(memory.entities, entities)
            }
        if tags:
            allowed_by_id = {
                mid: memory
                for mid, memory in allowed_by_id.items()
                if _contains_all(memory.tags, tags)
            }

        keyword_scores: dict[str, float] = {}
        if mode in {SearchMode.KEYWORD, SearchMode.HYBRID}:
            for memory, score in self.db.search_fts(
                query=query,
                project=project,
                include_archived=include_archived,
                include_private=include_private,
                include_sensitive=include_sensitive,
                limit=max(top_k * 4, 20),
            ):
                if memory.id in allowed_by_id:
                    keyword_scores[memory.id] = max(keyword_scores.get(memory.id, 0.0), score)
            for memory_id, score in _lexical_fallback_scores(query, allowed_by_id).items():
                keyword_scores[memory_id] = max(keyword_scores.get(memory_id, 0.0), score)

        semantic_scores: dict[str, float] = {}
        use_semantic = mode == SearchMode.SEMANTIC or not isinstance(
            self.embedding_backend,
            HashEmbeddingBackend,
        )
        if mode in {SearchMode.SEMANTIC, SearchMode.HYBRID} and use_semantic:
            query_vector = self.embedding_backend.embed(query)
            for memory_id, score in self.vector_store.search(
                query_vector,
                allowed_ids=list(allowed_by_id),
                limit=max(top_k * 4, 20),
            ):
                semantic_scores[memory_id] = score

        candidate_ids = set(keyword_scores) | set(semantic_scores)
        if mode == SearchMode.HYBRID:
            candidate_ids.update(_important_candidate_ids(allowed, project))
        if not candidate_ids and query.strip():
            candidate_ids.update(memory.id for memory in allowed[: max(top_k, 10)])

        results: list[SearchResult] = []
        for memory_id in candidate_ids:
            memory = allowed_by_id.get(memory_id)
            if not memory:
                continue
            score, reason = rank_memory(
                memory,
                keyword_score=keyword_scores.get(memory_id, 0.0),
                semantic_score=semantic_scores.get(memory_id, 0.0),
                project=project,
                include_private=include_private,
                include_sensitive=include_sensitive,
            )
            results.append(to_result(memory, score, reason))

        results.sort(key=lambda item: item.score, reverse=True)
        selected = _dedupe_results(results, top_k) if dedupe_results else results[:top_k]
        self.db.record_usage([result.memory_id for result in selected])
        return selected


def _important_candidate_ids(memories: list[MemoryRecord], project: str | None) -> set[str]:
    candidates: set[str] = set()
    for memory in memories:
        if memory.memory_type == "task_state":
            candidates.add(memory.id)
        if memory.importance >= 0.8:
            candidates.add(memory.id)
        if (
            project
            and memory.project == project
            and memory.memory_type in {"constraint", "decision", "failure", "project_context"}
            and memory.importance >= 0.65
        ):
            candidates.add(memory.id)
    return candidates


def _contains_all(values: list[str], requested: list[str]) -> bool:
    available = {value.casefold() for value in values}
    return all(value.casefold() in available for value in requested)


def _dedupe_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
    selected: list[SearchResult] = []
    seen_texts: list[str] = []
    for result in results:
        text = normalize_text(result.content).casefold()
        if _is_duplicate_result(text, seen_texts):
            continue
        selected.append(result)
        seen_texts.append(text)
        if len(selected) >= limit:
            break
    return selected


def _is_duplicate_result(text: str, seen_texts: list[str]) -> bool:
    if len(text) < 40:
        return text in seen_texts
    for existing in seen_texts:
        if len(existing) < 40:
            if text == existing:
                return True
            continue
        if text_similarity(text, existing) >= 0.92:
            return True
    return False


def _lexical_fallback_scores(query: str, memories: dict[str, MemoryRecord]) -> dict[str, float]:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return {}
    scores: dict[str, float] = {}
    for memory_id, memory in memories.items():
        haystack = " ".join(
            [
                memory.content,
                memory.summary or "",
                " ".join(memory.entities),
                " ".join(memory.tags),
                memory.project or "",
            ]
        )
        memory_tokens = set(tokenize(haystack))
        if not memory_tokens:
            continue
        overlap = query_tokens & memory_tokens
        if overlap:
            scores[memory_id] = min(1.0, len(overlap) / max(1, len(query_tokens)))
    return scores


def to_result(memory: MemoryRecord, score: float, reason: str) -> SearchResult:
    return SearchResult(
        memory_id=memory.id,
        content=memory.content,
        summary=memory.summary,
        memory_type=memory.memory_type,
        project=memory.project,
        importance=memory.importance,
        confidence=memory.confidence,
        status=memory.status,
        visibility=memory.visibility,
        score=score,
        matched_reason=reason,
        entities=memory.entities,
        tags=memory.tags,
    )
