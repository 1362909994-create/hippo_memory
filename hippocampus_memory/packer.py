from __future__ import annotations

from hippocampus_memory.conflict_detector import ConflictDetector
from hippocampus_memory.db import Database
from hippocampus_memory.models import SearchResult
from hippocampus_memory.retriever import Retriever
from hippocampus_memory.utils import estimate_tokens, normalize_text, text_similarity

LOW_CONFIDENCE_THRESHOLD = 0.7
NEAR_DUPLICATE_SIMILARITY = 0.92


class MemoryPacker:
    def __init__(self, db: Database, retriever: Retriever | None = None) -> None:
        self.db = db
        self.retriever = retriever or Retriever(db)
        self.seen_memory_ids: set[str] = set()
        self.last_included_memory_ids: list[str] = []

    def pack(
        self,
        query: str,
        project: str | None = None,
        min_tokens: int = 300,
        max_tokens: int = 1500,
        candidate_k: int = 30,
        source_chunk_limit: int = 2,
        compact: bool = False,
        exclude_memory_ids: list[str] | None = None,
        session_dedupe: bool = False,
    ) -> str:
        self.last_included_memory_ids = []
        if compact:
            candidate_k = min(candidate_k, 12)
            source_chunk_limit = min(source_chunk_limit, 1)
        source_chunk_limit = max(0, min(10, source_chunk_limit))
        results = self.retriever.search(
            query=query,
            project=project,
            top_k=candidate_k,
            include_sensitive=False,
            include_private=False,
            search_mode="hybrid",
        )
        excluded = set(exclude_memory_ids or [])
        if session_dedupe:
            excluded.update(self.seen_memory_ids)
        if excluded:
            results = [result for result in results if result.memory_id not in excluded]
        deduped = _dedupe(results)
        conflicts = ConflictDetector(self.db).detect_for_project(project)
        sections = _group(deduped)
        lines = _header(query, project, compact)
        if not deduped:
            lines.append("No strong memory found for this query.")
            return "\n".join(lines)

        included_ids: list[str] = []
        limits = _section_limits(compact)
        _extend_section(
            lines,
            "Constraints",
            sections["constraints"],
            max_items=limits["constraints"],
            included_ids=included_ids,
        )
        _extend_section(
            lines,
            "Decisions",
            sections["decisions"],
            max_items=limits["decisions"],
            included_ids=included_ids,
        )
        _extend_section(
            lines,
            "Current project state",
            sections["task_state"],
            max_items=limits["task_state"],
            included_ids=included_ids,
        )
        _extend_section(
            lines,
            "Confirmed facts",
            sections["facts"],
            max_items=limits["facts"],
            included_ids=included_ids,
        )
        _extend_section(
            lines,
            "Failed attempts / do-not-repeat",
            sections["failures"],
            max_items=limits["failures"],
            included_ids=included_ids,
        )
        _extend_section(
            lines,
            "Project / user context",
            sections["context"],
            max_items=limits["context"],
            included_ids=included_ids,
        )
        _extend_section(
            lines,
            "Other relevant memory",
            sections["relevant"],
            max_items=limits["relevant"],
            included_ids=included_ids,
        )
        if conflicts and not compact:
            lines.append("Possible conflicts:")
            for idx, conflict in enumerate(conflicts[:3], 1):
                lines.append(f"{idx}. {conflict['description']}")
        if not compact:
            lines.append("Open questions:")
            lines.append(
                "1. Confirm any low-confidence or conflicting memories before acting on them."
            )
            lines.append("Suggested next step:")
            lines.append(_suggest_next_step(sections, query))
        _extend_section(
            lines,
            "Source context",
            sections["source_chunks"],
            max_items=source_chunk_limit,
            included_ids=included_ids,
        )
        if session_dedupe:
            self.seen_memory_ids.update(included_ids)
        self.last_included_memory_ids = included_ids
        return _trim_to_token_limit(lines, min_tokens=min_tokens, max_tokens=max_tokens)


def _header(query: str, project: str | None, compact: bool) -> list[str]:
    if compact:
        return [
            "Memory Pack:",
            f"Intent: {query.strip()}",
        ]
    return [
        "Memory Pack:",
        f"Project: {project or 'global'}",
        f"User intent: {query.strip()}",
    ]


def _section_limits(compact: bool) -> dict[str, int]:
    if compact:
        return {
            "constraints": 2,
            "decisions": 1,
            "task_state": 1,
            "facts": 2,
            "failures": 1,
            "context": 1,
            "relevant": 1,
        }
    return {
        "constraints": 3,
        "decisions": 3,
        "task_state": 3,
        "facts": 3,
        "failures": 3,
        "context": 3,
        "relevant": 3,
    }


def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    seen_texts: list[str] = []
    output: list[SearchResult] = []
    for result in results:
        key = normalize_text(result.content).casefold()
        if key in seen:
            continue
        if _is_near_duplicate(key, seen_texts):
            continue
        seen.add(key)
        seen_texts.append(key)
        output.append(result)
    return output


def _is_near_duplicate(text: str, seen_texts: list[str]) -> bool:
    if len(text) < 40:
        return False
    for existing in seen_texts:
        if len(existing) < 40:
            continue
        if text_similarity(text, existing) >= NEAR_DUPLICATE_SIMILARITY:
            return True
    return False


def _group(results: list[SearchResult]) -> dict[str, list[SearchResult]]:
    groups = {
        "task_state": [],
        "facts": [],
        "constraints": [],
        "decisions": [],
        "failures": [],
        "context": [],
        "source_chunks": [],
        "relevant": [],
    }
    for result in results:
        if result.memory_type == "task_state":
            groups["task_state"].append(result)
        elif result.memory_type == "technical_fact":
            groups["facts"].append(result)
        elif result.memory_type == "constraint":
            groups["constraints"].append(result)
        elif result.memory_type == "decision":
            groups["decisions"].append(result)
        elif result.memory_type == "failure":
            groups["failures"].append(result)
        elif result.memory_type in {"project_context", "user_preference"}:
            groups["context"].append(result)
        elif result.memory_type == "source_chunk":
            groups["source_chunks"].append(result)
        else:
            groups["relevant"].append(result)
    return groups


def _extend_section(
    lines: list[str],
    title: str,
    results: list[SearchResult],
    max_items: int,
    included_ids: list[str],
) -> None:
    if not results or max_items <= 0:
        return
    lines.append(f"{title}:")
    for idx, result in enumerate(results[:max_items], 1):
        lines.append(f"{idx}. {_format_result(result)}")
        included_ids.append(result.memory_id)


def _format_result(result: SearchResult) -> str:
    text = result.summary or result.content
    if result.confidence < LOW_CONFIDENCE_THRESHOLD:
        return f"[low confidence {result.confidence:.2f}] {text}"
    return text


def _suggest_next_step(sections: dict[str, list[SearchResult]], query: str) -> str:
    if sections["task_state"]:
        return f"1. Continue from current task state: {sections['task_state'][0].content}"
    if sections["constraints"]:
        return f"1. Plan the next action while respecting: {sections['constraints'][0].content}"
    return f"1. Use the retrieved memories to answer: {query.strip()}"


def _trim_to_token_limit(lines: list[str], min_tokens: int, max_tokens: int) -> str:
    del min_tokens
    kept: list[str] = []
    for line in lines:
        candidate = "\n".join([*kept, line])
        if estimate_tokens(candidate) > max_tokens:
            break
        kept.append(line)
    return "\n".join(kept)
