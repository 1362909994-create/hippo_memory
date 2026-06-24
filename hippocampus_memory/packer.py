from __future__ import annotations

from hippocampus_memory.conflict_detector import ConflictDetector
from hippocampus_memory.db import Database
from hippocampus_memory.models import MemoryRecord, SearchResult
from hippocampus_memory.retriever import Retriever
from hippocampus_memory.utils import estimate_tokens, normalize_text, text_similarity

LOW_CONFIDENCE_THRESHOLD = 0.7
NEAR_DUPLICATE_SIMILARITY = 0.92

_PROFILE_LAYER_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("orchestrator", ("orchestrator", "turnorchestrator")),
    ("scheduler", ("scheduler", "memoryscheduler")),
    ("policy", ("policy", "policyarbiter")),
    ("semantic", ("semantic", "semanticmemorymodel")),
    ("world_model", ("world model", "world-model", "world_model", "memoryworldmodel")),
    ("cognitive", ("cognitive", "cognitivedriveengine")),
)

_PROFILE_INTERFACE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CLI", ("cli",)),
    ("MCP", ("mcp",)),
    ("API", ("api",)),
)

_PROFILE_INTERFACE_NAMES = {"CLI", "MCP", "API"}

_PROFILE_BOUNDARY_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("entry_point", ("entry point", "single entry", "cli/mcp/api")),
    ("ownership", ("owns", "ownership", "owner", "responsibility")),
    ("decoupling", ("decouple", "decoupled", "separate", "separated")),
    ("communication_contract", ("communicate", "through reports", "report", "interface")),
    ("compatibility", ("compatibility", "compatible", "do not break", "without breaking")),
)


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
        preferred_memory_ids: list[str] | None = None,
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
        if preferred_memory_ids:
            results = _prioritize(results, preferred_memory_ids)
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
        _extend_architecture_runtime_profile(
            lines,
            self.db,
            [result.memory_id for result in deduped],
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


def _prioritize(
    results: list[SearchResult],
    preferred_memory_ids: list[str],
) -> list[SearchResult]:
    priority = {memory_id: index for index, memory_id in enumerate(preferred_memory_ids)}
    default_priority = len(priority)
    return [
        result
        for _, result in sorted(
            enumerate(results),
            key=lambda item: (priority.get(item[1].memory_id, default_priority), item[0]),
        )
    ]


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


def _extend_architecture_runtime_profile(
    lines: list[str],
    db: Database,
    included_ids: list[str],
) -> None:
    profile = _collect_architecture_runtime_profile(db, included_ids)
    if not profile:
        return
    lines.append("Architecture Runtime Profile:")
    for label, key in (
        ("Layers", "layers"),
        ("Interfaces", "interfaces"),
        ("Boundary signals", "boundary_signals"),
        ("Entities", "canonical_entities"),
    ):
        values = profile.get(key, [])
        if values:
            lines.append(f"- {label}: {', '.join(values)}")


def _collect_architecture_runtime_profile(
    db: Database,
    included_ids: list[str],
) -> dict[str, list[str]]:
    collected: dict[str, list[str]] = {
        "layers": [],
        "interfaces": [],
        "boundary_signals": [],
        "canonical_entities": [],
    }
    for memory_id in included_ids:
        memory = db.get_memory(memory_id)
        if memory is None:
            continue
        profile = memory.metadata.get("architecture_runtime_profile")
        if not isinstance(profile, dict):
            profile = _infer_architecture_runtime_profile(memory)
        if not profile:
            continue
        for key in collected:
            collected[key].extend(_string_items(profile.get(key)))
    return {key: _dedupe_strings(values) for key, values in collected.items() if values}


def _infer_architecture_runtime_profile(memory: MemoryRecord) -> dict[str, list[str]]:
    text = " ".join(
        [
            memory.content,
            memory.summary or "",
            " ".join(memory.tags),
            " ".join(memory.entities),
        ]
    ).casefold()
    layers = [name for name, terms in _PROFILE_LAYER_TERMS if _has_any(text, terms)]
    interfaces = [name for name, terms in _PROFILE_INTERFACE_TERMS if _has_any(text, terms)]
    boundary_signals = [name for name, terms in _PROFILE_BOUNDARY_TERMS if _has_any(text, terms)]
    canonical_entities = [
        entity for entity in memory.entities if entity not in _PROFILE_INTERFACE_NAMES
    ]
    if not (layers or interfaces or boundary_signals or canonical_entities):
        return {}
    return {
        "layers": layers,
        "interfaces": interfaces,
        "boundary_signals": boundary_signals,
        "canonical_entities": canonical_entities,
    }


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _string_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


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
