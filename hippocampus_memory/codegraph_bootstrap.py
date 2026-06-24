from __future__ import annotations

import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from hippocampus_memory.db import Database
from hippocampus_memory.file_filters import is_indexable_file, should_ignore_path
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.models import MemoryType
from hippocampus_memory.sensitive import is_sensitive_text
from hippocampus_memory.utils import normalize_text

BootstrapMode = Literal["preview", "queue", "write"]

BOOTSTRAP_SOURCE = "codegraph_bootstrap"
BOOTSTRAP_TAG = "codegraph_bootstrap"
STRUCTURAL_TAGS = {
    BOOTSTRAP_TAG,
    "architecture_map",
    "entrypoint_map",
    "module_boundary",
    "testing_strategy",
    "risk_note",
}
CONTEXT_OPERATIONS = {"auto_context", "memory_pack", "context_bundle", "context_callback"}
PROJECT_MARKERS = {
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "README.md",
    "src",
    "app",
    "lib",
}


@dataclass(frozen=True, slots=True)
class CodeGraphBootstrapSuggestion:
    recommended: bool
    requires_user_approval: bool
    project_state: str
    project: str
    root_path: str | None
    reason: str
    codegraph_cli_available: bool
    codegraph_mcp_hint: str
    tool: str = "codegraph_bootstrap"
    default_mode: BootstrapMode = "queue"
    next_step: str = (
        "Ask the user whether to run CodeGraph project bootstrap. If approved, call "
        "codegraph_context/codegraph_files first, then pass the compressed summary to "
        "hippo_memory_codegraph_bootstrap."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CodeGraphBootstrapItem:
    content: str
    memory_type: str
    confidence: float
    importance: float
    entities: list[str]
    tags: list[str]
    summary: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def codegraph_bootstrap_suggestion(
    db: Database,
    *,
    project: str | None,
    operation: str = "auto_context",
) -> CodeGraphBootstrapSuggestion | None:
    if not project or operation not in CONTEXT_OPERATIONS:
        return None
    record = db.get_project(project)
    root_path = str(record.get("root_path")) if record and record.get("root_path") else None
    root = Path(root_path) if root_path else None
    if root is None or not root.exists() or not root.is_dir():
        return None
    if _has_existing_structural_memory(db, project):
        return None

    state = _classify_project_root(root)
    if state["project_state"] != "existing_project_without_structure_memory":
        return None

    reason = (
        "This looks like an existing code project, but hippocampus-memory has no "
        "CodeGraph-derived structural memory for it yet. A confirmed bootstrap can "
        "store a compact architecture map before normal recall begins."
    )
    return CodeGraphBootstrapSuggestion(
        recommended=True,
        requires_user_approval=True,
        project_state=state["project_state"],
        project=project,
        root_path=str(root),
        reason=reason,
        codegraph_cli_available=shutil.which("codegraph") is not None,
        codegraph_mcp_hint=(
            "Use Codex CodeGraph MCP tools such as codegraph_context and codegraph_files "
            "when available; hippo-memory will only store the compressed summary you pass in."
        ),
    )


class CodeGraphBootstrapper:
    def __init__(self, db: Database) -> None:
        self.db = db

    def apply(
        self,
        *,
        project: str | None,
        codegraph_summary: str,
        mode: str = "queue",
        root_path: str | None = None,
        max_items: int = 6,
    ) -> dict[str, Any]:
        if not project:
            raise ValueError("project is required for CodeGraph bootstrap")
        normalized_mode = _normalize_mode(mode)
        items = build_codegraph_bootstrap_items(
            codegraph_summary,
            project=project,
            root_path=root_path,
            max_items=max_items,
        )
        queued_ids: list[str] = []
        written_ids: list[str] = []
        if normalized_mode == "queue":
            for item in items:
                queued_ids.append(
                    self.db.insert_candidate(
                        project=project,
                        content=item.content,
                        memory_type=item.memory_type,
                        confidence=item.confidence,
                        importance=item.importance,
                        source=BOOTSTRAP_SOURCE,
                        metadata=item.metadata,
                    )
                )
        elif normalized_mode == "write":
            writer = MemoryWriter(self.db)
            for item in items:
                result = writer.write(
                    content=item.content,
                    memory_type=item.memory_type,
                    project=project,
                    entities=item.entities,
                    tags=item.tags,
                    source=BOOTSTRAP_SOURCE,
                    confidence=item.confidence,
                    importance=item.importance,
                    visibility="project",
                    metadata=item.metadata,
                    summary=item.summary,
                )
                if result.created:
                    written_ids.append(result.memory_id)

        return {
            "project": project,
            "mode": normalized_mode,
            "source": BOOTSTRAP_SOURCE,
            "queued": len(queued_ids),
            "written": len(written_ids),
            "previewed": len(items) if normalized_mode == "preview" else 0,
            "candidate_ids": queued_ids,
            "memory_ids": written_ids,
            "items": [item.to_dict() for item in items],
            "text": _bootstrap_result_text(
                normalized_mode,
                len(items),
                len(queued_ids),
                len(written_ids),
            ),
        }


def build_codegraph_bootstrap_items(
    codegraph_summary: str,
    *,
    project: str,
    root_path: str | None = None,
    max_items: int = 6,
) -> list[CodeGraphBootstrapItem]:
    summary = normalize_text(codegraph_summary)
    if not summary:
        raise ValueError("codegraph_summary must not be empty")
    sections = _extract_sections(summary)
    if not sections:
        sections = [("architecture_map", "Architecture Map", summary)]

    items: list[CodeGraphBootstrapItem] = []
    for section_key, title, body in sections[: max(1, max_items)]:
        clean_body = _compact_section(body)
        if not clean_body or is_sensitive_text(clean_body):
            continue
        memory_type = _memory_type_for_section(section_key)
        content = f"CodeGraph bootstrap - {title}: {clean_body}"
        item = CodeGraphBootstrapItem(
            content=content,
            memory_type=memory_type,
            confidence=0.74 if memory_type == MemoryType.PROJECT_CONTEXT else 0.68,
            importance=0.78 if memory_type == MemoryType.PROJECT_CONTEXT else 0.7,
            entities=_unique(["CodeGraph", project, title]),
            tags=_unique([BOOTSTRAP_TAG, section_key]),
            summary=f"{title}: {clean_body[:180]}",
            metadata={
                "source": BOOTSTRAP_SOURCE,
                "project": project,
                "root_path": root_path,
                "section": section_key,
                "confirmed_by_user": True,
                "stores_source_code": False,
            },
        )
        items.append(item)
    if not items:
        raise ValueError("codegraph_summary did not contain safe structural content")
    return items


def _has_existing_structural_memory(db: Database, project: str) -> bool:
    for memory in db.list_memories(project=project, include_archived=True, limit=200):
        tags = set(memory.tags or [])
        if tags & STRUCTURAL_TAGS:
            return True
        if memory.source == BOOTSTRAP_SOURCE:
            return True
        content = memory.content.casefold()
        if "codegraph bootstrap" in content or "architecture map" in content:
            return True
    for candidate in db.list_candidates(project=project, status="pending", limit=100):
        content = str(candidate.get("content") or "").casefold()
        source = str(candidate.get("source") or "")
        if source == BOOTSTRAP_SOURCE or "codegraph bootstrap" in content:
            return True
    return False


def _classify_project_root(root: Path) -> dict[str, Any]:
    markers = sorted(marker for marker in PROJECT_MARKERS if (root / marker).exists())
    indexable_count = _count_indexable_files(root, stop_after=3)
    if markers or indexable_count >= 2:
        project_state = "existing_project_without_structure_memory"
    else:
        project_state = "empty_or_unrecognized_project"
    return {
        "project_state": project_state,
        "markers": markers,
        "indexable_file_count": indexable_count,
    }


def _count_indexable_files(root: Path, *, stop_after: int) -> int:
    count = 0
    stack = [root]
    scanned = 0
    while stack and scanned < 500:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            scanned += 1
            if child.is_dir():
                if not should_ignore_path(child):
                    stack.append(child)
                continue
            if child.is_file() and is_indexable_file(child):
                count += 1
                if count >= stop_after:
                    return count
    return count


def _extract_sections(summary: str) -> list[tuple[str, str, str]]:
    label_map = [
        (
            "architecture_map",
            "Architecture Map",
            r"(?:project\s+architecture|architecture|project\s+structure)",
        ),
        ("entrypoint_map", "Entrypoint Map", r"(?:entry\s*points?|entrypoints?|interfaces?)"),
        (
            "module_boundary",
            "Module Boundaries",
            r"(?:module\s+boundaries|boundaries|constraints?)",
        ),
        ("testing_strategy", "Testing Strategy", r"(?:tests?|testing|verification)"),
        ("risk_note", "Risk Notes", r"(?:risks?|failures?|regressions?)"),
    ]
    label_pattern = "|".join(f"(?P<{key}>{pattern})" for key, _, pattern in label_map)
    pattern = re.compile(rf"(?P<label>{label_pattern})\s*:\s*", re.IGNORECASE)
    matches = list(pattern.finditer(summary))
    sections: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        matched_key = next(key for key, _, _ in label_map if match.group(key))
        title = next(title for key, title, _ in label_map if key == matched_key)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(summary)
        body = summary[start:end].strip(" .;\n\t")
        if body:
            sections.append((matched_key, title, body))
    if sections:
        return sections

    lines = [line.strip(" -\t") for line in summary.splitlines() if line.strip()]
    if len(lines) > 1:
        return [("architecture_map", "Architecture Map", " ".join(lines[:8]))]
    return []


def _compact_section(text: str, *, max_chars: int = 900) -> str:
    compact = re.sub(r"\s+", " ", normalize_text(text)).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _memory_type_for_section(section_key: str) -> str:
    if section_key == "module_boundary":
        return MemoryType.CONSTRAINT.value
    if section_key in {"testing_strategy", "risk_note"}:
        return MemoryType.TECHNICAL_FACT.value
    return MemoryType.PROJECT_CONTEXT.value


def _normalize_mode(mode: str) -> BootstrapMode:
    normalized = mode.strip().casefold()
    if normalized not in {"preview", "queue", "write"}:
        raise ValueError("mode must be one of: preview, queue, write")
    return normalized  # type: ignore[return-value]


def _bootstrap_result_text(mode: BootstrapMode, items: int, queued: int, written: int) -> str:
    if mode == "preview":
        return f"CodeGraph bootstrap preview generated {items} structural memory item(s)."
    if mode == "queue":
        return f"CodeGraph bootstrap queued {queued} structural memory candidate(s)."
    return f"CodeGraph bootstrap wrote {written} structural memory item(s)."


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def attach_codegraph_bootstrap_suggestion(
    db: Database,
    *,
    project: str | None,
    operation: str,
    context_budget: dict[str, Any],
    recall_payload: dict[str, Any],
) -> None:
    suggestion = codegraph_bootstrap_suggestion(db, project=project, operation=operation)
    if suggestion is None:
        return
    payload = suggestion.to_dict()
    context_budget["codegraph_bootstrap"] = payload
    recall_payload["codegraph_bootstrap"] = payload
