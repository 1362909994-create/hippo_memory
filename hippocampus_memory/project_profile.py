from __future__ import annotations

from hippocampus_memory.context_utils import top_languages
from hippocampus_memory.db import Database
from hippocampus_memory.models import MemoryRecord
from hippocampus_memory.utils import normalize_text


class ProjectProfileBuilder:
    def __init__(self, db: Database) -> None:
        self.db = db

    def build(self, project: str, max_items: int = 5) -> str:
        project_row = self.db.get_project(project)
        files = self.db.list_files(project, limit=1000)
        symbols = self.db.list_code_symbols(project, limit=50)
        memories = self.db.list_memories(project=project, include_archived=True, limit=200)
        counts = self.db.memory_counts_by_type(project)
        conflicts = self.db.list_conflicts(project, limit=5)

        lines = [
            "Project Profile:",
            f"Project: {project}",
            f"Root path: {project_row.get('root_path') if project_row else 'unknown'}",
        ]
        if project_row and project_row.get("summary"):
            lines.append(f"Stored summary: {project_row['summary']}")
        _extend_memory_section(lines, "Goals / context", memories, ["project_context"], max_items)
        lines.extend(_inferred_project_understanding(files, symbols))
        lines.extend(_implementation_shape(files))
        lines.extend(_feature_inventory(memories, counts))
        _extend_memory_section(lines, "Current state", memories, ["task_state"], max_items)
        _extend_memory_section(lines, "Decisions", memories, ["decision"], max_items)
        _extend_memory_section(lines, "Constraints", memories, ["constraint"], max_items)
        _extend_memory_section(lines, "Known failures", memories, ["failure"], max_items)
        lines.extend(_risk_points(memories, files, conflicts))
        lines.extend(_unknowns(memories, files, conflicts))
        return "\n".join(lines)


def _extend_memory_section(
    lines: list[str],
    title: str,
    memories: list[MemoryRecord],
    memory_types: list[str],
    max_items: int,
) -> None:
    selected = [memory for memory in memories if memory.memory_type in memory_types]
    lines.append(f"{title}:")
    if not selected:
        lines.append("1. Not recorded yet.")
        return
    for index, memory in enumerate(selected[:max_items], 1):
        confidence = f" confidence={memory.confidence:.2f}" if memory.confidence < 0.7 else ""
        lines.append(f"{index}. {memory.summary or memory.content}{confidence}")


def _implementation_shape(files: list[dict]) -> list[str]:
    lines = ["Implementation shape:"]
    if not files:
        lines.append("1. No project files indexed yet.")
        return lines
    languages = ", ".join(f"{language}={count}" for language, count in top_languages(files))
    lines.append(f"1. Indexed files: {len(files)}. Languages: {languages}.")
    key_files = _key_files(files)
    for index, file_row in enumerate(key_files, 2):
        summary = file_row.get("summary") or "no summary"
        lines.append(f"{index}. {file_row['relative_path']}: {summary}")
    return lines


def _inferred_project_understanding(files: list[dict], symbols: list[dict]) -> list[str]:
    lines = ["Inferred project understanding:"]
    notes: list[str] = []
    readme = _first_matching_file(files, ("readme",))
    if readme and readme.get("summary"):
        notes.append(f"README/doc hint: {_short_note(str(readme['summary']))}")
    packages = _src_packages(files)
    if packages:
        notes.append(f"Likely source package roots: {', '.join(packages[:5])}.")
    tests = [file_row["relative_path"] for file_row in files if _is_test_path(file_row)]
    if tests:
        notes.append(f"Tests indexed: {', '.join(tests[:4])}.")
    examples = [file_row["relative_path"] for file_row in files if _is_example_path(file_row)]
    if examples:
        notes.append(f"Examples indexed: {', '.join(examples[:4])}.")
    key_symbols = _key_symbols(symbols)
    if key_symbols:
        notes.append(f"Key indexed symbols: {', '.join(key_symbols)}.")
    if not notes:
        lines.append("1. Not enough indexed files or symbols to infer a useful project shape.")
        return lines
    lines.append("1. The following is inferred from indexed files/symbols, not confirmed memory.")
    for index, note in enumerate(notes, 2):
        lines.append(f"{index}. {note}")
    return lines


def _feature_inventory(memories: list[MemoryRecord], counts: dict[str, int]) -> list[str]:
    lines = ["Feature inventory:"]
    if counts:
        count_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        lines.append(f"1. Memory inventory: {count_text}.")
    source_chunks = counts.get("source_chunk", 0)
    if source_chunks:
        lines.append(f"2. Code/document context is indexed through {source_chunks} source chunks.")
    has_pack_memory = any("pack" in memory.content.casefold() for memory in memories)
    if has_pack_memory:
        lines.append("3. Project contains explicit Memory Pack related context.")
    if len(lines) == 1:
        lines.append("1. No feature inventory memories recorded yet.")
    return lines


def _risk_points(
    memories: list[MemoryRecord],
    files: list[dict],
    conflicts: list[dict],
) -> list[str]:
    lines = ["Risk points:"]
    risks = [
        "Do not default-recall sensitive/private memory into packs.",
        "Deleted memories must stay excluded from search and packs.",
        "Memory Pack output should remain short and task-relevant.",
    ]
    if files:
        risks.append(
            "Project index stores paths, hashes, summaries, symbols and chunks; "
            "avoid copying raw projects."
        )
    if conflicts:
        risks.append(
            "Open conflict candidates exist; do not treat conflicting memories as confirmed facts."
        )
    low_confidence = [memory for memory in memories if memory.confidence < 0.7]
    if low_confidence:
        risks.append("Some memories are low-confidence and should be confirmed before acting.")
    for index, risk in enumerate(risks, 1):
        lines.append(f"{index}. {risk}")
    return lines


def _unknowns(
    memories: list[MemoryRecord],
    files: list[dict],
    conflicts: list[dict],
) -> list[str]:
    lines = ["Unknowns / open questions:"]
    unknowns = []
    if not any(memory.memory_type == "project_context" for memory in memories):
        unknowns.append("Project goal/context has not been explicitly recorded.")
    if not files:
        unknowns.append("Project has not been indexed, so code impact analysis is limited.")
    if conflicts:
        unknowns.append("Open conflicts need human resolution before becoming confirmed facts.")
    if not unknowns:
        unknowns.append(
            "No major unknowns recorded; keep this section updated as the project evolves."
        )
    for index, unknown in enumerate(unknowns, 1):
        lines.append(f"{index}. {unknown}")
    return lines


def _key_files(files: list[dict]) -> list[dict]:
    priority_names = ("README", "AGENTS", "pyproject", "api", "cli", "db", "models")

    def score(file_row: dict) -> tuple[int, str]:
        path = str(file_row.get("relative_path") or "").casefold()
        priority = 0 if any(name.casefold() in path for name in priority_names) else 1
        return priority, path

    return sorted(files, key=score)[:5]


def _first_matching_file(files: list[dict], names: tuple[str, ...]) -> dict | None:
    for file_row in files:
        path = str(file_row.get("relative_path") or "").casefold()
        stem = path.rsplit("/", 1)[-1]
        if any(name in stem for name in names):
            return file_row
    return None


def _src_packages(files: list[dict]) -> list[str]:
    packages: set[str] = set()
    for file_row in files:
        parts = str(file_row.get("relative_path") or "").replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0] == "src" and parts[1]:
            packages.add(parts[1])
    return sorted(packages)


def _is_test_path(file_row: dict) -> bool:
    path = str(file_row.get("relative_path") or "").casefold().replace("\\", "/")
    return path.startswith("tests/") or "/tests/" in path or path.endswith("_test.py")


def _is_example_path(file_row: dict) -> bool:
    path = str(file_row.get("relative_path") or "").casefold().replace("\\", "/")
    return path.startswith("examples/") or "/examples/" in path


def _key_symbols(symbols: list[dict], limit: int = 8) -> list[str]:
    ranked = sorted(symbols, key=lambda item: (_symbol_rank(item), str(item.get("name") or "")))
    names: list[str] = []
    seen: set[str] = set()
    for symbol in ranked:
        name = str(symbol.get("qualified_name") or symbol.get("name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= limit:
            break
    return names


def _symbol_rank(symbol: dict) -> tuple[int, str, int]:
    kind = str(symbol.get("kind") or "")
    kind_rank = 0 if kind == "class" else 1 if kind == "function" else 2
    return (
        kind_rank,
        str(symbol.get("relative_path") or ""),
        int(symbol.get("line") or 0),
    )


def _short_note(text: str, limit: int = 220) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
