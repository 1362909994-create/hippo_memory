from __future__ import annotations

from hippocampus_memory.code_intelligence import CodeIntelligence
from hippocampus_memory.code_map import CodeMapBuilder
from hippocampus_memory.context_utils import parse_json_list
from hippocampus_memory.db import Database
from hippocampus_memory.retriever import Retriever
from hippocampus_memory.utils import estimate_tokens


class ChangePlanner:
    def __init__(self, db: Database, retriever: Retriever | None = None) -> None:
        self.db = db
        self.retriever = retriever or Retriever(db)
        self.code_map = CodeMapBuilder(db)
        self.code_intelligence = CodeIntelligence(db)

    def plan(self, intent: str, project: str, max_tokens: int = 1200) -> str:
        memories = self.retriever.search(intent, project=project, top_k=8, search_mode="hybrid")
        files = self.code_map.relevant_files(project, intent, limit=8)
        symbol_impact = self.code_intelligence.impact_lines(project, intent, files)
        diagnostic_impact = self.code_intelligence.diagnostic_lines(project, files)
        call_impact = infer_call_impact(files, self.db.list_files(project, limit=1000))
        risks = infer_risks(intent, [file_row["relative_path"] for file_row in files])
        tests = infer_tests(intent, files)

        lines = [
            "Code Impact Pack:",
            f"Project: {project}",
            f"Change intent: {intent.strip()}",
        ]
        lines.extend(_format_memories(memories))
        lines.extend(_format_files(files))
        lines.extend(_format_list("Precise symbol impact", symbol_impact))
        lines.extend(_format_list("Stored LSP diagnostics", diagnostic_impact))
        lines.extend(_format_list("Potential call impact", call_impact))
        lines.extend(_format_list("Risks / invariants", risks))
        lines.extend(
            _format_list(
                "Suggested minimal change",
                minimal_change_guidance(intent, files),
            )
        )
        lines.extend(_format_list("Suggested tests", tests))
        return _trim(lines, max_tokens)


def infer_risks(intent: str, paths: list[str]) -> list[str]:
    text = " ".join([intent, *paths]).casefold()
    risks = [
        "Keep the change scoped; prefer existing module boundaries and APIs.",
        "Do not widen Memory Pack output unless the task explicitly requires it.",
    ]
    if any(term in text for term in ["search", "retriev", "rank", "fts", "semantic"]):
        risks.append(
            "Search changes must preserve project, deleted, private and sensitive filters."
        )
        risks.append(
            "Keyword search must still work when semantic/vector backends are unavailable."
        )
    if any(term in text for term in ["pack", "context", "memory pack"]):
        risks.append(
            "Packs must stay concise and must not include sensitive/private memories by default."
        )
    if any(term in text for term in ["db", "schema", "sqlite", "migration"]):
        risks.append("Schema changes need backward-compatible initialization and tests.")
    if any(term in text for term in ["api", "endpoint", "fastapi", "response"]):
        risks.append("API response shapes should remain stable or be versioned.")
    if any(term in text for term in ["cli", "command", "typer"]):
        risks.append(
            "CLI changes should keep existing command names and PowerShell examples working."
        )
    if any(term in text for term in ["index", "file", "code map", "symbol"]):
        risks.append("Indexing must keep ignoring generated, binary, large and dependency folders.")
    if any(term in text for term in ["forget", "delete", "sensitive"]):
        risks.append("Deletion must preserve the soft-delete vs hard-delete distinction.")
    return _dedupe(risks)


def infer_call_impact(relevant_files: list[dict], all_files: list[dict]) -> list[str]:
    if not relevant_files:
        return ["No indexed call impact available for this intent."]
    relevant_paths = {str(row["relative_path"]) for row in relevant_files}
    symbol_to_files: dict[str, list[str]] = {}
    for file_row in all_files:
        path = str(file_row["relative_path"])
        for symbol in parse_json_list(file_row.get("symbols")):
            symbol_to_files.setdefault(symbol, []).append(path)

    impacts: list[str] = []
    for file_row in relevant_files:
        source = str(file_row["relative_path"])
        for call in parse_json_list(file_row.get("calls")):
            for target in symbol_to_files.get(call, []):
                if target != source:
                    impacts.append(f"{source} calls {call} in {target}.")

    relevant_symbols: dict[str, str] = {}
    for file_row in relevant_files:
        path = str(file_row["relative_path"])
        for symbol in parse_json_list(file_row.get("symbols")):
            relevant_symbols[symbol] = path
    for file_row in all_files:
        source = str(file_row["relative_path"])
        if source in relevant_paths:
            continue
        for call in parse_json_list(file_row.get("calls")):
            target = relevant_symbols.get(call)
            if target:
                impacts.append(f"{source} may call {call} in {target}.")

    return _dedupe(impacts)[:8] or ["No cross-file call impact inferred from current index."]


def minimal_change_guidance(intent: str, files: list[dict]) -> list[str]:
    if files:
        primary = ", ".join(str(row["relative_path"]) for row in files[:3])
        guidance = [f"Start with the most relevant indexed files: {primary}."]
    else:
        guidance = ["Run project indexing first if code impact is unclear."]
    text = intent.casefold()
    if "search" in text or "rank" in text:
        guidance.append("Prefer changing retrieval/ranking logic before touching storage schema.")
    if "pack" in text:
        guidance.append(
            "Prefer changing pack composition rules before changing retriever output fields."
        )
    if "api" in text or "cli" in text:
        guidance.append(
            "Add the new behavior behind existing service classes, then expose API/CLI wrappers."
        )
    guidance.append("Add or update the narrowest tests that cover the changed behavior.")
    return _dedupe(guidance)


def infer_tests(intent: str, files: list[dict]) -> list[str]:
    text = " ".join([intent, *[str(row["relative_path"]) for row in files]]).casefold()
    tests = []
    mapping = [
        ("retriev search rank fts semantic", "tests/test_retriever.py"),
        ("pack context", "tests/test_packer.py"),
        ("index file symbol code-map code map", "tests/test_project_indexer.py"),
        ("conflict", "tests/test_conflict_detector.py"),
        ("cli command typer", "tests/test_cli.py"),
        ("db sqlite schema", "tests/test_db.py"),
        ("write writer memory", "tests/test_writer.py"),
    ]
    for terms, path in mapping:
        if any(term in text for term in terms.split()):
            tests.append(path)
    return _dedupe(tests) or ["Run the focused tests for the touched module, then full pytest."]


def _format_memories(memories) -> list[str]:
    lines = ["Relevant memory:"]
    if not memories:
        lines.append("1. No strong memory found for this change intent.")
        return lines
    for index, memory in enumerate(memories[:5], 1):
        lines.append(f"{index}. {memory.memory_type}: {memory.summary or memory.content}")
    return lines


def _format_files(files: list[dict]) -> list[str]:
    lines = ["Likely affected files:"]
    if not files:
        lines.append(
            "1. No indexed file match. Run hippo index-project for better impact analysis."
        )
        return lines
    for index, file_row in enumerate(files, 1):
        summary = file_row.get("summary") or "No summary."
        lines.append(f"{index}. {file_row['relative_path']} - {summary}")
    return lines


def _format_list(title: str, items: list[str]) -> list[str]:
    lines = [f"{title}:"]
    for index, item in enumerate(items, 1):
        lines.append(f"{index}. {item}")
    return lines


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _trim(lines: list[str], max_tokens: int) -> str:
    kept: list[str] = []
    for line in lines:
        candidate = "\n".join([*kept, line])
        if estimate_tokens(candidate) > max_tokens:
            break
        kept.append(line)
    return "\n".join(kept)
