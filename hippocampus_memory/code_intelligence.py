from __future__ import annotations

from typing import Any

from hippocampus_memory.db import Database
from hippocampus_memory.utils import tokenize


class CodeIntelligence:
    def __init__(self, db: Database) -> None:
        self.db = db

    def search_symbols(self, project: str, query: str | None = None, limit: int = 20) -> list[dict]:
        if query and len(query.split()) == 1:
            return self.db.list_code_symbols(project, query=query, limit=limit)
        symbols = self.db.list_code_symbols(project, limit=1000)
        if not query:
            return symbols[:limit]
        scored = [
            (_score_symbol(symbol, query, relevant_paths=set()), symbol)
            for symbol in symbols
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [symbol for score, symbol in scored[:limit] if score > 0]

    def references(self, project: str, symbol: str, limit: int = 50) -> list[dict]:
        return self.db.list_code_references(project, symbol=symbol, limit=limit)

    def diagnostics(
        self,
        project: str,
        *,
        relative_paths: list[str] | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        return self.db.list_code_diagnostics(
            project,
            relative_paths=relative_paths,
            severity=severity,
            limit=limit,
        )

    def diagnostic_lines(
        self,
        project: str,
        relevant_files: list[dict[str, Any]],
        limit: int = 6,
    ) -> list[str]:
        paths = [str(row["relative_path"]) for row in relevant_files]
        diagnostics = self.diagnostics(
            project,
            relative_paths=paths or None,
            limit=limit,
        )
        if not diagnostics and paths:
            diagnostics = self.diagnostics(project, limit=limit)
        if not diagnostics:
            return [
                "No stored LSP diagnostics for this project. "
                "Run hippo code-diagnostics --refresh."
            ]
        return [_format_diagnostic_line(diagnostic) for diagnostic in diagnostics[:limit]]

    def impact_lines(
        self,
        project: str,
        intent: str,
        relevant_files: list[dict[str, Any]],
        limit: int = 8,
    ) -> list[str]:
        relevant_paths = {str(row["relative_path"]) for row in relevant_files}
        symbols = self.db.list_code_symbols(project, limit=1000)
        scored = [
            (_score_symbol(symbol, intent, relevant_paths), symbol)
            for symbol in symbols
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [symbol for score, symbol in scored[:limit] if score > 0]
        if not selected:
            return ["No precise symbol impact available from current index."]

        lines: list[str] = []
        for symbol in selected:
            references = self.references(project, str(symbol["name"]), limit=20)
            call_refs = [ref for ref in references if ref["kind"] == "call"]
            lines.append(
                _format_symbol_line(
                    symbol,
                    reference_count=len(references),
                    call_count=len(call_refs),
                )
            )
            for reference in call_refs[:2]:
                lines.append(_format_reference_line(reference))
        return lines


def _score_symbol(symbol: dict[str, Any], query: str, relevant_paths: set[str]) -> float:
    query_tokens = set(tokenize(query))
    haystack = " ".join(
        [
            str(symbol.get("name") or ""),
            str(symbol.get("qualified_name") or ""),
            str(symbol.get("relative_path") or ""),
            str(symbol.get("kind") or ""),
            str(symbol.get("signature") or ""),
        ]
    )
    symbol_tokens = set(tokenize(haystack))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & symbol_tokens) / max(1, len(query_tokens))
    path_bonus = 0.4 if str(symbol.get("relative_path")) in relevant_paths else 0.0
    name_bonus = 0.3 if str(symbol.get("name") or "").casefold() in query.casefold() else 0.0
    kind_bonus = 0.1 if symbol.get("kind") in {"function", "async_function", "class"} else 0.0
    return min(1.0, overlap + path_bonus + name_bonus + kind_bonus)


def _format_symbol_line(symbol: dict[str, Any], *, reference_count: int, call_count: int) -> str:
    signature = symbol.get("signature") or symbol.get("qualified_name")
    return (
        f"{symbol['qualified_name']} [{symbol['kind']}] "
        f"{symbol['relative_path']}:{symbol['line']} "
        f"signature={signature}; refs={reference_count}; calls={call_count}"
    )


def _format_reference_line(reference: dict[str, Any]) -> str:
    container = f" in {reference['container']}" if reference.get("container") else ""
    context = f": {reference['context']}" if reference.get("context") else ""
    return (
        f"{reference['relative_path']}:{reference['line']} "
        f"{reference['kind']}s {reference['symbol']}{container}{context}"
    )


def _format_diagnostic_line(diagnostic: dict[str, Any]) -> str:
    rule = f" [{diagnostic['rule']}]" if diagnostic.get("rule") else ""
    return (
        f"{diagnostic['severity']} {diagnostic['relative_path']}:{diagnostic['line']}:"
        f"{diagnostic['column']} {diagnostic['message']}{rule}"
    )
