from __future__ import annotations

from hippocampus_memory.context_utils import compact_items, parse_json_list, score_file_for_query
from hippocampus_memory.db import Database


class CodeMapBuilder:
    def __init__(self, db: Database) -> None:
        self.db = db

    def build(self, project: str, query: str | None = None, max_files: int = 12) -> str:
        files = self.db.list_files(project, limit=1000)
        if query:
            files = sorted(
                files,
                key=lambda row: score_file_for_query(row, query),
                reverse=True,
            )
            files = [row for row in files if score_file_for_query(row, query) > 0][:max_files]
        else:
            files = files[:max_files]

        lines = [
            "Code Map:",
            f"Project: {project}",
            f"Filter: {query or 'none'}",
        ]
        if not files:
            lines.append("No indexed files matched. Run hippo index-project first.")
            return "\n".join(lines)

        for index, file_row in enumerate(files, 1):
            symbols = parse_json_list(file_row.get("symbols"))
            imports = parse_json_list(file_row.get("imports"))
            calls = parse_json_list(file_row.get("calls"))
            language = file_row.get("language") or "text"
            lines.append(f"{index}. {file_row['relative_path']} [{language}]")
            lines.append(f"   Summary: {file_row.get('summary') or 'No summary.'}")
            lines.append(f"   Symbols: {compact_items(symbols)}")
            lines.append(f"   Imports: {compact_items(imports)}")
            lines.append(f"   Calls: {compact_items(calls)}")
        return "\n".join(lines)

    def relevant_files(self, project: str, query: str, limit: int = 8) -> list[dict]:
        files = self.db.list_files(project, limit=1000)
        scored = [
            (score_file_for_query(file_row, query), file_row)
            for file_row in files
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [file_row for score, file_row in scored[:limit] if score > 0]
