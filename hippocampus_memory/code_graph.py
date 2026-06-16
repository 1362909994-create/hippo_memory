from __future__ import annotations

from collections import defaultdict

from hippocampus_memory.context_utils import parse_json_list
from hippocampus_memory.db import Database


class CodeGraphBuilder:
    def __init__(self, db: Database) -> None:
        self.db = db

    def build(self, project: str, limit: int = 50) -> str:
        files = self.db.list_files(project, limit=1000)
        symbol_to_files: dict[str, list[str]] = defaultdict(list)
        file_calls: dict[str, list[str]] = {}
        for file_row in files:
            path = str(file_row["relative_path"])
            for symbol in parse_json_list(file_row.get("symbols")):
                symbol_to_files[symbol].append(path)
            file_calls[path] = parse_json_list(file_row.get("calls"))

        lines = ["Code Graph:", f"Project: {project}"]
        edges = []
        for source, calls in file_calls.items():
            for call in calls:
                for target in symbol_to_files.get(call, []):
                    if target != source:
                        edges.append((source, call, target))
        if not edges:
            lines.append("No cross-file call edges inferred from current index.")
            return "\n".join(lines)
        for index, (source, call, target) in enumerate(edges[:limit], 1):
            lines.append(f"{index}. {source} --calls {call}--> {target}")
        return "\n".join(lines)
