from __future__ import annotations

from collections import Counter
from typing import Any

from hippocampus_memory.utils import loads_json, tokenize


def parse_json_list(value: str | None) -> list[str]:
    parsed = loads_json(value, [])
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def score_file_for_query(file_row: dict[str, Any], query: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    symbols = parse_json_list(file_row.get("symbols"))
    imports = parse_json_list(file_row.get("imports"))
    calls = parse_json_list(file_row.get("calls"))
    haystack = " ".join(
        [
            str(file_row.get("relative_path") or ""),
            str(file_row.get("language") or ""),
            str(file_row.get("summary") or ""),
            " ".join(symbols),
            " ".join(imports),
            " ".join(calls),
        ]
    )
    file_tokens = set(tokenize(haystack))
    overlap = query_tokens & file_tokens
    relative_path = str(file_row.get("relative_path"))
    path_bonus = 0.2 if any(token in relative_path for token in query_tokens) else 0
    return min(1.0, len(overlap) / max(1, len(query_tokens)) + path_bonus)


def top_languages(files: list[dict[str, Any]], limit: int = 5) -> list[tuple[str, int]]:
    counts = Counter(str(row.get("language") or "text") for row in files)
    return counts.most_common(limit)


def compact_items(items: list[str], limit: int = 5) -> str:
    if not items:
        return "none"
    return ", ".join(items[:limit])
