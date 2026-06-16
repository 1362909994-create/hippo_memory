from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hippocampus_memory.db import Database
from hippocampus_memory.packer import MemoryPacker
from hippocampus_memory.retriever import Retriever
from hippocampus_memory.utils import estimate_tokens


def evaluate_retrieval(db: Database, benchmark_path: str | Path) -> dict[str, Any]:
    cases = _load_cases(benchmark_path)
    retriever = Retriever(db)
    packer = MemoryPacker(db, retriever=retriever)
    total = len(cases)
    hits = 0
    details = []
    for case in cases:
        query = str(case["query"])
        project = case.get("project")
        mode = str(case.get("mode", "search"))
        expected = _lower_list(case.get("expected_contains", []))
        forbidden = _lower_list(case.get("forbidden_contains", []))
        max_tokens = case.get("max_tokens")
        if mode == "pack":
            text = packer.pack(
                query=query,
                project=project,
                max_tokens=int(max_tokens or 1500),
            )
            returned = []
        else:
            results = retriever.search(
                query=query,
                project=project,
                top_k=int(case.get("top_k", 10)),
            )
            text = "\n".join(result.content for result in results)
            returned = [result.memory_id for result in results]
        token_count = estimate_tokens(text)
        joined = text.casefold()
        ok = (
            all(fragment in joined for fragment in expected)
            and not any(fragment in joined for fragment in forbidden)
            and (max_tokens is None or token_count <= int(max_tokens))
        )
        hits += 1 if ok else 0
        details.append(
            {
                "query": query,
                "project": project,
                "mode": mode,
                "ok": ok,
                "expected_contains": expected,
                "forbidden_contains": forbidden,
                "token_count": token_count,
                "returned": returned,
            }
        )
    return {
        "total": total,
        "hits": hits,
        "hit_rate": hits / total if total else 0.0,
        "details": details,
    }


def _load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def _lower_list(items: Any) -> list[str]:
    return [str(item).casefold() for item in items]
