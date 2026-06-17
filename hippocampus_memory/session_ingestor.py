from __future__ import annotations

from typing import Any

from hippocampus_memory.db import Database
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.utils import loads_json


def write_summary_candidates(
    db: Database,
    summary: dict[str, list[dict[str, Any]]],
    project: str | None,
) -> dict[str, int]:
    writer = MemoryWriter(db)
    written = 0
    duplicates = 0
    for bucket, items in summary.items():
        for item in items:
            result = writer.write(
                project=project,
                source="session_summary",
                metadata={"summary_bucket": bucket},
                **item,
            )
            if result.duplicate:
                duplicates += 1
            elif result.created:
                written += 1
    return {"written": written, "duplicates": duplicates}


def queue_summary_candidates(
    db: Database,
    summary: dict[str, list[dict[str, Any]]],
    project: str | None,
) -> dict[str, Any]:
    candidate_ids: list[str] = []
    for bucket, items in summary.items():
        for item in items:
            candidate_ids.append(
                db.insert_candidate(
                    project=project,
                    content=str(item["content"]),
                    memory_type=str(item["memory_type"]),
                    confidence=float(item.get("confidence", 0.7)),
                    importance=float(item.get("importance", 0.5)),
                    source="session_summary",
                    metadata={"summary_bucket": bucket},
                )
            )
    return {"queued": len(candidate_ids), "candidate_ids": candidate_ids}


def accept_candidate(db: Database, candidate_id: str) -> dict[str, Any]:
    candidate = db.get_candidate(candidate_id)
    if not candidate:
        raise ValueError(f"candidate not found: {candidate_id}")
    if candidate["status"] != "pending":
        raise ValueError(f"candidate is not pending: {candidate_id}")
    metadata = loads_json(candidate.get("metadata"), {})
    result = MemoryWriter(db).write(
        project=candidate["project"],
        content=candidate["content"],
        memory_type=candidate["memory_type"],
        confidence=float(candidate["confidence"]),
        importance=float(candidate["importance"]),
        source=candidate["source"],
        entities=_list_metadata(metadata.get("entities")),
        tags=_list_metadata(metadata.get("tags")),
        visibility=metadata.get("visibility"),
        ttl_days=metadata.get("ttl_days"),
        metadata=metadata,
    )
    db.update_candidate_status(candidate_id, "accepted")
    return {"memory_id": result.memory_id, "created": result.created}


def discard_candidate(db: Database, candidate_id: str) -> bool:
    return db.update_candidate_status(candidate_id, "discarded")


def _list_metadata(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
