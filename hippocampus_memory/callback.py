from __future__ import annotations

from hippocampus_memory.db import Database
from hippocampus_memory.packer import MemoryPacker
from hippocampus_memory.utils import loads_json


def callback_pack(
    db: Database,
    *,
    project: str,
    intent: str,
    session_key: str = "default",
    max_tokens: int = 1500,
    source_chunk_limit: int = 2,
    compact: bool = True,
) -> dict[str, object]:
    session = db.get_callback_session(project, session_key)
    seen_memory_ids = loads_json(session.get("seen_memory_ids") if session else None, [])
    packer = MemoryPacker(db)
    pack = packer.pack(
        intent,
        project=project,
        max_tokens=max_tokens,
        source_chunk_limit=source_chunk_limit,
        compact=compact,
        exclude_memory_ids=seen_memory_ids,
    )
    included_memory_ids = packer.last_included_memory_ids
    updated_seen_ids = _dedupe_ids([*seen_memory_ids, *included_memory_ids])
    db.update_callback_session(
        project=project,
        session_key=session_key,
        seen_memory_ids=updated_seen_ids,
    )
    return {
        "project": project,
        "session_key": session_key,
        "excluded_memory_ids": seen_memory_ids,
        "included_memory_ids": included_memory_ids,
        "seen_memory_ids": updated_seen_ids,
        "text": pack,
        "pack": pack,
    }


def reset_callback_pack(
    db: Database,
    *,
    project: str,
    session_key: str = "default",
) -> dict[str, object]:
    return {
        "project": project,
        "session_key": session_key,
        "reset": db.reset_callback_session(project, session_key),
    }


def _dedupe_ids(memory_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for memory_id in memory_ids:
        if memory_id in seen:
            continue
        seen.add(memory_id)
        output.append(memory_id)
    return output
