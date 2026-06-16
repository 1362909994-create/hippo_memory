from __future__ import annotations

from itertools import combinations

from hippocampus_memory.db import Database
from hippocampus_memory.models import MemoryRecord
from hippocampus_memory.utils import dumps_json, stable_id, utc_now

CONFLICT_TYPES = {"technical_fact", "constraint", "task_state"}
NEGATION_PAIRS = [
    ("是", "不是"),
    ("已确认", "未确认"),
    ("使用", "不使用"),
    ("accept", "reject"),
    ("confirmed", "unconfirmed"),
]
MUTEX_INTERFACES = {"spi", "rgb", "i2c", "uart"}


class ConflictDetector:
    def __init__(self, db: Database) -> None:
        self.db = db

    def detect_for_project(self, project: str | None) -> list[dict[str, str]]:
        memories = self.db.list_memories(
            project=project,
            include_archived=True,
            include_private=True,
            include_sensitive=True,
            limit=500,
        )
        conflicts = detect_conflicts(memories)
        self._persist(conflicts, project)
        return conflicts

    def _persist(self, conflicts: list[dict[str, str]], project: str | None) -> None:
        if not conflicts:
            return
        now = utc_now()
        with self.db.connect() as conn:
            for conflict in conflicts:
                memory_ids = sorted(conflict["memory_ids"].split(","))
                existing = conn.execute(
                    """
                    SELECT id FROM conflicts
                    WHERE status = 'open'
                      AND COALESCE(project, '') = COALESCE(?, '')
                      AND entity = ?
                      AND memory_ids = ?
                    LIMIT 1
                    """,
                    (project, conflict.get("entity"), dumps_json(memory_ids)),
                ).fetchone()
                if existing:
                    continue
                conn.execute(
                    """
                    INSERT INTO conflicts
                        (id, project, entity, attribute, memory_ids, description,
                         resolution, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, 'open', ?, ?)
                    """,
                    (
                        stable_id("cfl"),
                        project,
                        conflict.get("entity"),
                        conflict.get("attribute", "content"),
                        dumps_json(memory_ids),
                        conflict["description"],
                        now,
                        now,
                    ),
                )


def detect_conflicts(memories: list[MemoryRecord]) -> list[dict[str, str]]:
    relevant = [memory for memory in memories if memory.memory_type in CONFLICT_TYPES]
    conflicts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for left, right in combinations(relevant, 2):
        shared_entities = set(left.entities) & set(right.entities)
        if not shared_entities:
            continue
        if not _contents_conflict(left.content, right.content):
            continue
        key = tuple(sorted([left.id, right.id]))
        if key in seen:
            continue
        seen.add(key)
        entity = sorted(shared_entities)[0]
        conflicts.append(
            {
                "entity": entity,
                "attribute": "content",
                "memory_ids": ",".join(key),
                "description": (
                    f"Possible conflict for {entity}: memory {left.id} and {right.id} "
                    "contain mutually exclusive or negated statements."
                ),
            }
        )
    return conflicts


def _contents_conflict(left: str, right: str) -> bool:
    l_text = left.casefold()
    r_text = right.casefold()
    for positive, negative in NEGATION_PAIRS:
        p = positive.casefold()
        n = negative.casefold()
        if (p in l_text and n in r_text) or (n in l_text and p in r_text):
            return True
    left_interfaces = {item for item in MUTEX_INTERFACES if item in l_text}
    right_interfaces = {item for item in MUTEX_INTERFACES if item in r_text}
    return bool(
        left_interfaces
        and right_interfaces
        and left_interfaces.isdisjoint(right_interfaces)
    )
