from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from hippocampus_memory.conflict_detector import ConflictDetector
from hippocampus_memory.db import Database
from hippocampus_memory.models import MemoryStatus


class Consolidator:
    def __init__(self, db: Database) -> None:
        self.db = db

    def consolidate(self, project: str | None = None) -> dict[str, int | str | list[str]]:
        memories = self.db.list_memories(
            project=project,
            include_archived=True,
            include_private=True,
            include_sensitive=True,
            limit=1000,
        )
        expired_count = self._archive_expired(memories)
        missing_source_count = self._archive_missing_source_chunks(project)
        by_hash: dict[str, list[str]] = defaultdict(list)
        for memory in memories:
            if memory.content_hash:
                by_hash[memory.content_hash].append(memory.id)
        merged_count = 0
        for ids in by_hash.values():
            if len(ids) <= 1:
                continue
            for duplicate_id in ids[1:]:
                if self.db.update_memory_status(duplicate_id, MemoryStatus.ARCHIVED):
                    merged_count += 1

        archived_count = 0
        for memory in memories:
            if memory.importance < 0.2 and memory.usage_count == 0:
                if self.db.update_memory_status(memory.id, MemoryStatus.ARCHIVED):
                    archived_count += 1

        conflicts = ConflictDetector(self.db).detect_for_project(project)
        summary = self._project_summary(project)
        if project:
            self.db.insert_or_update_project(project, summary=summary)
        return {
            "merged_count": merged_count,
            "archived_count": archived_count + expired_count + missing_source_count,
            "expired_count": expired_count,
            "missing_source_chunk_count": missing_source_count,
            "conflict_count": len(conflicts),
            "generated_summary": summary,
            "warnings": [],
        }

    def _archive_expired(self, memories) -> int:
        count = 0
        now = datetime.now(UTC)
        for memory in memories:
            if not memory.expires_at:
                continue
            try:
                expires_at = datetime.fromisoformat(memory.expires_at)
            except ValueError:
                continue
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= now and self.db.update_memory_status(memory.id, MemoryStatus.ARCHIVED):
                count += 1
        return count

    def _archive_missing_source_chunks(self, project: str | None) -> int:
        if not project:
            return 0
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT c.memory_id
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                JOIN memories m ON m.id = c.memory_id
                WHERE c.project = ?
                  AND f.status = 'missing'
                  AND m.status = 'active'
                  AND m.source = 'project_index'
                """,
                (project,),
            ).fetchall()
            memory_ids = [row["memory_id"] for row in rows if row["memory_id"]]
            if not memory_ids:
                return 0
            now = datetime.now(UTC).replace(microsecond=0).isoformat()
            conn.executemany(
                """
                UPDATE memories
                SET status = ?, updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                [(MemoryStatus.ARCHIVED, now, memory_id) for memory_id in memory_ids],
            )
            return len(memory_ids)

    def _project_summary(self, project: str | None) -> str:
        memories = self.db.list_memories(project=project, limit=20)
        if not memories:
            return "No active memories yet."
        task_states = [memory.content for memory in memories if memory.memory_type == "task_state"]
        constraints = [memory.content for memory in memories if memory.memory_type == "constraint"]
        lines = []
        if task_states:
            lines.append(f"Current state: {task_states[0]}")
        if constraints:
            lines.append(f"Key constraint: {constraints[0]}")
        return " ".join(lines) or f"{len(memories)} active memories."
