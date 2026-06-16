from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hippocampus_memory.config import Settings
from hippocampus_memory.models import MemoryRecord, MemoryStatus
from hippocampus_memory.utils import cjk_search_terms, dumps_json, loads_json, stable_id, utc_now


class Database:
    def __init__(self, path: str | Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env(path)
        self.path = self.settings.db_path

    def connect(self) -> sqlite3.Connection:
        self.settings.ensure_parent()
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            _ensure_column(conn, "files", "calls", "TEXT NOT NULL DEFAULT '[]'")

    def insert_memory(self, memory: MemoryRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, content, summary, memory_type, project, entities, tags, source,
                    source_ref, confidence, importance, status, visibility, created_at,
                    updated_at, last_used_at, usage_count, ttl_days, expires_at,
                    content_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.content,
                    memory.summary,
                    memory.memory_type,
                    memory.project,
                    dumps_json(memory.entities),
                    dumps_json(memory.tags),
                    memory.source,
                    memory.source_ref,
                    memory.confidence,
                    memory.importance,
                    memory.status,
                    memory.visibility,
                    memory.created_at,
                    memory.updated_at,
                    memory.last_used_at,
                    memory.usage_count,
                    memory.ttl_days,
                    memory.expires_at,
                    memory.content_hash,
                    dumps_json(memory.metadata),
                ),
            )
            self.upsert_memory_fts(conn, memory)
            conn.execute(
                """
                INSERT INTO events (id, event_type, project, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    stable_id("evt"),
                    "memory.write",
                    memory.project,
                    dumps_json({"memory_id": memory.id, "memory_type": memory.memory_type}),
                    utc_now(),
                ),
            )

    def update_memory_status(self, memory_id: str, status: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), memory_id),
            )
            if status == MemoryStatus.DELETED:
                conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            return cur.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        with self.connect() as conn:
            conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount > 0

    def delete_project_memories(self, project: str, hard: bool) -> int:
        with self.connect() as conn:
            rows = conn.execute("SELECT id FROM memories WHERE project = ?", (project,)).fetchall()
            ids = [row["id"] for row in rows]
            fts_rows = [(mid,) for mid in ids]
            if hard:
                conn.executemany("DELETE FROM memory_fts WHERE memory_id = ?", fts_rows)
                conn.executemany("DELETE FROM memory_vectors WHERE memory_id = ?", fts_rows)
                conn.execute("DELETE FROM memories WHERE project = ?", (project,))
            else:
                conn.execute(
                    "UPDATE memories SET status = ?, updated_at = ? WHERE project = ?",
                    (MemoryStatus.DELETED, utc_now(), project),
                )
                conn.executemany("DELETE FROM memory_fts WHERE memory_id = ?", fts_rows)
            return len(ids)

    def find_duplicate(
        self,
        content_hash: str,
        project: str | None,
        memory_type: str,
    ) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM memories
                WHERE content_hash = ?
                  AND COALESCE(project, '') = COALESCE(?, '')
                  AND memory_type = ?
                  AND status = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (content_hash, project, memory_type, MemoryStatus.ACTIVE),
            ).fetchone()
            return row["id"] if row else None

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return row_to_memory(row) if row else None

    def list_memories(
        self,
        project: str | None = None,
        include_archived: bool = False,
        include_private: bool = False,
        include_sensitive: bool = False,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        where = ["status != ?"]
        params: list[Any] = [MemoryStatus.DELETED]
        if project is not None:
            where.append("project = ?")
            params.append(project)
        if not include_archived:
            where.append("status != ?")
            params.append(MemoryStatus.ARCHIVED)
        if not include_private:
            where.append("visibility != 'private'")
        if not include_sensitive:
            where.append("visibility != 'sensitive'")
        params.append(limit)
        sql = f"""
            SELECT * FROM memories
            WHERE {' AND '.join(where)}
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
        """
        with self.connect() as conn:
            return [row_to_memory(row) for row in conn.execute(sql, params).fetchall()]

    def search_fts(
        self,
        query: str,
        project: str | None,
        include_archived: bool,
        include_private: bool,
        include_sensitive: bool,
        limit: int,
    ) -> list[tuple[MemoryRecord, float]]:
        match_query = build_fts_query(query)
        if not match_query:
            return []
        where = ["m.status != ?"]
        params: list[Any] = [MemoryStatus.DELETED]
        if project:
            where.append("m.project = ?")
            params.append(project)
        if not include_archived:
            where.append("m.status != ?")
            params.append(MemoryStatus.ARCHIVED)
        if not include_private:
            where.append("m.visibility != 'private'")
        if not include_sensitive:
            where.append("m.visibility != 'sensitive'")
        params.extend([match_query, limit])
        sql = f"""
            SELECT m.*, bm25(memory_fts) AS rank
            FROM memory_fts
            JOIN memories m ON m.id = memory_fts.memory_id
            WHERE {' AND '.join(where)}
              AND memory_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
        """
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        results: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            rank = float(row["rank"] or 0.0)
            keyword_score = 1.0 / (1.0 + max(0.0, abs(rank)))
            results.append((row_to_memory(row), keyword_score))
        return results

    def upsert_memory_fts(self, conn: sqlite3.Connection, memory: MemoryRecord) -> None:
        conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory.id,))
        if memory.status == MemoryStatus.DELETED:
            return
        conn.execute(
            """
            INSERT INTO memory_fts (memory_id, content, summary, entities, tags, project)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.content,
                memory.summary or "",
                " ".join(memory.entities),
                " ".join(memory.tags),
                memory.project or "",
            ),
        )

    def upsert_vector(self, memory_id: str, vector: list[float]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_vectors (memory_id, vector_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    vector_json = excluded.vector_json,
                    updated_at = excluded.updated_at
                """,
                (memory_id, dumps_json(vector), utc_now()),
            )

    def get_vectors(self, memory_ids: Iterable[str] | None = None) -> dict[str, list[float]]:
        params: list[Any] = []
        where = ""
        if memory_ids is not None:
            ids = list(memory_ids)
            if not ids:
                return {}
            where = f"WHERE memory_id IN ({','.join('?' for _ in ids)})"
            params.extend(ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT memory_id, vector_json FROM memory_vectors {where}",
                params,
            )
            return {row["memory_id"]: loads_json(row["vector_json"], []) for row in rows.fetchall()}

    def insert_or_update_project(
        self,
        name: str,
        root_path: str | None = None,
        description: str | None = None,
        summary: str | None = None,
    ) -> str:
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM projects WHERE name = ?", (name,)).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE projects
                    SET root_path = COALESCE(?, root_path),
                        description = COALESCE(?, description),
                        summary = COALESCE(?, summary),
                        updated_at = ?
                    WHERE name = ?
                    """,
                    (root_path, description, summary, now, name),
                )
                return row["id"]
            project_id = stable_id("prj")
            conn.execute(
                """
                INSERT INTO projects
                    (id, name, root_path, description, created_at, updated_at,
                     summary, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', '{}')
                """,
                (project_id, name, root_path, description, now, now, summary),
            )
            return project_id

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
            return [dict(row) for row in rows]

    def get_project(self, name: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def list_files(self, project: str, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM files
                WHERE project = ? AND status = 'active'
                ORDER BY relative_path ASC
                LIMIT ?
                """,
                (project, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_conflicts(self, project: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        where = "WHERE status = 'open'"
        params: list[Any] = []
        if project:
            where += " AND project = ?"
            params.append(project)
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM conflicts
                {where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def update_conflict(
        self,
        conflict_id: str,
        *,
        resolution: str | None = None,
        status: str = "resolved",
    ) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE conflicts
                SET resolution = COALESCE(?, resolution),
                    status = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (resolution, status, utc_now(), conflict_id),
            )
            return cur.rowcount > 0

    def memory_counts_by_type(self, project: str | None = None) -> dict[str, int]:
        where = "WHERE status != 'deleted'"
        params: list[Any] = []
        if project:
            where += " AND project = ?"
            params.append(project)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_type, COUNT(*) AS count
                FROM memories
                {where}
                GROUP BY memory_type
                ORDER BY memory_type ASC
                """,
                params,
            ).fetchall()
            return {row["memory_type"]: int(row["count"]) for row in rows}

    def insert_candidate(
        self,
        *,
        project: str | None,
        content: str,
        memory_type: str,
        confidence: float,
        importance: float,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        candidate_id = stable_id("cand")
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_candidates
                    (id, project, content, memory_type, confidence, importance,
                     source, metadata, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    candidate_id,
                    project,
                    content,
                    memory_type,
                    confidence,
                    importance,
                    source,
                    dumps_json(metadata or {}),
                    now,
                    now,
                ),
            )
        return candidate_id

    def list_candidates(
        self,
        project: str | None = None,
        status: str = "pending",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where = ["status = ?"]
        params: list[Any] = [status]
        if project:
            where.append("project = ?")
            params.append(project)
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memory_candidates
                WHERE {' AND '.join(where)}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            return dict(row) if row else None

    def update_candidate_status(self, candidate_id: str, status: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE memory_candidates SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), candidate_id),
            )
            return cur.rowcount > 0

    def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> bool:
        old_memory = self.get_memory(old_memory_id)
        new_memory = self.get_memory(new_memory_id)
        if not old_memory or not new_memory:
            return False
        if old_memory.project != new_memory.project:
            return False
        metadata = dict(old_memory.metadata)
        metadata["superseded_by"] = new_memory_id
        metadata["superseded_at"] = utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE memories
                SET status = ?, metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    MemoryStatus.OUTDATED,
                    dumps_json(metadata),
                    metadata["superseded_at"],
                    old_memory_id,
                ),
            )
            return cur.rowcount > 0

    def get_callback_session(self, project: str, session_key: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM callback_sessions
                WHERE project = ? AND session_key = ?
                """,
                (project, session_key),
            ).fetchone()
            return dict(row) if row else None

    def update_callback_session(
        self,
        *,
        project: str,
        session_key: str,
        seen_memory_ids: list[str],
    ) -> str:
        now = utc_now()
        session_id = stable_id("cb")
        deduped_ids = []
        seen = set()
        for memory_id in seen_memory_ids:
            if memory_id in seen:
                continue
            seen.add(memory_id)
            deduped_ids.append(memory_id)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO callback_sessions
                    (id, project, session_key, seen_memory_ids, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project, session_key) DO UPDATE SET
                    seen_memory_ids = excluded.seen_memory_ids,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    project,
                    session_key,
                    dumps_json(deduped_ids),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                """
                SELECT id FROM callback_sessions
                WHERE project = ? AND session_key = ?
                """,
                (project, session_key),
            ).fetchone()
            return row["id"]

    def reset_callback_session(self, project: str, session_key: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM callback_sessions WHERE project = ? AND session_key = ?",
                (project, session_key),
            )
            return cur.rowcount > 0

    def replace_code_intelligence(
        self,
        *,
        project: str,
        file_id: str,
        relative_path: str,
        symbols: list[Any],
        references: list[Any],
    ) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute("DELETE FROM code_symbols WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM code_references WHERE file_id = ?", (file_id,))
            conn.executemany(
                """
                INSERT INTO code_symbols
                    (id, project, file_id, relative_path, name, qualified_name, kind,
                     container, line, end_line, signature, docstring, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        stable_id("sym"),
                        project,
                        file_id,
                        relative_path,
                        symbol.name,
                        symbol.qualified_name,
                        symbol.kind,
                        symbol.container,
                        symbol.line,
                        symbol.end_line,
                        symbol.signature,
                        symbol.docstring,
                        now,
                    )
                    for symbol in symbols
                ],
            )
            conn.executemany(
                """
                INSERT INTO code_references
                    (id, project, file_id, relative_path, symbol, kind, line,
                     column, container, context, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        stable_id("ref"),
                        project,
                        file_id,
                        relative_path,
                        reference.symbol,
                        reference.kind,
                        reference.line,
                        reference.column,
                        reference.container,
                        reference.context,
                        now,
                    )
                    for reference in references
                ],
            )

    def clear_code_intelligence_for_missing_files(self, project: str) -> None:
        with self.connect() as conn:
            missing_file_ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM files WHERE project = ? AND status = 'missing'",
                    (project,),
                ).fetchall()
            ]
            if not missing_file_ids:
                return
            conn.executemany(
                "DELETE FROM code_symbols WHERE file_id = ?",
                [(file_id,) for file_id in missing_file_ids],
            )
            conn.executemany(
                "DELETE FROM code_references WHERE file_id = ?",
                [(file_id,) for file_id in missing_file_ids],
            )

    def list_code_symbols(
        self,
        project: str,
        query: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        where = ["project = ?"]
        params: list[Any] = [project]
        if query:
            like = f"%{query}%"
            where.append("(name LIKE ? OR qualified_name LIKE ? OR relative_path LIKE ?)")
            params.extend([like, like, like])
        params.append(max(1, min(500, limit)))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM code_symbols
                WHERE {' AND '.join(where)}
                ORDER BY relative_path ASC, line ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def list_code_references(
        self,
        project: str,
        symbol: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM code_references
                WHERE project = ? AND symbol = ?
                ORDER BY relative_path ASC, line ASC, column ASC
                LIMIT ?
                """,
                (project, symbol, max(1, min(1000, limit))),
            ).fetchall()
            return [dict(row) for row in rows]

    def replace_code_diagnostics(
        self,
        *,
        project: str,
        diagnostics: list[Any],
        source: str,
    ) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM code_diagnostics WHERE project = ? AND source = ?",
                (project, source),
            )
            conn.executemany(
                """
                INSERT INTO code_diagnostics
                    (id, project, relative_path, severity, message, line, column,
                     end_line, end_column, rule, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        stable_id("diag"),
                        project,
                        diagnostic.relative_path,
                        diagnostic.severity,
                        diagnostic.message,
                        diagnostic.line,
                        diagnostic.column,
                        diagnostic.end_line,
                        diagnostic.end_column,
                        diagnostic.rule,
                        diagnostic.source,
                        now,
                    )
                    for diagnostic in diagnostics
                ],
            )

    def list_code_diagnostics(
        self,
        project: str,
        *,
        relative_paths: list[str] | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where = ["project = ?"]
        params: list[Any] = [project]
        if relative_paths:
            placeholders = ",".join("?" for _ in relative_paths)
            where.append(f"relative_path IN ({placeholders})")
            params.extend(relative_paths)
        if severity:
            where.append("severity = ?")
            params.append(severity)
        params.append(max(1, min(1000, limit)))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM code_diagnostics
                WHERE {' AND '.join(where)}
                ORDER BY
                    CASE severity
                        WHEN 'error' THEN 0
                        WHEN 'warning' THEN 1
                        ELSE 2
                    END,
                    relative_path ASC,
                    line ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def insert_token_ledger(
        self,
        *,
        project: str,
        intent: str,
        context_type: str,
        baseline_tokens: int,
        output_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        baseline_tokens = max(0, int(baseline_tokens))
        output_tokens = max(0, int(output_tokens))
        saved_tokens = max(0, baseline_tokens - output_tokens)
        savings_ratio = saved_tokens / baseline_tokens if baseline_tokens else 0.0
        ledger_id = stable_id("tok")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO token_ledger
                    (id, project, intent, context_type, baseline_tokens,
                     output_tokens, saved_tokens, savings_ratio, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ledger_id,
                    project,
                    intent,
                    context_type,
                    baseline_tokens,
                    output_tokens,
                    saved_tokens,
                    savings_ratio,
                    dumps_json(metadata or {}),
                    utc_now(),
                ),
            )
        return ledger_id

    def list_token_ledger(self, project: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM token_ledger
                WHERE project = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def token_ledger_summary(self, project: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS entry_count,
                    COALESCE(SUM(baseline_tokens), 0) AS baseline_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                    COALESCE(AVG(savings_ratio), 0) AS average_savings_ratio
                FROM token_ledger
                WHERE project = ?
                """,
                (project,),
            ).fetchone()
            by_type_rows = conn.execute(
                """
                SELECT
                    context_type,
                    COUNT(*) AS entry_count,
                    COALESCE(SUM(baseline_tokens), 0) AS baseline_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                    COALESCE(AVG(savings_ratio), 0) AS average_savings_ratio
                FROM token_ledger
                WHERE project = ?
                GROUP BY context_type
                ORDER BY context_type ASC
                """,
                (project,),
            ).fetchall()
        return {
            "project": project,
            "entry_count": int(row["entry_count"]),
            "baseline_tokens": int(row["baseline_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "saved_tokens": int(row["saved_tokens"]),
            "average_savings_ratio": float(row["average_savings_ratio"]),
            "by_context_type": [dict(item) for item in by_type_rows],
        }

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            counts = {
                "memories": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
                "active_memories": conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE status = 'active'"
                ).fetchone()[0],
                "projects": conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0],
                "files": conn.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "conflicts": conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0],
                "pending_candidates": conn.execute(
                    "SELECT COUNT(*) FROM memory_candidates WHERE status = 'pending'"
                ).fetchone()[0],
                "token_ledger_entries": conn.execute(
                    "SELECT COUNT(*) FROM token_ledger"
                ).fetchone()[0],
                "callback_sessions": conn.execute(
                    "SELECT COUNT(*) FROM callback_sessions"
                ).fetchone()[0],
                "code_symbols": conn.execute("SELECT COUNT(*) FROM code_symbols").fetchone()[0],
                "code_references": conn.execute(
                    "SELECT COUNT(*) FROM code_references"
                ).fetchone()[0],
                "code_diagnostics": conn.execute(
                    "SELECT COUNT(*) FROM code_diagnostics"
                ).fetchone()[0],
            }
        return {"db_path": str(self.path), **counts}

    def record_usage(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = utc_now()
        with self.connect() as conn:
            conn.executemany(
                """
                UPDATE memories
                SET usage_count = usage_count + 1, last_used_at = ?, updated_at = updated_at
                WHERE id = ?
                """,
                [(now, memory_id) for memory_id in memory_ids],
            )


def row_to_memory(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        content=row["content"],
        summary=row["summary"],
        memory_type=row["memory_type"],
        project=row["project"],
        entities=loads_json(row["entities"], []),
        tags=loads_json(row["tags"], []),
        source=row["source"],
        source_ref=row["source_ref"],
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        status=row["status"],
        visibility=row["visibility"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_used_at=row["last_used_at"],
        usage_count=int(row["usage_count"] or 0),
        ttl_days=row["ttl_days"],
        expires_at=row["expires_at"],
        content_hash=row["content_hash"],
        metadata=loads_json(row["metadata"], {}),
    )


def build_fts_query(query: str) -> str:
    import re

    ascii_terms = re.findall(r"[A-Za-z0-9_]+", query)
    terms: list[str] = []
    for term in [*ascii_terms, *cjk_search_terms(query)]:
        if term not in terms:
            terms.append(term)
    return " OR ".join(f'"{term}"' for term in terms[:16])


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT,
    memory_type TEXT NOT NULL,
    project TEXT,
    entities TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    source TEXT,
    source_ref TEXT,
    confidence REAL NOT NULL DEFAULT 0.8,
    importance REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    visibility TEXT NOT NULL DEFAULT 'project',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT,
    usage_count INTEGER NOT NULL DEFAULT 0,
    ttl_days INTEGER,
    expires_at TEXT,
    content_hash TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    memory_id UNINDEXED,
    content,
    summary,
    entities,
    tags,
    project
);

CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id TEXT PRIMARY KEY,
    vector_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    root_path TEXT,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    summary TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    language TEXT,
    size_bytes INTEGER NOT NULL,
    modified_at TEXT,
    indexed_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    summary TEXT,
    symbols TEXT NOT NULL DEFAULT '[]',
    imports TEXT NOT NULL DEFAULT '[]',
    calls TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    UNIQUE(project, relative_path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    file_id TEXT,
    memory_id TEXT,
    chunk_type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    start_line INTEGER,
    end_line INTEGER,
    token_estimate INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS conflicts (
    id TEXT PRIMARY KEY,
    project TEXT,
    entity TEXT,
    attribute TEXT,
    memory_ids TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL,
    resolution TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    project TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_candidates (
    id TEXT PRIMARY KEY,
    project TEXT,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.7,
    importance REAL NOT NULL DEFAULT 0.5,
    source TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS token_ledger (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    intent TEXT NOT NULL,
    context_type TEXT NOT NULL,
    baseline_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    saved_tokens INTEGER NOT NULL DEFAULT 0,
    savings_ratio REAL NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_ledger_project ON token_ledger(project);
CREATE INDEX IF NOT EXISTS idx_token_ledger_context_type ON token_ledger(context_type);

CREATE TABLE IF NOT EXISTS callback_sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    session_key TEXT NOT NULL,
    seen_memory_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project, session_key)
);

CREATE INDEX IF NOT EXISTS idx_callback_sessions_project ON callback_sessions(project);

CREATE TABLE IF NOT EXISTS code_symbols (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    file_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    container TEXT,
    line INTEGER NOT NULL,
    end_line INTEGER,
    signature TEXT,
    docstring TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_code_symbols_project_name
    ON code_symbols(project, name);
CREATE INDEX IF NOT EXISTS idx_code_symbols_project_qualified
    ON code_symbols(project, qualified_name);

CREATE TABLE IF NOT EXISTS code_references (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    file_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,
    line INTEGER NOT NULL,
    column INTEGER NOT NULL DEFAULT 0,
    container TEXT,
    context TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_code_references_project_symbol
    ON code_references(project, symbol);

CREATE TABLE IF NOT EXISTS code_diagnostics (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    line INTEGER NOT NULL,
    column INTEGER NOT NULL DEFAULT 0,
    end_line INTEGER,
    end_column INTEGER,
    rule TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_code_diagnostics_project_path
    ON code_diagnostics(project, relative_path);
CREATE INDEX IF NOT EXISTS idx_code_diagnostics_project_severity
    ON code_diagnostics(project, severity);
"""
