from __future__ import annotations

import re
from pathlib import Path

from hippocampus_memory.ast_indexer import extract_python_index
from hippocampus_memory.db import Database
from hippocampus_memory.file_filters import is_indexable_file, should_ignore_path
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.models import MemoryType
from hippocampus_memory.python_lsp import extract_python_lsp_index
from hippocampus_memory.utils import (
    content_hash,
    dumps_json,
    estimate_tokens,
    normalize_text,
    safe_relative,
    stable_id,
    utc_now,
)


class ProjectIndexer:
    def __init__(self, db: Database, writer: MemoryWriter | None = None) -> None:
        self.db = db
        self.writer = writer or MemoryWriter(db)

    def index_project(self, root_path: str | Path, project: str) -> dict[str, int]:
        root = Path(root_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"project path does not exist or is not a directory: {root}")
        self.db.insert_or_update_project(project, root_path=str(root))
        indexed = 0
        skipped = 0
        active_paths: set[str] = set()
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            rel = Path(safe_relative(path, root))
            if should_ignore_path(rel) or not is_indexable_file(
                path,
                self.db.settings.max_file_size_bytes,
            ):
                skipped += 1
                continue
            if self._index_file(path, root, project):
                active_paths.add(rel.as_posix())
                indexed += 1
            else:
                skipped += 1
        stale_files = self._mark_missing_files(project, active_paths)
        orphaned_source_memories = self._archive_unlinked_source_memories(project)
        return {
            "indexed_files": indexed,
            "skipped_files": skipped,
            "stale_files": stale_files,
            "orphaned_source_memories": orphaned_source_memories,
        }

    def _index_file(self, path: Path, root: Path, project: str) -> bool:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if not normalize_text(raw):
            return False
        relative = safe_relative(path, root)
        digest = content_hash(raw)
        symbols, imports, calls = extract_file_index(raw, path.suffix)
        code_symbols, code_references = extract_code_intelligence(raw, path.suffix)
        summary = summarize_file(raw, relative, symbols=symbols, imports=imports, calls=calls)
        file_id = stable_id("file")
        now = utc_now()
        with self.db.connect() as conn:
            existing = conn.execute(
                """
                SELECT id, content_hash FROM files
                WHERE project = ? AND relative_path = ?
                """,
                (project, relative),
            ).fetchone()
            existing_file_id = existing["id"] if existing else None
            existing_hash = existing["content_hash"] if existing else None
            chunk_count = 0
            if existing_file_id:
                chunk_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM chunks WHERE file_id = ?",
                        (existing_file_id,),
                    ).fetchone()[0]
                )
            conn.execute(
                """
                INSERT INTO files
                    (id, project, path, relative_path, language, size_bytes, modified_at,
                     indexed_at, content_hash, summary, symbols, imports, calls, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                ON CONFLICT(project, relative_path) DO UPDATE SET
                    path = excluded.path,
                    language = excluded.language,
                    size_bytes = excluded.size_bytes,
                    modified_at = excluded.modified_at,
                    indexed_at = excluded.indexed_at,
                    content_hash = excluded.content_hash,
                    summary = excluded.summary,
                    symbols = excluded.symbols,
                    imports = excluded.imports,
                    calls = excluded.calls,
                    status = 'active'
                """,
                (
                    file_id,
                    project,
                    str(path),
                    relative,
                    language_for_suffix(path.suffix),
                    path.stat().st_size,
                    utc_now(),
                    now,
                    digest,
                    summary,
                    dumps_json(symbols),
                    dumps_json(imports),
                    dumps_json(calls),
                ),
            )
            row = conn.execute(
                "SELECT id FROM files WHERE project = ? AND relative_path = ?",
                (project, relative),
            ).fetchone()
            actual_file_id = row["id"]
            summary_refresh_needed = False
            if existing_hash == digest and chunk_count > 0:
                summary_refresh_needed = _chunks_need_summary_refresh(conn, actual_file_id)
            skip_chunks = existing_hash == digest and chunk_count > 0 and not summary_refresh_needed
            if skip_chunks:
                old_memory_ids = []
            else:
                old_memory_ids = [
                    row["memory_id"]
                    for row in conn.execute(
                        "SELECT memory_id FROM chunks WHERE file_id = ? AND memory_id IS NOT NULL",
                        (actual_file_id,),
                    ).fetchall()
                ]
                conn.execute("DELETE FROM chunks WHERE file_id = ?", (actual_file_id,))
            if old_memory_ids and (existing_hash != digest or summary_refresh_needed):
                conn.executemany(
                    """
                    UPDATE memories
                    SET status = 'archived', updated_at = ?
                    WHERE id = ? AND status = 'active'
                    """,
                    [(now, memory_id) for memory_id in old_memory_ids],
                )

        self.db.replace_code_intelligence(
            project=project,
            file_id=actual_file_id,
            relative_path=relative,
            symbols=code_symbols,
            references=code_references,
        )
        if skip_chunks:
            return True

        for chunk in split_chunks(raw):
            chunk_summary = summarize_chunk(relative, summary, chunk)
            write_result = self.writer.write(
                content=chunk["content"],
                memory_type=MemoryType.SOURCE_CHUNK,
                project=project,
                source="project_index",
                source_ref=relative,
                confidence=0.8,
                importance=0.4,
                tags=["file", language_for_suffix(path.suffix)],
                metadata={"relative_path": relative, "start_line": chunk["start_line"]},
                summary=chunk_summary,
            )
            with self.db.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO chunks
                        (id, project, file_id, memory_id, chunk_type, content, summary,
                         start_line, end_line, token_estimate, content_hash, created_at)
                    VALUES (?, ?, ?, ?, 'file', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("chk"),
                        project,
                        actual_file_id,
                        write_result.memory_id,
                        chunk["content"],
                        chunk_summary,
                        chunk["start_line"],
                        chunk["end_line"],
                        estimate_tokens(chunk["content"]),
                        content_hash(chunk["content"]),
                        now,
                    ),
                )
        return True

    def _mark_missing_files(self, project: str, active_paths: set[str]) -> int:
        with self.db.connect() as conn:
            if not active_paths:
                cur = conn.execute(
                    "UPDATE files SET status = 'missing' WHERE project = ? AND status = 'active'",
                    (project,),
                )
                stale_count = cur.rowcount
            else:
                placeholders = ",".join("?" for _ in active_paths)
                params = [project, *sorted(active_paths)]
                cur = conn.execute(
                    f"""
                    UPDATE files
                    SET status = 'missing'
                    WHERE project = ?
                      AND status = 'active'
                      AND relative_path NOT IN ({placeholders})
                    """,
                    params,
                )
                stale_count = cur.rowcount
        self.db.clear_code_intelligence_for_missing_files(project)
        return stale_count

    def _archive_unlinked_source_memories(self, project: str) -> int:
        now = utc_now()
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT m.id
                FROM memories m
                LEFT JOIN chunks c ON c.memory_id = m.id
                WHERE m.project = ?
                  AND m.memory_type = ?
                  AND m.source = 'project_index'
                  AND m.status = 'active'
                  AND c.memory_id IS NULL
                """,
                (project, MemoryType.SOURCE_CHUNK),
            ).fetchall()
            memory_ids = [row["id"] for row in rows]
            if not memory_ids:
                return 0
            conn.executemany(
                """
                UPDATE memories
                SET status = 'archived', updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                [(now, memory_id) for memory_id in memory_ids],
            )
            return len(memory_ids)


def summarize_file(
    content: str,
    relative_path: str,
    *,
    symbols: list[str] | None = None,
    imports: list[str] | None = None,
    calls: list[str] | None = None,
) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return f"{relative_path} is empty."
    first = " ".join(lines[:5])
    parts = [f"{relative_path}: {first[:220]}"]
    if symbols:
        parts.append(f"symbols={', '.join(symbols[:12])}")
    if imports:
        parts.append(f"imports={', '.join(imports[:8])}")
    if calls:
        parts.append(f"calls={', '.join(calls[:12])}")
    return "; ".join(parts)[:500]


def summarize_chunk(relative_path: str, file_summary: str, chunk: dict[str, int | str]) -> str:
    start = int(chunk["start_line"])
    end = int(chunk["end_line"])
    return f"{relative_path}:L{start}-{end}: {file_summary}"


def split_chunks(content: str, max_lines: int = 120) -> list[dict[str, int | str]]:
    lines = content.splitlines()
    chunks: list[dict[str, int | str]] = []
    for start in range(0, len(lines), max_lines):
        part = lines[start : start + max_lines]
        if not any(line.strip() for line in part):
            continue
        chunks.append(
            {
                "content": "\n".join(part),
                "start_line": start + 1,
                "end_line": start + len(part),
            }
        )
    return chunks or [{"content": content, "start_line": 1, "end_line": max(1, len(lines))}]


def _chunks_need_summary_refresh(conn, file_id: str) -> bool:
    rows = conn.execute(
        """
        SELECT c.summary, c.start_line, c.end_line, m.summary AS memory_summary
        FROM chunks c
        LEFT JOIN memories m ON m.id = c.memory_id
        WHERE c.file_id = ?
        """,
        (file_id,),
    ).fetchall()
    for row in rows:
        marker = f":L{int(row['start_line'] or 1)}-{int(row['end_line'] or 1)}:"
        if marker not in str(row["summary"] or ""):
            return True
        if marker not in str(row["memory_summary"] or ""):
            return True
    return False


def extract_symbols(content: str, suffix: str) -> list[str]:
    patterns = [
        r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*(?:export\s+)?(?:class|interface|type)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"^\s*(?:public|private|protected)?\s*(?:class|interface)\s+([A-Za-z_][A-Za-z0-9_]*)",
    ]
    del suffix
    symbols: list[str] = []
    for pattern in patterns:
        symbols.extend(re.findall(pattern, content, flags=re.MULTILINE))
    return sorted(set(symbols))


def extract_file_index(content: str, suffix: str) -> tuple[list[str], list[str], list[str]]:
    if suffix.casefold() == ".py":
        symbols, imports, calls = extract_python_index(content)
        if symbols or imports or calls:
            return symbols, imports, calls
    return (
        extract_symbols(content, suffix),
        extract_imports(content, suffix),
        extract_calls(content),
    )


def extract_code_intelligence(content: str, suffix: str):
    if suffix.casefold() == ".py":
        return extract_python_lsp_index(content)
    return [], []


def extract_imports(content: str, suffix: str) -> list[str]:
    patterns = [
        r"^\s*import\s+(.+)$",
        r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+",
        r"^\s*#include\s+[<\"]([^>\"]+)[>\"]",
        r"^\s*use\s+(.+?);",
        r"^\s*using\s+(.+?);",
    ]
    del suffix
    imports: list[str] = []
    for pattern in patterns:
        imports.extend(re.findall(pattern, content, flags=re.MULTILINE))
    return sorted({item.strip()[:160] for item in imports})


def extract_calls(content: str) -> list[str]:
    ignored = {
        "if",
        "for",
        "while",
        "switch",
        "return",
        "def",
        "class",
        "function",
        "catch",
        "with",
        "print",
    }
    calls = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", content)
    return sorted({call for call in calls if call.casefold() not in ignored})


def language_for_suffix(suffix: str) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript-react",
        ".jsx": "javascript-react",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ps1": "powershell",
        ".sh": "shell",
    }.get(suffix.casefold(), suffix.casefold().lstrip(".") or "text")
