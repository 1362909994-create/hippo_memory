from __future__ import annotations

from pathlib import Path
from typing import Any

from hippocampus_memory.db import Database
from hippocampus_memory.git_utils import git_snapshot
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.models import MemoryType
from hippocampus_memory.utils import dumps_json, stable_id, utc_now


def record_run_session(
    db: Database,
    *,
    project: str,
    intent: str,
    command: list[str],
    returncode: int,
    context_file: Path | None,
    stdout: str,
    stderr: str,
    cwd: str | Path | None = None,
    write_memory: bool = False,
) -> dict[str, Any]:
    payload = {
        "intent": intent,
        "command": command,
        "returncode": returncode,
        "context_file": str(context_file) if context_file else None,
        "stdout_excerpt": _excerpt(stdout),
        "stderr_excerpt": _excerpt(stderr),
        "git": git_snapshot(cwd),
    }
    event_id = stable_id("evt")
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO events (id, event_type, project, payload, created_at)
            VALUES (?, 'session.run', ?, ?, ?)
            """,
            (event_id, project, dumps_json(payload), utc_now()),
        )
    memory_id = None
    if write_memory:
        memory_type = MemoryType.TASK_STATE if returncode == 0 else MemoryType.FAILURE
        content = _session_memory_content(intent, command, returncode, payload)
        result = MemoryWriter(db).write(
            project=project,
            memory_type=memory_type,
            content=content,
            source="hippo_run",
            confidence=0.75 if returncode == 0 else 0.85,
            importance=0.65 if returncode == 0 else 0.8,
            metadata={"event_id": event_id},
        )
        memory_id = result.memory_id
    return {"event_id": event_id, "memory_id": memory_id}


def _excerpt(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _session_memory_content(
    intent: str,
    command: list[str],
    returncode: int,
    payload: dict[str, Any],
) -> str:
    status = "succeeded" if returncode == 0 else f"failed with return code {returncode}"
    command_text = " ".join(command) if command else "context-only run"
    lines = [
        f"Vibe coding session for intent '{intent}' {status}.",
        f"Command: {command_text}.",
    ]
    git = payload.get("git")
    if isinstance(git, dict) and git.get("status_short"):
        lines.append("Git status changed or has pending files; inspect diff before next step.")
    if payload.get("stderr_excerpt"):
        lines.append(f"Error excerpt: {payload['stderr_excerpt'][:300]}")
    return " ".join(lines)
