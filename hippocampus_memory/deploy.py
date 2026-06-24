from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from hippocampus_memory.db import Database
from hippocampus_memory.project_indexer import ProjectIndexer
from hippocampus_memory.project_resolver import (
    find_project_config,
    resolve_project_name,
    write_project_config,
)

HIPPO_DIR_NAME = ".hippo"
HIPPO_DB_NAME = "hippo.db"
CODEX_MEMORY_FILES = (
    "AGENTS.md",
    "AGENT.md",
)
HIPPO_MEMORY_START = "<!-- hippocampus-memory:start -->"
HIPPO_MEMORY_END = "<!-- hippocampus-memory:end -->"


def mcp_client_config(command: str | None = None) -> dict[str, object]:
    cmd = command or sys.executable
    return {
        "mcpServers": {
            "hippocampus-memory": {
                "command": cmd,
                "args": ["-m", "hippocampus_memory", "mcp"],
            }
        }
    }


def write_mcp_client_config(output: str | Path, command: str | None = None) -> Path:
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mcp_client_config(command), indent=2), encoding="utf-8")
    return path


def write_daemon_script(
    output: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> Path:
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        f"python -m hippocampus_memory daemon --host {host} --port {port}\n"
    )
    path.write_text(script, encoding="utf-8")
    return path


def deploy_codex(
    root: str | Path = ".",
    *,
    project: str | None = None,
    force_project_config: bool = False,
    index_project: bool = True,
    project_memory: bool = True,
) -> dict[str, Any]:
    """Prepare one project for Codex-oriented MCP memory usage."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"project path does not exist or is not a directory: {root_path}")

    project_name = project or resolve_project_name(cwd=root_path)
    hippo_dir = root_path / HIPPO_DIR_NAME
    hippo_dir.mkdir(parents=True, exist_ok=True)
    db_path = hippo_dir / HIPPO_DB_NAME

    config_written = _ensure_project_config(root_path, project_name, force=force_project_config)
    db = Database(db_path)
    db.initialize()
    db.insert_or_update_project(project_name, root_path=str(root_path))
    index_result = (
        ProjectIndexer(db).index_project(root_path, project_name)
        if index_project
        else {"indexed_files": 0, "skipped_files": 0, "stale_files": 0}
    )
    mcp_script = write_project_mcp_script(hippo_dir)
    mcp_config_path = hippo_dir / "codex-mcp-config.json"
    mcp_config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "hippo_memory": {
                        "command": "hippo",
                        "args": ["mcp-project", "--root", str(root_path)],
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    memory_file = None
    memory_file_updated = False
    if project_memory:
        memory_file, memory_file_updated = ensure_codex_project_memory(root_path, project_name)
    gitignore_updated = ensure_gitignore_entry(root_path, f"{HIPPO_DIR_NAME}/")
    return {
        "project": project_name,
        "root": str(root_path),
        "db_path": str(db_path),
        "project_config": str(root_path / ".hippo.toml"),
        "project_config_written": config_written,
        "index": index_result,
        "mcp_script": str(mcp_script),
        "codex_mcp_config": str(mcp_config_path),
        "codex_project_memory": str(memory_file) if memory_file else None,
        "codex_project_memory_updated": memory_file_updated,
        "gitignore_updated": gitignore_updated,
        "next": {
            "mcp": f"Use {mcp_config_path} as the Codex MCP server config snippet.",
            "memory": "Open Codex in this project; AGENTS.md will tell Codex when to use Hippo.",
        },
    }


def write_project_mcp_script(hippo_dir: str | Path) -> Path:
    path = Path(hippo_dir).expanduser().resolve() / "hippo-mcp.ps1"
    path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        "$ProjectRoot = Split-Path -Parent $PSScriptRoot\n"
        "Set-Location $ProjectRoot\n"
        "hippo mcp-project --root $ProjectRoot\n"
    )
    path.write_text(script, encoding="utf-8")
    return path


def ensure_codex_project_memory(root: str | Path, project: str) -> tuple[Path, bool]:
    root_path = Path(root).expanduser().resolve()
    target = _find_codex_memory_file(root_path) or (root_path / "AGENTS.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    block = codex_project_memory_block(project)
    old = target.read_text(encoding="utf-8", errors="ignore") if target.exists() else ""
    new = _upsert_marked_block(old, block)
    if new == old:
        return target, False
    target.write_text(new, encoding="utf-8")
    return target, True


def codex_project_memory_block(project: str) -> str:
    return (
        f"{HIPPO_MEMORY_START}\n"
        "## Hippocampus Memory\n\n"
        f"- This project is deployed with project-local hippocampus-memory as `{project}`.\n"
        "- The MCP server name is `hippo_memory`; prefer its automatic tools when external "
        "project memory, code impact analysis, or compact context would help.\n"
        "- At the start of non-trivial coding/debugging/architecture tasks, call "
        "`hippo_memory_context_auto` with the current task intent. Use "
        '`session_key="codex"` unless the user gives a better session name. Trust the '
        "tool when it returns that no external memory is needed.\n"
        "- For direct symbol questions use `hippo_memory_code_symbols` or "
        "`hippo_memory_code_references`; otherwise prefer `hippo_memory_context_auto` over "
        "manually choosing profile, impact, callback, or bundle tools.\n"
        "- Near the end of meaningful work, call `hippo_memory_memory_auto_store` with a "
        "concise transcript summary. It will write high-confidence non-sensitive memories, "
        "queue uncertain memories, and skip low-value content.\n"
        "- Do not recall private/sensitive memories unless explicitly requested. Do not force "
        "long-term writes for sensitive or uncertain facts.\n"
        "- Keep recalled context short, cite files when making code claims, make minimal "
        "changes, and run relevant tests.\n"
        f"{HIPPO_MEMORY_END}\n"
    )


def codex_doctor(root: str | Path = ".") -> dict[str, Any]:
    requested = Path(root).expanduser().resolve()
    project_root = find_project_root(requested) or requested
    hippo_dir = project_root / HIPPO_DIR_NAME
    memory_file = _find_codex_memory_file(project_root)
    report: dict[str, Any] = {
        "diagnostic": "hippo_codex",
        "read_only": True,
        "requested_root": str(requested),
        "root": str(project_root),
        "db_exists": (hippo_dir / HIPPO_DB_NAME).exists(),
        "project_config_exists": (project_root / ".hippo.toml").exists(),
        "mcp_script_exists": (hippo_dir / "hippo-mcp.ps1").exists(),
        "codex_mcp_config_exists": (hippo_dir / "codex-mcp-config.json").exists(),
        "project_memory": str(memory_file) if memory_file else None,
        "project_memory_has_hippo_block": False,
    }
    if memory_file:
        text = memory_file.read_text(encoding="utf-8", errors="ignore")
        report["project_memory_has_hippo_block"] = (
            HIPPO_MEMORY_START in text and HIPPO_MEMORY_END in text
        )
    report["ready"] = bool(
        report["db_exists"]
        and report["project_config_exists"]
        and report["mcp_script_exists"]
        and report["codex_mcp_config_exists"]
        and report["project_memory_has_hippo_block"]
    )
    report["recommendations"] = [] if report["ready"] else [
        f"Run: hippo codex-deploy --root {project_root}"
    ]
    return report


def project_mcp_database(root: str | Path = ".") -> Database:
    project_root = find_project_root(root)
    if project_root is None:
        db = Database()
        db.initialize()
        return db
    db_path = project_root / HIPPO_DIR_NAME / HIPPO_DB_NAME
    if not db_path.exists():
        db = Database()
        db.initialize()
        return db
    db = Database(db_path)
    db.initialize()
    return db


def find_project_root(cwd: str | Path = ".") -> Path | None:
    current = Path(cwd).expanduser().resolve()
    for path in [current, *current.parents]:
        if (path / HIPPO_DIR_NAME / HIPPO_DB_NAME).exists() or find_project_config(path) == (
            path / ".hippo.toml"
        ):
            return path
    return None


def ensure_gitignore_entry(root: str | Path, entry: str) -> bool:
    path = Path(root).expanduser().resolve() / ".gitignore"
    if path.exists():
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if entry in lines:
            return False
        text = "\n".join(lines)
        if text:
            text += "\n"
        text += entry + "\n"
    else:
        text = entry + "\n"
    path.write_text(text, encoding="utf-8")
    return True


def _find_codex_memory_file(root: Path) -> Path | None:
    for relative in CODEX_MEMORY_FILES:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def _ensure_project_config(root: Path, project: str, *, force: bool) -> bool:
    path = root / ".hippo.toml"
    if path.exists() and not force:
        return False
    write_project_config(root, project, force=force)
    return True


def _upsert_marked_block(text: str, block: str) -> str:
    without_old = _remove_marked_block(text)
    if without_old and not without_old.endswith("\n"):
        without_old += "\n"
    if without_old.strip():
        return without_old.rstrip() + "\n\n" + block
    return block


def _remove_marked_block(text: str) -> str:
    start = text.find(HIPPO_MEMORY_START)
    end = text.find(HIPPO_MEMORY_END)
    if start < 0 or end < start:
        return text
    end += len(HIPPO_MEMORY_END)
    prefix = text[:start].rstrip()
    suffix = text[end:].lstrip()
    if prefix and suffix:
        return prefix + "\n\n" + suffix
    if prefix:
        return prefix + "\n"
    if suffix:
        return suffix
    return ""
