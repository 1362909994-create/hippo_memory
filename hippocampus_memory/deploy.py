from __future__ import annotations

import json
import os
import re
import sys
import time
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
REASONIX_SERVER_NAME = "hippo_memory"
REASONIX_PROJECT_SPEC = f"{REASONIX_SERVER_NAME}=hippo mcp-project"
REASONIX_SHIM_MARKER = "HIPPO_MEMORY_REASONIX_SHIM"
REASONIX_STATUS_PATCH_MARKER = "HIPPO_REASONIX_STATUS_BAR_PATCH"
REASONIX_STATUS_PATCH_VERSION = "v9"
REASONIX_MEMORY_FILES = (
    "REASONIX.md",
    ".claude/CLAUDE.md",
    "CLAUDE.md",
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


def default_reasonix_config_path() -> Path:
    return Path.home() / ".reasonix" / "config.json"


def deploy_reasonix(
    root: str | Path = ".",
    *,
    project: str | None = None,
    config_path: str | Path | None = None,
    install_global: bool = True,
    force_project_config: bool = False,
    index_project: bool = True,
    project_memory: bool = True,
) -> dict[str, Any]:
    """Prepare one project for Reasonix MCP usage.

    The database remains project-local under .hippo/hippo.db. The optional
    Reasonix config entry is only a transport hook that starts hippo in the
    current Reasonix workspace.
    """
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
    fixed_spec = reasonix_fixed_mcp_spec(mcp_script)
    fixed_spec_path = hippo_dir / "reasonix-mcp-spec.txt"
    fixed_spec_path.write_text(fixed_spec + "\n", encoding="utf-8")
    global_spec_path = hippo_dir / "reasonix-global-mcp-spec.txt"
    global_spec_path.write_text(REASONIX_PROJECT_SPEC + "\n", encoding="utf-8")
    reasonix_launcher = write_reasonix_launcher(hippo_dir)
    memory_file = None
    memory_file_updated = False
    if project_memory:
        memory_file, memory_file_updated = ensure_reasonix_project_memory(root_path, project_name)
    gitignore_updated = ensure_gitignore_entry(root_path, f"{HIPPO_DIR_NAME}/")

    reasonix_config = None
    reasonix_config_updated = False
    reasonix_global_memory = None
    reasonix_global_memory_updated = False
    reasonix_shim = None
    if install_global:
        reasonix_config = (
            Path(config_path).expanduser() if config_path else default_reasonix_config_path()
        )
        reasonix_config_updated = install_reasonix_mcp_spec(
            reasonix_config,
            REASONIX_PROJECT_SPEC,
            server_name=REASONIX_SERVER_NAME,
        )
        reasonix_global_memory, reasonix_global_memory_updated = ensure_reasonix_global_memory(
            reasonix_config.parent
        )
        if config_path is None:
            reasonix_shim = install_reasonix_command_shims()

    return {
        "project": project_name,
        "root": str(root_path),
        "db_path": str(db_path),
        "project_config": str(root_path / ".hippo.toml"),
        "project_config_written": config_written,
        "index": index_result,
        "mcp_script": str(mcp_script),
        "reasonix_mcp_spec": REASONIX_PROJECT_SPEC,
        "fixed_project_mcp_spec": fixed_spec,
        "fixed_project_mcp_spec_file": str(fixed_spec_path),
        "global_mcp_spec_file": str(global_spec_path),
        "reasonix_launcher": str(reasonix_launcher),
        "reasonix_project_memory": str(memory_file) if memory_file else None,
        "reasonix_project_memory_updated": memory_file_updated,
        "reasonix_config": str(reasonix_config) if reasonix_config else None,
        "reasonix_config_updated": reasonix_config_updated,
        "reasonix_global_memory": (
            str(reasonix_global_memory) if reasonix_global_memory else None
        ),
        "reasonix_global_memory_updated": reasonix_global_memory_updated,
        "reasonix_shim": reasonix_shim,
        "gitignore_updated": gitignore_updated,
        "next": {
            "auto": f"reasonix code {root_path}",
            "manual": f'reasonix code {root_path} --mcp "{fixed_spec}"',
            "check": f'reasonix mcp inspect "{fixed_spec}" --json',
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


def write_reasonix_launcher(hippo_dir: str | Path) -> Path:
    path = Path(hippo_dir).expanduser().resolve() / "reasonix-with-memory.ps1"
    script = (
        "param([string]$Model = '', [switch]$Resume)\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$ProjectRoot = Split-Path -Parent $PSScriptRoot\n"
        "$SpecPath = Join-Path $PSScriptRoot 'reasonix-mcp-spec.txt'\n"
        "$Spec = (Get-Content -Raw -Path $SpecPath).Trim()\n"
        "$ContextPath = Join-Path $PSScriptRoot 'reasonix-system-append.md'\n"
        "$StatusPath = Join-Path $PSScriptRoot 'reasonix-status.json'\n"
        "hippo reasonix-bootstrap-context --root $ProjectRoot --output $ContextPath "
        "--status-output $StatusPath | Out-Null\n"
        "if (Test-Path -LiteralPath $StatusPath) { "
        "$env:HIPPO_REASONIX_STATUS_FILE = $StatusPath }\n"
        "Set-Location $ProjectRoot\n"
        "$Args = @('code', '.', '--mcp', $Spec, '--system-append-file', $ContextPath)\n"
        "if ($Resume) { $Args += '--resume' }\n"
        "if ($Model) { $Args += @('--model', $Model) }\n"
        "reasonix @Args\n"
    )
    path.write_text(script, encoding="utf-8")
    return path


def write_reasonix_bootstrap_context(
    root: str | Path = ".",
    output: str | Path | None = None,
    *,
    intent: str = "project overview and coding session bootstrap",
    status_output: str | Path | None = None,
    auto_project: bool = True,
) -> Path:
    root_path = Path(root).expanduser().resolve()
    output_path = (
        Path(output).expanduser()
        if output
        else root_path / HIPPO_DIR_NAME / "reasonix-system-append.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if auto_project:
        ensure_reasonix_bootstrap_project(root_path)
    project = resolve_project_name(cwd=root_path)
    token_savings: dict[str, Any] | None = None
    try:
        from hippocampus_memory.recall_policy import build_auto_context

        payload = build_auto_context(
            project_mcp_database(root_path),
            intent=intent,
            project=project,
            session_key="reasonix-bootstrap",
            track_token_savings=True,
            include_savings_in_text=True,
        )
        body = str(payload["text"])
        if isinstance(payload.get("token_savings"), dict):
            token_savings = payload["token_savings"]
    except Exception as exc:  # pragma: no cover - defensive launcher fallback
        body = f"Hippocampus bootstrap context unavailable: {exc}"
    text = (
        "Hippocampus Memory bootstrap for Reasonix.\n"
        "Use the context below before reading project files.\n"
        "If a token savings line is present, include it visibly in the final user-facing "
        "Reasonix UI reply.\n\n"
        f"{body}\n"
    )
    output_path.write_text(text, encoding="utf-8")
    if status_output:
        write_reasonix_status_file(
            status_output,
            token_savings,
            project=project,
            root=root_path,
        )
    return output_path


def ensure_reasonix_bootstrap_project(
    root: str | Path = ".",
    *,
    project: str | None = None,
    index_project: bool = True,
) -> dict[str, Any]:
    """Create the minimal project-local store needed by the Reasonix shim.

    This is intentionally smaller than ``deploy_reasonix``: opening Reasonix in
    a new folder should create the DB/index needed for context and status-bar
    accounting, but it should not rewrite Reasonix project prompt files.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        return {"ready": False, "reason": "root_not_directory", "root": str(root_path)}

    project_root = find_project_root(root_path) or root_path
    skip_reason = reasonix_auto_project_skip_reason(project_root)
    if skip_reason:
        return {
            "ready": False,
            "reason": skip_reason,
            "root": str(project_root),
        }

    project_name = project or resolve_project_name(cwd=project_root)
    hippo_dir = project_root / HIPPO_DIR_NAME
    db_path = hippo_dir / HIPPO_DB_NAME
    db_existed = db_path.exists()

    try:
        hippo_dir.mkdir(parents=True, exist_ok=True)
        config_written = _ensure_project_config(project_root, project_name, force=False)
        db = Database(db_path)
        db.initialize()
        db.insert_or_update_project(project_name, root_path=str(project_root))
        has_indexed_files = _project_has_indexed_files(db, project_name)
        should_index = index_project and (not db_existed or not has_indexed_files)
        index_result = (
            ProjectIndexer(db).index_project(project_root, project_name)
            if should_index
            else {"indexed_files": 0, "skipped_files": 0, "stale_files": 0}
        )
        try:
            gitignore_updated = ensure_gitignore_entry(project_root, f"{HIPPO_DIR_NAME}/")
        except OSError:
            gitignore_updated = False
    except Exception as exc:  # pragma: no cover - defensive shim bootstrap
        return {
            "ready": False,
            "reason": "project_bootstrap_failed",
            "error": str(exc),
            "root": str(project_root),
        }

    return {
        "ready": True,
        "root": str(project_root),
        "project": project_name,
        "db_path": str(db_path),
        "db_created": not db_existed,
        "project_config_written": config_written,
        "index": index_result,
        "gitignore_updated": gitignore_updated,
    }


def reasonix_auto_project_skip_reason(root: str | Path) -> str | None:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        return "root_not_directory"
    if root_path.parent == root_path:
        return "drive_root"
    home = Path.home().resolve()
    if root_path == home:
        return "home_directory"
    windir_raw = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if windir_raw:
        windir = Path(windir_raw).expanduser().resolve()
        if root_path == windir or windir in root_path.parents:
            return "windows_system_directory"
    return None


def _project_has_indexed_files(db: Database, project: str) -> bool:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM files WHERE project = ? AND status = 'active'",
            (project,),
        ).fetchone()
    return bool(row and int(row[0]) > 0)


def write_reasonix_status_file(
    output: str | Path,
    token_savings: dict[str, Any] | None,
    *,
    project: str | None = None,
    root: str | Path | None = None,
) -> Path:
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = f"{path.stem}-{os.getpid()}-{int(time.time() * 1000)}"
    session_ledger_dir = reasonix_session_savings_dir(root) if root is not None else None
    if not token_savings:
        path.write_text(
            json.dumps(
                {
                    "available": True,
                    "scope": "reasonix_session",
                    "run_id": run_id,
                    "project": project,
                    "root": str(root) if root is not None else None,
                    "session_ledger_dir": (
                        str(session_ledger_dir) if session_ledger_dir else None
                    ),
                    "saved_tokens": 0,
                    "session_saved_tokens": 0,
                    "total_saved_tokens": 0,
                    "project_total_saved_tokens": 0,
                    "baseline_tokens": 0,
                    "output_tokens": 0,
                    "savings_ratio": 0.0,
                    "average_savings_ratio": 0.0,
                    "text": "",
                    "reason": "no_token_savings_available",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path
    saved_tokens = int(token_savings.get("saved_tokens") or 0)
    project_total_saved = int(token_savings.get("total_saved_tokens") or 0)
    payload = {
        "available": True,
        "scope": "reasonix_session",
        "run_id": run_id,
        "project": project,
        "root": str(root) if root is not None else None,
        "session_ledger_dir": str(session_ledger_dir) if session_ledger_dir else None,
        "saved_tokens": saved_tokens,
        "session_saved_tokens": 0,
        "total_saved_tokens": 0,
        "project_total_saved_tokens": project_total_saved,
        "baseline_tokens": int(token_savings.get("baseline_tokens") or 0),
        "output_tokens": int(token_savings.get("output_tokens") or 0),
        "savings_ratio": float(token_savings.get("savings_ratio") or 0.0),
        "average_savings_ratio": float(token_savings.get("average_savings_ratio") or 0.0),
        "text": token_savings.get("token_savings_text") or "",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def reasonix_session_savings_dir(root: str | Path = ".") -> Path:
    project_root = find_project_root(root)
    if project_root is not None:
        return project_root / HIPPO_DIR_NAME / "reasonix-session-savings"
    local_appdata = os.environ.get("LOCALAPPDATA")
    base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
    return base / "hippo_memory" / "reasonix-session-savings"


def ensure_reasonix_project_memory(root: str | Path, project: str) -> tuple[Path, bool]:
    root_path = Path(root).expanduser().resolve()
    target = _find_reasonix_memory_file(root_path) or (root_path / "REASONIX.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    block = reasonix_project_memory_block(project)
    if target.exists():
        old = target.read_text(encoding="utf-8", errors="ignore")
    else:
        old = ""
    new = _upsert_marked_block(old, block)
    if new == old:
        return target, False
    target.write_text(new, encoding="utf-8")
    return target, True


def reasonix_project_memory_block(project: str) -> str:
    return (
        f"{HIPPO_MEMORY_START}\n"
        "## Hippocampus Memory\n\n"
        f"- This project is deployed with project-local hippocampus-memory as `{project}`.\n"
        "- The MCP server name is `hippo_memory`; prefer its automatic tools when external "
        "project memory, impact analysis, or compact context would help.\n"
        "- At the start of non-trivial coding/debugging/architecture tasks, call "
        "`hippo_memory_context_auto` with the current task intent. Use "
        "`session_key=\"reasonix\"` unless the user gives a better session name. Trust the "
        "tool when it returns that no external memory is needed.\n"
        "- When `hippo_memory_context_auto` returns `token_savings_text`, or its text says "
        "`Show this token savings line to the user:`, include that token savings line in "
        "the final user-facing Reasonix UI reply. Do not leave it hidden in tool output "
        "or reasoning only.\n"
        "- For direct symbol questions use `hippo_memory_code_symbols` or "
        "`hippo_memory_code_references`; otherwise prefer `hippo_memory_context_auto` over "
        "manually choosing profile, impact, callback, or bundle tools.\n"
        "- Near the end of a meaningful session, call `hippo_memory_memory_auto_store` with "
        "a concise transcript summary. It will write high-confidence non-sensitive memories, "
        "queue uncertain memories, and skip low-value content.\n"
        "- Do not recall private/sensitive memories unless explicitly requested. Do not force "
        "long-term writes for sensitive or uncertain facts.\n"
        "- Keep recalled context short, cite files when making code claims, make minimal "
        "changes, and run relevant tests.\n"
        f"{HIPPO_MEMORY_END}\n"
    )


def ensure_reasonix_global_memory(reasonix_dir: str | Path | None = None) -> tuple[Path, bool]:
    target = (
        Path(reasonix_dir).expanduser()
        if reasonix_dir
        else default_reasonix_config_path().parent
    )
    target.mkdir(parents=True, exist_ok=True)
    path = target / "REASONIX.md"
    old = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    new = _upsert_marked_block(old, reasonix_global_memory_block())
    if new == old:
        return path, False
    path.write_text(new, encoding="utf-8")
    return path, True


def reasonix_global_memory_block() -> str:
    return (
        f"{HIPPO_MEMORY_START}\n"
        "## Hippocampus Memory\n\n"
        "- A global Reasonix MCP server named `hippo_memory` is installed for automatic "
        "external memory and compact context.\n"
        "- At the start of non-trivial coding/debugging/project-review/architecture tasks, "
        "call `hippo_memory_context_auto` before reading lots of files. Pass the current "
        "task intent and the project name when you can infer it; otherwise omit `project` "
        "and let the MCP server use its default project.\n"
        "- When `hippo_memory_context_auto` returns `token_savings_text`, or its text says "
        "`Show this token savings line to the user:`, include that token savings line in "
        "the final user-facing Reasonix UI reply.\n"
        "- Near the end of meaningful work, call `hippo_memory_memory_auto_store` with a "
        "concise transcript summary so durable, non-sensitive facts can be written or queued.\n"
        "- Do not recall private/sensitive memories unless explicitly requested. Do not force "
        "long-term writes for sensitive or uncertain facts.\n"
        f"{HIPPO_MEMORY_END}\n"
    )


def _find_reasonix_memory_file(root: Path) -> Path | None:
    for relative in REASONIX_MEMORY_FILES:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def _upsert_marked_block(text: str, block: str) -> str:
    start = text.find(HIPPO_MEMORY_START)
    end = text.find(HIPPO_MEMORY_END)
    if start >= 0 and end >= start:
        end += len(HIPPO_MEMORY_END)
        suffix = text[end:]
        if suffix.startswith("\n"):
            suffix = suffix[1:]
        prefix = text[:start].rstrip()
        replacement = block.rstrip()
        if prefix:
            replacement = prefix + "\n\n" + replacement
        if suffix.strip():
            replacement += "\n\n" + suffix.lstrip()
        else:
            replacement += "\n"
        return replacement
    if not text.strip():
        return block
    return text.rstrip() + "\n\n" + block


def reasonix_fixed_mcp_spec(
    script_path: str | Path,
    *,
    server_name: str = REASONIX_SERVER_NAME,
) -> str:
    quoted = _quote_mcp_arg(str(Path(script_path).expanduser().resolve()))
    return f"{server_name}=powershell.exe -NoProfile -ExecutionPolicy Bypass -File {quoted}"


def install_reasonix_mcp_spec(
    config_path: str | Path,
    spec: str = REASONIX_PROJECT_SPEC,
    *,
    server_name: str = REASONIX_SERVER_NAME,
) -> bool:
    path = Path(config_path).expanduser()
    cfg = _read_json_object(path)
    raw_mcp = cfg.get("mcp")
    mcp = [item for item in raw_mcp if isinstance(item, str)] if isinstance(raw_mcp, list) else []
    kept = [item for item in mcp if not _mcp_spec_has_name(item, server_name)]
    new_mcp = [*kept, spec]
    raw_disabled = cfg.get("mcpDisabled")
    if isinstance(raw_disabled, list):
        new_disabled = [item for item in raw_disabled if item != server_name]
    else:
        new_disabled = raw_disabled
    changed = mcp != new_mcp or raw_mcp != new_mcp or raw_disabled != new_disabled
    if not changed:
        return False
    cfg["mcp"] = new_mcp
    if new_disabled:
        cfg["mcpDisabled"] = new_disabled
    else:
        cfg.pop("mcpDisabled", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def default_reasonix_bin_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "npm"
    return Path.home() / "AppData" / "Roaming" / "npm"


def install_reasonix_command_shims(bin_dir: str | Path | None = None) -> dict[str, Any]:
    target = Path(bin_dir).expanduser() if bin_dir else default_reasonix_bin_dir()
    target.mkdir(parents=True, exist_ok=True)
    ps1_path = target / "reasonix.ps1"
    cmd_path = target / "reasonix.cmd"
    sh_path = target / "reasonix"
    ps1_updated = _write_reasonix_shim_file(ps1_path, _reasonix_ps1_shim())
    cmd_updated = _write_reasonix_shim_file(cmd_path, _reasonix_cmd_shim())
    sh_updated = _write_reasonix_shim_file(sh_path, _reasonix_sh_shim())
    status_bar_patch = patch_reasonix_status_bar(target)
    return {
        "bin_dir": str(target),
        "ps1": str(ps1_path),
        "cmd": str(cmd_path),
        "sh": str(sh_path),
        "ps1_updated": ps1_updated,
        "cmd_updated": cmd_updated,
        "sh_updated": sh_updated,
        "status_bar_patch": status_bar_patch,
    }


def patch_reasonix_status_bar(bin_dir: str | Path | None = None) -> dict[str, Any]:
    """Patch the installed Reasonix TUI status row to show Hippo savings.

    Reasonix exposes token/cost data only inside its bundled Ink UI. The Hippo
    shim writes a tiny JSON status file, and this patch teaches the UI to read
    that file and render one extra status-bar pill.
    """
    target = Path(bin_dir).expanduser() if bin_dir else default_reasonix_bin_dir()
    cli_dir = target / "node_modules" / "reasonix" / "dist" / "cli"
    if not cli_dir.exists():
        return {
            "bin_dir": str(target),
            "patched": False,
            "reason": "reasonix_cli_dir_not_found",
        }
    candidates = sorted(cli_dir.glob("chunk-*.js"))
    for candidate in candidates:
        old = candidate.read_text(encoding="utf-8", errors="ignore")
        if (
            "function StatusRow({" in old
            and "statusBar.showCtxUsage" in old
            and "formatTokens" in old
        ):
            return _patch_reasonix_status_bar_file(candidate, old)
    return {
        "bin_dir": str(target),
        "patched": False,
        "reason": "status_bar_chunk_not_found",
    }


def _patch_reasonix_status_bar_file(path: Path, old: str) -> dict[str, Any]:
    backup = path.with_suffix(path.suffix + ".hippo-status-original")
    if f"{REASONIX_STATUS_PATCH_MARKER} {REASONIX_STATUS_PATCH_VERSION}" in old:
        return {
            "status_bar_file": str(path),
            "patched": False,
            "reason": "already_patched",
        }
    if REASONIX_STATUS_PATCH_MARKER in old:
        if not backup.exists():
            return {
                "status_bar_file": str(path),
                "patched": False,
                "reason": "patched_source_backup_missing",
            }
        old = backup.read_text(encoding="utf-8", errors="ignore")
    react_match = re.search(
        r"function Pill\(\{ children \}\) \{\s+return /\* @__PURE__ \*/ "
        r"(import_react\d+)\.default\.createElement",
        old,
    )
    react_name = react_match.group(1) if react_match else "import_react16"
    helper = _reasonix_status_bar_patch_helper(react_name)
    status_anchor = "function StatusRow({\n"
    if status_anchor not in old:
        return {
            "status_bar_file": str(path),
            "patched": False,
            "reason": "status_row_anchor_not_found",
        }
    new = old.replace(status_anchor, helper + "\n" + status_anchor, 1)
    cache_anchor = (
        '`${t("statusBar.cache")} ${Math.round(status2.cacheHit * 100)}%`))), '
        "statusBar.showCtxUsage"
    )
    if cache_anchor not in new:
        return {
            "status_bar_file": str(path),
            "patched": False,
            "reason": "cache_context_anchor_not_found",
        }
    new = new.replace(
        cache_anchor,
        cache_anchor.replace(
            "statusBar.showCtxUsage",
            f"{react_name}.default.createElement("
            "HippoSavingsPill, { "
            "sessionId: session.id, "
            "promptTokens: status2.promptTokens, "
            "turnCost: status2.cost, "
            "workspace: session.workspace "
            "}), "
            "statusBar.showCtxUsage",
        ),
        1,
    )
    if not backup.exists():
        backup.write_text(old, encoding="utf-8")
    path.write_text(new, encoding="utf-8")
    return {
        "status_bar_file": str(path),
        "backup": str(backup),
        "patched": True,
    }


def _reasonix_status_bar_patch_helper(react_name: str) -> str:
    return f"""// {REASONIX_STATUS_PATCH_MARKER} {REASONIX_STATUS_PATCH_VERSION}
function hippoSafeSessionFileName(value) {{
  const raw = String(value || "unknown");
  const safe = raw.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 160);
  return safe || "unknown";
}}
function readHippoJsonFile(fs, file) {{
  if (!file || !fs.existsSync(file)) return null;
  return JSON.parse(fs.readFileSync(file, "utf8"));
}}
function writeHippoJsonFile(fs, file, data) {{
  fs.writeFileSync(file, JSON.stringify(data), "utf8");
}}
function hippoSamePath(a, b) {{
  return String(a || "").trim().toLowerCase() === String(b || "").trim().toLowerCase();
}}
function shouldRefreshHippoStatusForWorkspace(data, workspace) {{
  const root = String(workspace || "").trim();
  if (!root) return false;
  if (data && (hippoSamePath(data.root, root) || hippoSamePath(data.workspace_root, root))) {{
    return false;
  }}
  const baseline = Number(data && data.baseline_tokens || 0);
  return !data || !data.project || data.reason === "no_token_savings_available" || baseline <= 0;
}}
function refreshHippoReasonixStatusForWorkspace(requireFn, fs, path, file, workspace) {{
  try {{
    const root = String(workspace || "").trim();
    if (!root || !fs.existsSync(root) || !fs.statSync(root).isDirectory()) return null;
    const globalKey = "__HIPPO_REASONIX_WORKSPACE_REFRESHED__";
    const cache = globalThis[globalKey] || (globalThis[globalKey] = {{}});
    const key = `${{file}}|${{root}}`;
    if (cache[key]) return null;
    cache[key] = true;
    const childProcess = requireFn("child_process");
    const contextDir = path.dirname(file);
    const output = path.join(contextDir, `system-append-ui-${{process.pid}}.md`);
    const result = childProcess.spawnSync(
      "hippo",
      [
        "reasonix-bootstrap-context",
        "--root",
        root,
        "--output",
        output,
        "--status-output",
        file
      ],
      {{ windowsHide: true, encoding: "utf8", timeout: 2e4 }}
    );
    if (result.error || result.status !== 0) return null;
    const refreshed = readHippoJsonFile(fs, file);
    if (refreshed) {{
      refreshed.workspace_root = root;
      writeHippoJsonFile(fs, file, refreshed);
    }}
    return refreshed;
  }} catch {{
    return null;
  }}
}}
function readHippoReasonixStatus(sessionId, promptTokens, turnCost, workspace) {{
  try {{
    const file = process.env.HIPPO_REASONIX_STATUS_FILE;
    if (!file) return null;
    const requireFn = globalThis.require;
    if (!requireFn) return null;
    const fs = requireFn("fs");
    const path = requireFn("path");
    let data = readHippoJsonFile(fs, file);
    if (shouldRefreshHippoStatusForWorkspace(data, workspace)) {{
      data = refreshHippoReasonixStatusForWorkspace(requireFn, fs, path, file, workspace) || data;
    }}
    if (!data || !data.available) return null;
    const run = Number(data.saved_tokens || 0);
    if (!Number.isFinite(run)) return null;
    let sessionTotal = 0;
    let lastSaved = 0;
    let contextCount = 0;
    const sessionKey = sessionId ? hippoSafeSessionFileName(sessionId) : null;
    const ledgerDir = data.session_ledger_dir;
    if (sessionKey && ledgerDir) {{
      fs.mkdirSync(ledgerDir, {{ recursive: true }});
      const ledgerFile = path.join(ledgerDir, `${{sessionKey}}.json`);
      const existing = readHippoJsonFile(fs, ledgerFile) || {{}};
      const runId = String(data.run_id || file);
      const trackingMode = "reasonix_context_runs_v1";
      const currentLedger = existing.tracking_mode === trackingMode;
      const runs = currentLedger && Array.isArray(existing.runs) ? existing.runs : [];
      const ledger = {{
        ...(currentLedger ? existing : {{
          legacy_tracking_mode: existing.tracking_mode || null,
          legacy_saved_tokens: Number(existing.saved_tokens || 0)
        }}),
        project: data.project || existing.project || null,
        session_id: sessionId,
        saved_tokens: currentLedger ? Number(existing.saved_tokens || 0) : 0,
        baseline_tokens: currentLedger ? Number(existing.baseline_tokens || 0) : 0,
        output_tokens: currentLedger ? Number(existing.output_tokens || 0) : 0,
        last_saved_tokens: currentLedger ? Number(existing.last_saved_tokens || 0) : 0,
        context_count: currentLedger ? Number(existing.context_count || 0) : 0,
        runs,
        tracking_mode: trackingMode
      }};
      if (runId && !ledger.runs.includes(runId)) {{
        ledger.runs = [...ledger.runs, runId];
        ledger.saved_tokens += run;
        ledger.baseline_tokens += Number(data.baseline_tokens || 0);
        ledger.output_tokens += Number(data.output_tokens || 0);
        ledger.last_saved_tokens = run;
        ledger.last_run_id = runId;
        ledger.context_count += 1;
        ledger.updated_at = new Date().toISOString();
        writeHippoJsonFile(fs, ledgerFile, ledger);
      }}
      sessionTotal = Number(ledger.saved_tokens || 0);
      lastSaved = run;
      contextCount = Number(ledger.context_count || 0);
    }}
    if (!Number.isFinite(sessionTotal)) sessionTotal = 0;
    if (!Number.isFinite(lastSaved)) lastSaved = 0;
    if (!Number.isFinite(contextCount)) contextCount = 0;
    return {{ run: lastSaved, sessionTotal, contextCount }};
  }} catch {{
    return null;
  }}
}}
function HippoSavingsPill({{ sessionId, promptTokens, turnCost, workspace }}) {{
  const data = readHippoReasonixStatus(sessionId, promptTokens, turnCost, workspace);
  if (!data) return null;
  return /* @__PURE__ */ {react_name}.default.createElement(
    {react_name}.default.Fragment,
    null,
    /* @__PURE__ */ {react_name}.default.createElement(Gap, null),
    /* @__PURE__ */ {react_name}.default.createElement(
      Pill,
      null,
      /* @__PURE__ */ {react_name}.default.createElement(
        Text,
        {{ color: TONE.ok, wrap: "truncate" }},
        "记忆节省 "
      ),
      /* @__PURE__ */ {react_name}.default.createElement(
        Text,
        {{ bold: true, color: TONE.ok, wrap: "truncate" }},
        formatTokens(data.run)
      ),
      /* @__PURE__ */ {react_name}.default.createElement(
        Text,
        {{ color: FG.faint, wrap: "truncate" }},
        ` / 会话 ${{formatTokens(data.sessionTotal)}}`
      )
    )
  );
}}"""


def _write_reasonix_shim_file(path: Path, content: str) -> bool:
    old = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    if old == content:
        return False
    backup = path.with_suffix(path.suffix + ".hippo-original")
    if old and REASONIX_SHIM_MARKER not in old and not backup.exists():
        backup.write_text(old, encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    return True


def _reasonix_cmd_shim() -> str:
    return (
        "@ECHO off\n"
        f"REM {REASONIX_SHIM_MARKER} v1\n"
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0reasonix.ps1" %*\n'
    )


def _reasonix_sh_shim() -> str:
    return f"""#!/bin/sh
# {REASONIX_SHIM_MARKER} v1
basedir=$(dirname "$(echo "$0" | sed -e 's,\\\\,/,g')")
script="$basedir/reasonix.ps1"
case `uname` in
  *CYGWIN*|*MINGW*|*MSYS*)
    if command -v cygpath > /dev/null 2>&1; then
      script=`cygpath -w "$script"`
    fi
  ;;
esac
exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$script" "$@"
"""


def _reasonix_ps1_shim() -> str:
    return f"""#!/usr/bin/env pwsh
# {REASONIX_SHIM_MARKER} v1
$basedir = Split-Path $MyInvocation.MyCommand.Definition -Parent
$exe = ""
if ($PSVersionTable.PSVersion -lt "6.0" -or $IsWindows) {{
  $exe = ".exe"
}}
if (Test-Path "$basedir/node$exe") {{
  $node = "$basedir/node$exe"
}} else {{
  $node = "node$exe"
}}
$reasonixIndex = Join-Path $basedir "node_modules/reasonix/dist/cli/index.js"

function Invoke-ReasonixOriginal([string[]]$ArgList) {{
  if ($MyInvocation.ExpectingInput) {{
    $input | & $node $reasonixIndex @ArgList
  }} else {{
    & $node $reasonixIndex @ArgList
  }}
  exit $LASTEXITCODE
}}

function Test-HasSystemAppend([string[]]$ArgList) {{
  foreach ($item in $ArgList) {{
    if ($item -eq "--system-append" -or $item -eq "--system-append-file") {{
      return $true
    }}
  }}
  return $false
}}

function Test-IsBareCodeFlag([string]$Item) {{
  return $Item -eq "-c" -or $Item -eq "--continue" -or
    $Item -eq "--no-mouse" -or $Item -eq "--no-proxy"
}}

function Test-IsHelpFlag([string[]]$ArgList) {{
  foreach ($item in $ArgList) {{
    if ($item -eq "-h" -or $item -eq "--help") {{ return $true }}
  }}
  return $false
}}

function Test-HasSessionMode([string[]]$ArgList) {{
  foreach ($item in $ArgList) {{
    if (
      $item -eq "-c" -or $item -eq "--continue" -or
      $item -eq "-r" -or $item -eq "--resume" -or
      $item -eq "-n" -or $item -eq "--new" -or
      $item -eq "--no-session"
    ) {{
      return $true
    }}
  }}
  return $false
}}

function Add-NewSessionDefault([string[]]$ArgList) {{
  if (Test-IsHelpFlag $ArgList) {{ return $ArgList }}
  if (Test-HasSessionMode $ArgList) {{ return $ArgList }}
  return @($ArgList + "--new")
}}

function Test-IsUnsafeCodeRoot([string]$Path) {{
  try {{
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
  }} catch {{
    return $true
  }}
  $trimmed = $resolved.TrimEnd("\\")
  $driveRoot = ([System.IO.Path]::GetPathRoot($resolved)).TrimEnd("\\")
  if ($trimmed -eq $driveRoot) {{ return $true }}
  $userHome = [Environment]::GetFolderPath("UserProfile").TrimEnd("\\")
  if ($trimmed -ieq $userHome) {{ return $true }}
  $windir = $env:WINDIR
  if (-not $windir) {{ $windir = $env:SystemRoot }}
  if ($windir) {{
    $win = $windir.TrimEnd("\\")
    $underWin = $trimmed.StartsWith(
      $win + "\\",
      [StringComparison]::OrdinalIgnoreCase
    )
    if ($trimmed -ieq $win -or $underWin) {{
      return $true
    }}
  }}
  return $false
}}

function Get-ReasonixWorkspaceFromMeta([string]$Path) {{
  try {{
    $raw = Get-Content -Raw -LiteralPath $Path
  }} catch {{
    return $null
  }}
  try {{
    $meta = $raw | ConvertFrom-Json
    if ($meta.workspace) {{ return [string]$meta.workspace }}
  }} catch {{
  }}
  $match = [regex]::Match($raw, '"workspace"\\s*:\\s*"((?:\\\\.|[^"])*)"')
  if ($match.Success) {{
    return $match.Groups[1].Value.Replace("\\\\", "\\")
  }}
  return $null
}}

function Get-RecentReasonixWorkspace {{
  $sessionDir = Join-Path $HOME ".reasonix/sessions"
  if (-not (Test-Path -LiteralPath $sessionDir)) {{ return $null }}
  $files = Get-ChildItem `
    -LiteralPath $sessionDir `
    -Filter "*.meta.json" `
    -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending
  foreach ($file in $files) {{
    try {{
      $workspace = Get-ReasonixWorkspaceFromMeta $file.FullName
      $exists = $workspace -and (Test-Path -LiteralPath $workspace)
      if ($exists -and -not (Test-IsUnsafeCodeRoot $workspace)) {{
        return (Resolve-Path -LiteralPath $workspace).Path
      }}
    }} catch {{
    }}
  }}
  return $null
}}

function Resolve-DefaultCodeRoot {{
  $cwd = (Get-Location).Path
  if (-not (Test-IsUnsafeCodeRoot $cwd)) {{ return $cwd }}
  $recent = Get-RecentReasonixWorkspace
  if ($recent) {{ return $recent }}
  return $cwd
}}

function Test-ShouldInject([string[]]$ArgList) {{
  if ($ArgList.Count -eq 0) {{ return $true }}
  if ($ArgList[0] -eq "code") {{ return $true }}
  if ($ArgList[0].StartsWith("-")) {{
    foreach ($item in $ArgList) {{
      if (-not (Test-IsBareCodeFlag $item)) {{ return $false }}
    }}
    return $true
  }}
  return $false
}}

function Convert-ToCodeArgs([string[]]$ArgList) {{
  if ($ArgList.Count -eq 0) {{
    return @("code", (Resolve-DefaultCodeRoot), "--new")
  }}
  if ($ArgList[0] -eq "code") {{ return Add-NewSessionDefault $ArgList }}
  $out = @("code", (Resolve-DefaultCodeRoot))
  foreach ($item in $ArgList) {{
    if ($item -eq "-c" -or $item -eq "--continue") {{
      $out += "--resume"
    }} else {{
      $out += $item
    }}
  }}
  return Add-NewSessionDefault $out
}}

function Get-CodeRoot([string[]]$ArgList) {{
  if ($ArgList.Count -ge 2 -and $ArgList[0] -eq "code" -and -not $ArgList[1].StartsWith("-")) {{
    try {{
      return (Resolve-Path -LiteralPath $ArgList[1]).Path
    }} catch {{
      return $ArgList[1]
    }}
  }}
  return (Get-Location).Path
}}

$argList = @($args)
if (Test-ShouldInject $argList) {{
  $argList = Convert-ToCodeArgs $argList
  if (-not (Test-HasSystemAppend $argList)) {{
    $root = Get-CodeRoot $argList
    $contextDir = Join-Path ([System.IO.Path]::GetTempPath()) "hippo-reasonix"
    New-Item -ItemType Directory -Force -Path $contextDir | Out-Null
    $contextPath = Join-Path $contextDir ("system-append-" + $PID + ".md")
    $statusPath = Join-Path $contextDir ("status-" + $PID + ".json")
    try {{
      $bootstrapArgs = @(
        "reasonix-bootstrap-context",
        "--root",
        $root,
        "--output",
        $contextPath,
        "--status-output",
        $statusPath
      )
      & hippo @bootstrapArgs *> $null
    }} catch {{
    }}
    if (Test-Path -LiteralPath $statusPath) {{
      $env:HIPPO_REASONIX_STATUS_FILE = $statusPath
    }}
    if (Test-Path -LiteralPath $contextPath) {{
      $argList += @("--system-append-file", $contextPath)
    }}
  }}
}}

Invoke-ReasonixOriginal $argList
"""


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


def _ensure_project_config(root: Path, project: str, *, force: bool) -> bool:
    path = root / ".hippo.toml"
    if path.exists() and not force:
        return False
    write_project_config(root, project, force=force)
    return True


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8-sig")
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Reasonix config must be a JSON object: {path}")
    return parsed


def _mcp_spec_has_name(spec: str, name: str) -> bool:
    return spec.strip().startswith(f"{name}=")


def _quote_mcp_arg(value: str) -> str:
    if not any(ch.isspace() or ch == '"' for ch in value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
