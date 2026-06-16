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
REASONIX_SERVER_NAME = "hippo_memory"
REASONIX_PROJECT_SPEC = f"{REASONIX_SERVER_NAME}=hippo mcp-project"
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
    if install_global:
        reasonix_config = (
            Path(config_path).expanduser() if config_path else default_reasonix_config_path()
        )
        reasonix_config_updated = install_reasonix_mcp_spec(
            reasonix_config,
            REASONIX_PROJECT_SPEC,
            server_name=REASONIX_SERVER_NAME,
        )

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
        "Set-Location $ProjectRoot\n"
        "$Args = @('code', '.', '--mcp', $Spec)\n"
        "if ($Resume) { $Args += '--resume' }\n"
        "if ($Model) { $Args += @('--model', $Model) }\n"
        "reasonix @Args\n"
    )
    path.write_text(script, encoding="utf-8")
    return path


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
        "- The MCP server name is `hippo_memory`; prefer its tools when external project "
        "memory, impact analysis, or compact context would help.\n"
        "- At the start of non-trivial coding/debugging/architecture tasks, decide whether "
        "to call `hippo_memory_context_callback` with the current task intent. Use "
        "`session_key=\"reasonix\"` unless the user gives a better session name.\n"
        "- For broad orientation use `hippo_memory_context_bundle`; before risky edits use "
        "`hippo_memory_project_impact`; for symbol questions use `hippo_memory_code_symbols` "
        "or `hippo_memory_code_references`.\n"
        "- Do not write long-term memories unless the user explicitly asks. Prefer queued "
        "candidates or ask before storing uncertain facts. Do not recall private/sensitive "
        "memories unless explicitly requested.\n"
        "- Keep recalled context short, cite files when making code claims, make minimal "
        "changes, and run relevant tests.\n"
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
        return text[:start].rstrip() + "\n\n" + block.rstrip() + "\n\n" + suffix.lstrip()
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


def project_mcp_database(root: str | Path = ".") -> Database:
    project_root = find_project_root(root)
    if project_root is None:
        raise FileNotFoundError(
            "No .hippo project found. Run `hippo reasonix-deploy` in this project first."
        )
    db_path = project_root / HIPPO_DIR_NAME / HIPPO_DB_NAME
    if not db_path.exists():
        raise FileNotFoundError(
            f"Project memory database not found: {db_path}. "
            "Run `hippo reasonix-deploy` in this project first."
        )
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
