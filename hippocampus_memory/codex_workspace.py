from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hippocampus_memory.db import Database
from hippocampus_memory.deploy import HIPPO_DB_NAME, HIPPO_DIR_NAME, find_project_root
from hippocampus_memory.project_resolver import (
    git_root_path,
    resolve_project_name,
    write_project_config,
)

WORKSPACE_ARGUMENT_KEYS = (
    "workspace_root",
    "codex_workspace_root",
    "project_root",
    "root",
    "cwd",
)
WORKSPACE_ENV_KEYS = (
    "HIPPO_PROJECT_ROOT",
    "HIPPO_WORKSPACE_ROOT",
    "CODEX_WORKSPACE_ROOT",
    "CODEX_PROJECT_ROOT",
    "CODEX_CWD",
    "CODEX_WORKDIR",
)
TRUST_EXACT_ARGUMENT_KEYS = {"workspace_root", "codex_workspace_root", "project_root", "root"}
TRUST_EXACT_ENV_KEYS = {
    "HIPPO_PROJECT_ROOT",
    "HIPPO_WORKSPACE_ROOT",
    "CODEX_WORKSPACE_ROOT",
    "CODEX_PROJECT_ROOT",
    "CODEX_WORKDIR",
}


@dataclass(frozen=True)
class ResolvedCodexWorkspace:
    root: Path
    project: str
    db: Database
    arguments: dict[str, Any]
    scheduler_state_path: Path
    source: str
    auto_created: bool


class CodexWorkspaceResolver:
    """Resolve the active Codex workspace into an isolated Hippo project database."""

    def __init__(
        self,
        *,
        fallback_root: str | Path | None = None,
        auto_create: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.fallback_root = Path(fallback_root).expanduser().resolve() if fallback_root else None
        self.auto_create = auto_create
        self.env = env

    @staticmethod
    def fallback_database(path: str | Path) -> Database:
        db = Database(path)
        db.initialize()
        return db

    def resolve(self, arguments: Mapping[str, Any]) -> ResolvedCodexWorkspace:
        root, source = self._resolve_root(arguments)
        project = self._resolve_project(arguments, root)
        db, auto_created = self._database_for(root, project)
        clean_arguments = {
            key: value
            for key, value in dict(arguments).items()
            if key not in WORKSPACE_ARGUMENT_KEYS
        }
        clean_arguments.setdefault("project", project)
        return ResolvedCodexWorkspace(
            root=root,
            project=project,
            db=db,
            arguments=clean_arguments,
            scheduler_state_path=(root / HIPPO_DIR_NAME / "hippo.scheduler.json"),
            source=source,
            auto_created=auto_created,
        )

    def _resolve_root(self, arguments: Mapping[str, Any]) -> tuple[Path, str]:
        for key in WORKSPACE_ARGUMENT_KEYS:
            root = self._usable_root(
                arguments.get(key),
                trust_exact=key in TRUST_EXACT_ARGUMENT_KEYS,
            )
            if root is not None:
                return root, f"argument:{key}"
        env = self.env if self.env is not None else os.environ
        for key in WORKSPACE_ENV_KEYS:
            root = self._usable_root(env.get(key), trust_exact=key in TRUST_EXACT_ENV_KEYS)
            if root is not None:
                return root, f"env:{key}"
        root = self._usable_root(Path.cwd(), trust_exact=True)
        if root is not None:
            return root, "cwd"
        if self.fallback_root is not None:
            root = self._usable_root(self.fallback_root, trust_exact=True)
            if root is not None:
                return root, "fallback_root"
        raise ValueError("Could not resolve a safe Codex workspace root for hippo_memory")

    def _resolve_project(self, arguments: Mapping[str, Any], root: Path) -> str:
        if arguments.get("project"):
            return str(arguments["project"])
        if self._has_local_project_marker(root):
            return resolve_project_name(cwd=root)
        if self.auto_create:
            return root.name
        return resolve_project_name(cwd=root)

    def _usable_root(self, value: object, *, trust_exact: bool = False) -> Path | None:
        if value is None or str(value).strip() == "":
            return None
        try:
            path = Path(str(value)).expanduser().resolve()
        except OSError:
            return None
        if path.is_file():
            path = path.parent
        if not path.exists() or not path.is_dir():
            return None
        if trust_exact:
            if self._has_local_project_marker(path):
                return path
            if self.auto_create and self._safe_to_auto_create(path):
                return path
        deployed = find_project_root(path)
        if deployed is not None:
            return deployed.resolve()
        if not self.auto_create or not self._safe_to_auto_create(path):
            return None
        git_root = git_root_path(path)
        return (git_root or path).resolve()

    def _has_local_project_marker(self, path: Path) -> bool:
        return (path / ".hippo.toml").exists() or (path / HIPPO_DIR_NAME / HIPPO_DB_NAME).exists()

    def _database_for(self, root: Path, project: str) -> tuple[Database, bool]:
        hippo_dir = root / HIPPO_DIR_NAME
        db_path = hippo_dir / HIPPO_DB_NAME
        auto_created = False
        if not db_path.exists():
            if not self.auto_create:
                return self.fallback_database(db_path), False
            hippo_dir.mkdir(parents=True, exist_ok=True)
            auto_created = True
        if self.auto_create and not (root / ".hippo.toml").exists():
            write_project_config(root, project, force=False)
            auto_created = True
        db = Database(db_path)
        db.initialize()
        db.insert_or_update_project(project, root_path=str(root))
        return db, auto_created

    def _safe_to_auto_create(self, path: Path) -> bool:
        if path.anchor == str(path):
            return False
        if path.name.casefold() in {"windows", "system32"}:
            return False
        windir = os.environ.get("WINDIR")
        if windir:
            try:
                path.relative_to(Path(windir).resolve())
                return False
            except ValueError:
                pass
        return True
