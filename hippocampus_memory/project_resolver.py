from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

CONFIG_NAME = ".hippo.toml"


def resolve_project_name(project: str | None = None, cwd: str | Path | None = None) -> str:
    if project:
        return project
    root = Path(cwd or Path.cwd()).resolve()
    config = find_project_config(root)
    if config:
        data = tomllib.loads(config.read_text(encoding="utf-8"))
        name = data.get("project", {}).get("name")
        if name:
            return str(name)
    git_root = git_root_path(root)
    if git_root:
        return git_root.name
    return root.name


def find_project_config(cwd: str | Path | None = None) -> Path | None:
    current = Path(cwd or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        candidate = path / CONFIG_NAME
        if candidate.exists():
            return candidate
    return None


def git_root_path(cwd: str | Path | None = None) -> Path | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(Path(cwd or Path.cwd())),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return Path(value) if value else None


def write_project_config(root: str | Path, name: str, force: bool = False) -> Path:
    path = Path(root).expanduser().resolve() / CONFIG_NAME
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass force=True to overwrite")
    content = f'[project]\nname = "{_escape_toml(name)}"\n'
    path.write_text(content, encoding="utf-8")
    return path


def _escape_toml(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
