from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CodeDiagnostic:
    relative_path: str
    severity: str
    message: str
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    rule: str | None = None
    source: str = "pyright"


def run_python_diagnostics(
    root_path: str | Path,
    *,
    checker: str | None = None,
    timeout_seconds: int = 45,
) -> dict[str, Any]:
    root = Path(root_path).expanduser().resolve()
    executable = _resolve_checker(checker)
    if executable is None:
        return {
            "available": False,
            "tool": checker or "basedpyright|pyright",
            "diagnostics": [],
            "error": "No basedpyright or pyright executable found on PATH.",
        }
    command = [executable, "--outputjson", str(root)]
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": True,
            "tool": executable,
            "diagnostics": [],
            "error": str(exc),
        }
    diagnostics, parse_error = parse_pyright_output(
        completed.stdout,
        root_path=root,
        source=Path(executable).name,
    )
    return {
        "available": True,
        "tool": executable,
        "returncode": completed.returncode,
        "diagnostics": diagnostics,
        "error": parse_error or (completed.stderr.strip() or None),
    }


def parse_pyright_output(
    output: str,
    *,
    root_path: str | Path,
    source: str = "pyright",
) -> tuple[list[CodeDiagnostic], str | None]:
    if not output.strip():
        return [], "No JSON output from pyright."
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return [], f"Invalid pyright JSON output: {exc}"
    root = Path(root_path).expanduser().resolve()
    diagnostics: list[CodeDiagnostic] = []
    for item in payload.get("generalDiagnostics", []):
        diagnostics.append(_diagnostic_from_json(item, root, source))
    return diagnostics, None


def _diagnostic_from_json(item: dict[str, Any], root: Path, source: str) -> CodeDiagnostic:
    file_path = Path(str(item.get("file") or "")).expanduser()
    try:
        relative_path = file_path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        relative_path = file_path.as_posix()
    range_data = item.get("range") or {}
    start = range_data.get("start") or {}
    end = range_data.get("end") or {}
    return CodeDiagnostic(
        relative_path=relative_path,
        severity=str(item.get("severity") or "information"),
        message=str(item.get("message") or ""),
        line=int(start.get("line", 0)) + 1,
        column=int(start.get("character", 0)) + 1,
        end_line=int(end["line"]) + 1 if "line" in end else None,
        end_column=int(end["character"]) + 1 if "character" in end else None,
        rule=_diagnostic_rule(item),
        source=source,
    )


def _diagnostic_rule(item: dict[str, Any]) -> str | None:
    rule = item.get("rule")
    if rule:
        return str(rule)
    code = item.get("code")
    if isinstance(code, str):
        return code
    if isinstance(code, dict) and code.get("value"):
        return str(code["value"])
    return None


def _resolve_checker(checker: str | None) -> str | None:
    candidates = [checker] if checker else ["basedpyright", "pyright"]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None
