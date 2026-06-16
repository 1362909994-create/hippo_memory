from __future__ import annotations

import subprocess
from pathlib import Path


def git_snapshot(cwd: str | Path | None = None) -> dict[str, object]:
    root = Path(cwd or Path.cwd())
    if not (root / ".git").exists():
        found = _run_git(["rev-parse", "--show-toplevel"], root)
        if found.returncode != 0:
            return {"available": False, "reason": "not a git repository"}
        root = Path(found.stdout.strip())
    status = _run_git(["status", "--short"], root)
    branch = _run_git(["branch", "--show-current"], root)
    diff = _run_git(["diff", "--stat"], root)
    staged = _run_git(["diff", "--cached", "--stat"], root)
    return {
        "available": True,
        "root": str(root),
        "branch": branch.stdout.strip(),
        "status_short": status.stdout.strip(),
        "diff_stat": diff.stdout.strip(),
        "staged_diff_stat": staged.stdout.strip(),
    }


def format_git_snapshot(snapshot: dict[str, object]) -> str:
    if not snapshot.get("available"):
        return f"Git: unavailable ({snapshot.get('reason', 'unknown')})"
    lines = [
        f"Git root: {snapshot.get('root')}",
        f"Branch: {snapshot.get('branch') or 'unknown'}",
    ]
    if snapshot.get("status_short"):
        lines.append("Status:")
        lines.extend(f"  {line}" for line in str(snapshot["status_short"]).splitlines()[:20])
    if snapshot.get("diff_stat"):
        lines.append("Diff stat:")
        lines.extend(f"  {line}" for line in str(snapshot["diff_stat"]).splitlines()[:20])
    if snapshot.get("staged_diff_stat"):
        lines.append("Staged diff stat:")
        lines.extend(f"  {line}" for line in str(snapshot["staged_diff_stat"]).splitlines()[:20])
    return "\n".join(lines)


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
