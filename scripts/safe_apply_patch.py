from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

OperationKind = Literal["add", "delete", "update"]


class SafePatchError(RuntimeError):
    """Raised when a patch is unsafe or cannot be applied exactly."""


@dataclass(frozen=True)
class UpdateHunk:
    old: str
    new: str


@dataclass(frozen=True)
class PatchOperation:
    kind: OperationKind
    path: str
    content: str = ""
    hunks: tuple[UpdateHunk, ...] = ()


@dataclass(frozen=True)
class PatchResult:
    changed_files: list[Path]
    dry_run: bool = False
    backups: list[Path] = field(default_factory=list)


def _is_operation_header(line: str) -> bool:
    return (
        line.startswith("*** Add File: ")
        or line.startswith("*** Delete File: ")
        or line.startswith("*** Update File: ")
        or line == "*** End Patch"
    )


def _line_payload(line: str) -> str:
    if not line:
        raise SafePatchError("patch change lines must start with '+', '-', or space")
    marker = line[0]
    if marker not in {"+", "-", " "}:
        raise SafePatchError(f"invalid patch change line: {line!r}")
    return line[1:] + "\n"


def parse_patch(patch_text: str) -> list[PatchOperation]:
    lines = patch_text.splitlines()
    if not lines or lines[0] != "*** Begin Patch":
        raise SafePatchError("patch must start with '*** Begin Patch'")
    if lines[-1] != "*** End Patch":
        raise SafePatchError("patch must end with '*** End Patch'")

    operations: list[PatchOperation] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if line.startswith("*** Add File: "):
            path = line.removeprefix("*** Add File: ").strip()
            index += 1
            content_parts: list[str] = []
            while index < len(lines) and not _is_operation_header(lines[index]):
                change_line = lines[index]
                if not change_line.startswith("+"):
                    raise SafePatchError("add-file content lines must start with '+'")
                content_parts.append(change_line[1:] + "\n")
                index += 1
            operations.append(PatchOperation("add", path, content="".join(content_parts)))
            continue

        if line.startswith("*** Delete File: "):
            path = line.removeprefix("*** Delete File: ").strip()
            operations.append(PatchOperation("delete", path))
            index += 1
            continue

        if line.startswith("*** Update File: "):
            path = line.removeprefix("*** Update File: ").strip()
            index += 1
            hunks: list[UpdateHunk] = []
            while index < len(lines) and not _is_operation_header(lines[index]):
                if not lines[index].startswith("@@"):
                    raise SafePatchError("update hunks must start with '@@'")
                index += 1
                old_parts: list[str] = []
                new_parts: list[str] = []
                while (
                    index < len(lines)
                    and not lines[index].startswith("@@")
                    and not _is_operation_header(lines[index])
                ):
                    change_line = lines[index]
                    if change_line.startswith(" "):
                        payload = _line_payload(change_line)
                        old_parts.append(payload)
                        new_parts.append(payload)
                    elif change_line.startswith("-"):
                        old_parts.append(_line_payload(change_line))
                    elif change_line.startswith("+"):
                        new_parts.append(_line_payload(change_line))
                    else:
                        raise SafePatchError(f"invalid update line: {change_line!r}")
                    index += 1
                old_text = "".join(old_parts)
                new_text = "".join(new_parts)
                if not old_text:
                    raise SafePatchError(
                        "update hunk must include at least one context or removal line"
                    )
                hunks.append(UpdateHunk(old=old_text, new=new_text))
            if not hunks:
                raise SafePatchError("update operation requires at least one hunk")
            operations.append(PatchOperation("update", path, hunks=tuple(hunks)))
            continue

        raise SafePatchError(f"unknown patch operation: {line!r}")

    return operations


def _resolve_repo_path(repo: Path, patch_path: str) -> Path:
    if not patch_path:
        raise SafePatchError("patch path cannot be empty")
    raw = Path(patch_path)
    if raw.is_absolute():
        raise SafePatchError(f"absolute paths are not allowed: {patch_path}")
    resolved = (repo / raw).resolve()
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise SafePatchError(f"path is outside repository: {patch_path}") from exc
    if resolved == repo:
        raise SafePatchError("patch path must refer to a file")
    return resolved


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def _backup_file(repo: Path, backup_root: Path, path: Path) -> Path | None:
    if not path.exists():
        return None
    relative = path.relative_to(repo)
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return destination


def apply_patch_text(
    repo: Path | str,
    patch_text: str,
    *,
    dry_run: bool = False,
    backup: bool = False,
) -> PatchResult:
    repo_path = Path(repo).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise SafePatchError(f"repository directory does not exist: {repo_path}")

    operations = parse_patch(patch_text)
    staged: dict[Path, str | None] = {}
    changed: list[Path] = []

    def current_content(path: Path) -> str | None:
        if path in staged:
            return staged[path]
        if not path.exists():
            return None
        if not path.is_file():
            raise SafePatchError(f"path is not a file: {path}")
        return _read_text(path)

    for operation in operations:
        path = _resolve_repo_path(repo_path, operation.path)
        before = current_content(path)

        if operation.kind == "add":
            if before is not None:
                raise SafePatchError(f"add target already exists: {operation.path}")
            staged[path] = operation.content

        elif operation.kind == "delete":
            if before is None:
                raise SafePatchError(f"delete target does not exist: {operation.path}")
            staged[path] = None

        elif operation.kind == "update":
            if before is None:
                raise SafePatchError(f"update target does not exist: {operation.path}")
            after = before
            for hunk in operation.hunks:
                count = after.count(hunk.old)
                if count == 0:
                    raise SafePatchError(f"context did not match for {operation.path}")
                if count > 1:
                    raise SafePatchError(f"context is ambiguous for {operation.path}")
                after = after.replace(hunk.old, hunk.new, 1)
            staged[path] = after

        else:  # pragma: no cover - defensive for future operation kinds
            raise SafePatchError(f"unsupported operation kind: {operation.kind}")

        if path not in changed:
            changed.append(path)

    backups: list[Path] = []
    if dry_run:
        return PatchResult(changed_files=changed, dry_run=True, backups=backups)

    backup_root: Path | None = None
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_root = repo_path / ".safe_apply_patch_backup" / timestamp

    for path, content in staged.items():
        if backup_root is not None:
            backup_path = _backup_file(repo_path, backup_root, path)
            if backup_path is not None:
                backups.append(backup_path)
        if content is None:
            path.unlink(missing_ok=True)
        else:
            _write_text(path, content)

    return PatchResult(changed_files=changed, dry_run=False, backups=backups)


def cleanup_if_native_ok(*, script_path: Path, probe_command: list[str], yes: bool) -> bool:
    if not probe_command:
        raise SafePatchError("cleanup requires a native probe command")
    completed = subprocess.run(probe_command, check=False)
    if completed.returncode != 0:
        return False
    if not yes:
        return False
    script_path.unlink(missing_ok=True)
    return True


def _read_patch_argument(patch_arg: str) -> str:
    if patch_arg == "-":
        return sys.stdin.read()
    return Path(patch_arg).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Temporary safe patch applicator for this repository "
            "while Codex apply_patch is unavailable."
        )
    )
    parser.add_argument(
        "--repo", default=".", help="Repository root. Defaults to the current directory."
    )
    parser.add_argument("--patch", default="-", help="Patch file path, or '-' for stdin.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report changes without writing files."
    )
    parser.add_argument(
        "--backup", action="store_true", help="Copy overwritten/deleted files before applying."
    )
    parser.add_argument(
        "--cleanup-if-native-ok",
        action="store_true",
        help=(
            "Run a probe command and delete this script only if "
            "the probe succeeds and --yes is set."
        ),
    )
    parser.add_argument(
        "--yes", action="store_true", help="Confirm cleanup after a successful native probe."
    )
    parser.add_argument(
        "--probe-command",
        nargs=argparse.REMAINDER,
        help="Command used by --cleanup-if-native-ok to prove the native patch path works.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.cleanup_if_native_ok:
            cleaned = cleanup_if_native_ok(
                script_path=Path(__file__).resolve(),
                probe_command=args.probe_command or [],
                yes=args.yes,
            )
            print(
                "cleanup: removed safe_apply_patch.py"
                if cleaned
                else "cleanup: kept safe_apply_patch.py"
            )
            return 0

        patch_text = _read_patch_argument(args.patch)
        result = apply_patch_text(args.repo, patch_text, dry_run=args.dry_run, backup=args.backup)
    except SafePatchError as exc:
        print(f"safe_apply_patch: {exc}", file=sys.stderr)
        return 2

    action = "would change" if result.dry_run else "changed"
    print(f"safe_apply_patch: {action} {len(result.changed_files)} file(s)")
    repo = Path(args.repo).resolve()
    for path in result.changed_files:
        try:
            display = path.relative_to(repo)
        except ValueError:
            display = path
        print(f"- {display}")
    if result.backups:
        print(f"backups: {len(result.backups)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
