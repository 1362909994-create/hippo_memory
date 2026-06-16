from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class InjectMode(StrEnum):
    PRINT = "print"
    FILE = "file"
    ENV = "env"
    STDIN = "stdin"
    ARG = "arg"


@dataclass(slots=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    context_file: Path | None


def run_with_context(
    command: list[str],
    context: str,
    project: str,
    intent: str,
    inject: str,
    cwd: str | Path | None = None,
    context_file: str | Path | None = None,
) -> RunResult:
    mode = InjectMode(inject)
    if mode == InjectMode.PRINT:
        return RunResult(returncode=0, stdout=context + "\n", stderr="", context_file=None)
    if not command:
        raise ValueError("a command is required unless --inject print is used")

    context_path = write_context_file(context, context_file)
    env = os.environ.copy()
    env["HIPPO_PROJECT"] = project
    env["HIPPO_INTENT"] = intent
    env["HIPPO_CONTEXT_FILE"] = str(context_path)
    if len(context) < 24_000:
        env["HIPPO_CONTEXT"] = context
    else:
        env["HIPPO_CONTEXT"] = f"Context is too large for env; read {context_path}"

    final_command = list(command)
    stdin_data = None
    if mode == InjectMode.STDIN:
        stdin_data = context + "\n"
    elif mode == InjectMode.ARG:
        final_command.append(context)
    elif mode not in {InjectMode.FILE, InjectMode.ENV}:
        raise ValueError(f"unsupported inject mode: {inject}")

    completed = subprocess.run(
        final_command,
        input=stdin_data,
        text=True,
        capture_output=True,
        env=env,
        cwd=str(cwd) if cwd else None,
        check=False,
    )
    return RunResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        context_file=context_path,
    )


def write_context_file(context: str, context_file: str | Path | None = None) -> Path:
    if context_file:
        path = Path(context_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(context, encoding="utf-8")
        return path

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="hippo-context-",
        suffix=".md",
        delete=False,
    )
    try:
        handle.write(context)
        return Path(handle.name)
    finally:
        handle.close()
