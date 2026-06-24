from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from hippocampus_memory.db import Database
from hippocampus_memory.deploy import deploy_codex
from hippocampus_memory.memory_writer import MemoryWriter

Condition = Literal["A_baseline", "B_memory"]
RunMode = Literal["dry_run", "codex"]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

_IGNORE_NAMES = {
    ".git",
    ".hippo",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "hippocampus_memory.egg-info",
    "codex_home",
    "output",
    "tmp",
    "workspace_before_codex",
}
_IGNORE_SETUP_FILES = {".codex_last_message.md", ".hippo.toml"}
_IGNORE_SUFFIXES = {".pyc", ".db", ".sqlite", ".sqlite3"}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    category: str
    prompt: str
    test_command: list[str]
    fixture_strategy: str
    task_dir: Path
    source: dict[str, Any]
    memory_seed_path: Path | None = None

    @property
    def fixture_path(self) -> Path:
        if self.fixture_strategy == "current_repo":
            return _repo_root()
        return self.task_dir / "repo"

    @property
    def memory_seed(self) -> str:
        if self.memory_seed_path is None or not self.memory_seed_path.exists():
            return ""
        return self.memory_seed_path.read_text(encoding="utf-8")

    def memory_seed_for(self, run_id: str) -> str:
        if _is_dynamic_long_memory_task(self):
            return _build_dynamic_long_policy(self.task_id, run_id).memory_seed
        return self.memory_seed


def load_task_specs(tasks_root: Path) -> list[TaskSpec]:
    root = tasks_root.resolve()
    specs: list[TaskSpec] = []
    for task_file in sorted(root.glob("*/task.json")):
        raw = json.loads(task_file.read_text(encoding="utf-8"))
        task_dir = task_file.parent
        memory_seed_path = task_dir / "memory_seed.md"
        specs.append(
            TaskSpec(
                task_id=raw["task_id"],
                title=raw["title"],
                category=raw["category"],
                prompt=raw["prompt"],
                test_command=list(raw["test_command"]),
                fixture_strategy=raw.get("fixture_strategy", "local_fixture"),
                task_dir=task_dir,
                source=raw.get("source", {}),
                memory_seed_path=memory_seed_path if memory_seed_path.exists() else None,
            )
        )
    return specs


@dataclass(frozen=True)
class DynamicLongPolicy:
    memory_seed: str
    hidden_check: str


def _is_dynamic_long_memory_task(spec: TaskSpec) -> bool:
    return spec.source.get("type") == "local-fixture-with-dynamic-long-memory"


def _write_generated_hidden_check(spec: TaskSpec, run_id: str, condition_root: Path) -> Path | None:
    if not _is_dynamic_long_memory_task(spec):
        return None
    policy = _build_dynamic_long_policy(spec.task_id, run_id)
    check_path = condition_root / "generated_hidden_check.py"
    check_path.write_text(policy.hidden_check, encoding="utf-8")
    return check_path


def _build_dynamic_long_policy(task_id: str, run_id: str) -> DynamicLongPolicy:
    digest = hashlib.sha256(f"{task_id}:{run_id}".encode()).hexdigest()
    short = digest[:6]
    schema_label = f"schema_{digest[6:10]}"
    refund_label = f"refund_{digest[10:14]}"
    chargeback_label = f"chargeback_{digest[14:18]}"
    credential_label = f"credential_leak_{digest[18:22]}"
    migration_component = f"migration_{digest[22:26]}"
    mobile_component = f"mobile_{digest[26:30]}"
    crash_term = f"crash_{digest[30:34]}"
    amount_threshold = 800 + (int(digest[34:36], 16) % 5) * 125
    lanes = {
        "containment": f"lane_containment_{short}",
        "architect": f"lane_architect_{digest[36:42]}",
        "revenue": f"lane_revenue_{digest[42:48]}",
        "mobile": f"lane_mobile_{digest[48:54]}",
        "privacy": f"lane_privacy_{digest[54:60]}",
        "standard": "standard",
    }
    memory_seed = _render_dynamic_long_memory(
        run_id=run_id,
        schema_label=schema_label,
        refund_label=refund_label,
        chargeback_label=chargeback_label,
        credential_label=credential_label,
        migration_component=migration_component,
        mobile_component=mobile_component,
        crash_term=crash_term,
        amount_threshold=amount_threshold,
        lanes=lanes,
    )
    hidden_check = _render_dynamic_hidden_check(
        schema_label=schema_label,
        refund_label=refund_label,
        chargeback_label=chargeback_label,
        credential_label=credential_label,
        migration_component=migration_component,
        mobile_component=mobile_component,
        crash_term=crash_term,
        amount_threshold=amount_threshold,
        lanes=lanes,
    )
    return DynamicLongPolicy(memory_seed=memory_seed, hidden_check=hidden_check)


def _render_dynamic_long_memory(
    *,
    run_id: str,
    schema_label: str,
    refund_label: str,
    chargeback_label: str,
    credential_label: str,
    migration_component: str,
    mobile_component: str,
    crash_term: str,
    amount_threshold: int,
    lanes: dict[str, str],
) -> str:
    distractors = [
        "Earlier queue names such as amber, blue, concierge, and west are display-only.",
        "Old mobile notes mentioned emergency_mobile, but that lane is retired.",
        "Legacy billing docs mention fraud_review, which must not be returned.",
        "General privacy wording in free text is not enough to escalate.",
        "Enterprise status only matters for the explicit migration or schema rule.",
        "The policy is deterministic and must not use timestamps, storage, or network calls.",
    ]
    lines = [
        "# Dynamic long-context routing archive",
        f"Run id: {run_id}",
        "This memory is intentionally long and noisy. The current policy table is buried below.",
    ]
    for index in range(1, 25):
        note = distractors[index % len(distractors)]
        lines.append(f"Section {index:02d} - background note. {note}")
    lines.extend(
        [
            "Section 25 - current policy table begins.",
            "Normalize string fields case-insensitively. Treat missing fields as empty "
            "strings, empty lists, false booleans, or zero numeric values.",
            "Rule 1: If incident_code starts with SEC- OR labels contains "
            f"{credential_label}, return {lanes['containment']}.",
            "Rule 2: Else if customer_tier is enterprise AND (component is "
            f"{migration_component} OR labels contains {schema_label}), "
            f"return {lanes['architect']}.",
            "Rule 3: Else if labels contains "
            f"{chargeback_label} OR amount_usd is greater than or equal to "
            f"{amount_threshold} and labels contains {refund_label}, "
            f"return {lanes['revenue']}.",
            "Rule 4: Else if component is "
            f"{mobile_component} AND severity is p0 or p1 AND symptom contains "
            f"{crash_term}, return {lanes['mobile']}.",
            "Rule 5: Else if region is eu AND data_subject_request is true, "
            f"return {lanes['privacy']}.",
            "Rule 6: Else return standard.",
            "Section 26 - current policy table ends.",
            "Precedence matters: use the first matching rule from Section 25.",
        ]
    )
    for index in range(27, 41):
        note = distractors[index % len(distractors)]
        lines.append(f"Section {index:02d} - archived exception. {note}")
    return "\n".join(lines) + "\n"


def _render_dynamic_hidden_check(
    *,
    schema_label: str,
    refund_label: str,
    chargeback_label: str,
    credential_label: str,
    migration_component: str,
    mobile_component: str,
    crash_term: str,
    amount_threshold: int,
    lanes: dict[str, str],
) -> str:
    cases = [
        ({"incident_code": "SEC-1042", "labels": []}, lanes["containment"]),
        ({"incident_code": "GEN-1", "labels": [credential_label]}, lanes["containment"]),
        (
            {"customer_tier": "Enterprise", "component": migration_component, "labels": []},
            lanes["architect"],
        ),
        (
            {"customer_tier": "enterprise", "component": "docs", "labels": [schema_label]},
            lanes["architect"],
        ),
        ({"labels": [chargeback_label], "amount_usd": 12}, lanes["revenue"]),
        ({"labels": [refund_label], "amount_usd": amount_threshold}, lanes["revenue"]),
        (
            {
                "component": mobile_component,
                "severity": "P1",
                "symptom": f"repeat {crash_term} loop",
            },
            lanes["mobile"],
        ),
        ({"region": "EU", "data_subject_request": True}, lanes["privacy"]),
        ({"labels": [refund_label], "amount_usd": amount_threshold - 1}, lanes["standard"]),
    ]
    return (
        "from __future__ import annotations\n\n"
        "import os\n"
        "import sys\n\n"
        "sys.path.insert(0, os.getcwd())\n\n"
        "from routing_policy import route_case\n\n"
        f"CASES = {cases!r}\n\n"
        "for index, (ticket, expected) in enumerate(CASES, 1):\n"
        "    actual = route_case(ticket)\n"
        "    assert actual == expected, f'case {index}: expected {expected!r}, got {actual!r}'\n"
    )


class CodingBenchmarkHarness:
    def __init__(
        self,
        benchmark_root: Path,
        runs_root: Path | None = None,
        command_runner: CommandRunner | None = None,
        codex_timeout_seconds: int = 900,
    ) -> None:
        self.benchmark_root = benchmark_root.resolve()
        self.tasks_root = self.benchmark_root / "tasks"
        self.runs_root = (runs_root or self.benchmark_root / "runs").resolve()
        self.command_runner = command_runner or subprocess.run
        self.codex_timeout_seconds = codex_timeout_seconds

    def run(
        self,
        *,
        task_ids: list[str] | None = None,
        timestamp: str | None = None,
        mode: RunMode = "dry_run",
        append: bool = False,
    ) -> dict[str, Any]:
        selected = self._select_tasks(task_ids)
        run_id = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_root = self.runs_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        task_results = []
        for spec in selected:
            task_results.append(self._run_task(spec, run_root, mode))

        if append:
            task_results = _merge_existing_task_results(run_root, task_results)

        summary: dict[str, Any] = {
            "run_id": run_id,
            "mode": mode,
            "codex_executed": mode == "codex",
            "task_count": len(task_results),
            "tasks": task_results,
        }
        (run_root / "scores.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_root / "benchmark_report.md").write_text(
            _render_report(summary),
            encoding="utf-8",
        )
        return summary

    def _select_tasks(self, task_ids: list[str] | None) -> list[TaskSpec]:
        specs = load_task_specs(self.tasks_root)
        if task_ids is None:
            return specs
        by_id = {spec.task_id: spec for spec in specs}
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"unknown benchmark task id(s): {', '.join(missing)}")
        return [by_id[task_id] for task_id in task_ids]

    def _run_task(self, spec: TaskSpec, run_root: Path, mode: RunMode) -> dict[str, Any]:
        conditions: dict[str, Any] = {}
        for condition in ["A_baseline", "B_memory"]:
            condition_result = self._prepare_and_score_condition(spec, run_root, condition, mode)
            conditions[condition] = condition_result
        return {
            "task_id": spec.task_id,
            "title": spec.title,
            "category": spec.category,
            "source": spec.source,
            "conditions": conditions,
        }

    def _prepare_and_score_condition(
        self,
        spec: TaskSpec,
        run_root: Path,
        condition: Condition,
        mode: RunMode,
    ) -> dict[str, Any]:
        condition_root = run_root / condition / spec.task_id
        workspace = condition_root / "workspace"
        condition_root.mkdir(parents=True, exist_ok=True)
        self._copy_fixture(spec, workspace)

        memory_mcp_called = False
        memory_context = ""
        if condition == "B_memory":
            memory_context, memory_mcp_called = self._prepare_memory_context(
                spec,
                condition_root,
                workspace,
                mode,
                run_root.name,
            )

        prompt = _condition_prompt(spec, condition, memory_context)
        (condition_root / "prompt.md").write_text(prompt, encoding="utf-8")
        snapshot = condition_root / "workspace_before_codex"
        self._copy_workspace_snapshot(workspace, snapshot)

        codex_returncode: int | None = None
        if mode == "codex":
            codex_completed = self._run_codex(prompt, condition_root, workspace)
            codex_returncode = codex_completed.returncode

        generated_check = _write_generated_hidden_check(spec, run_root.name, condition_root)
        completed = self._run_test_command(
            spec.test_command,
            workspace,
            spec.task_dir,
            generated_check=generated_check,
        )
        patch_text = _diff_trees(snapshot, workspace)
        (condition_root / "patch.diff").write_text(patch_text, encoding="utf-8")
        log_text = completed.stdout + completed.stderr
        (condition_root / "pytest.log").write_text(log_text, encoding="utf-8")
        score = _score_result(
            returncode=completed.returncode,
            output=log_text,
            patch_text=patch_text,
            codex_executed=mode == "codex",
            codex_returncode=codex_returncode,
            memory_mcp_called=memory_mcp_called,
        )
        (condition_root / "score.json").write_text(
            json.dumps(score, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return score

    def _copy_fixture(self, spec: TaskSpec, destination: Path) -> None:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(spec.fixture_path, destination, ignore=_ignore_fixture_paths)

    def _copy_workspace_snapshot(self, workspace: Path, snapshot: Path) -> None:
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(workspace, snapshot, ignore=_ignore_fixture_paths)

    def _prepare_memory_context(
        self,
        spec: TaskSpec,
        condition_root: Path,
        workspace: Path,
        mode: RunMode,
        run_id: str,
    ) -> tuple[str, bool]:
        memory_seed = spec.memory_seed_for(run_id)
        if mode == "dry_run":
            memory_context = memory_seed
            (condition_root / "memory_context.md").write_text(memory_context, encoding="utf-8")
            (condition_root / "memory_probe.json").write_text(
                json.dumps(
                    {
                        "mode": "dry_run",
                        "memory_mcp_called": False,
                        "memory_seed_present": bool(memory_seed.strip()),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return memory_context, False

        deploy_codex(workspace, project=spec.task_id, index_project=False)
        if memory_seed.strip():
            db = Database(workspace / ".hippo" / "hippo.db")
            db.initialize()
            MemoryWriter(db).write(
                project=spec.task_id,
                memory_type="constraint",
                content=memory_seed,
                tags=["benchmark", "memory_seed", spec.category],
                importance=1.0,
                confidence=0.95,
            )

        probe = self._run_memory_probe(spec, condition_root, workspace, memory_seed)
        memory_context = probe["memory_context"]
        (condition_root / "memory_context.md").write_text(memory_context, encoding="utf-8")
        return memory_context, bool(probe["memory_mcp_called"])

    def _run_memory_probe(
        self,
        spec: TaskSpec,
        condition_root: Path,
        workspace: Path,
        memory_seed: str,
    ) -> dict[str, Any]:
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "memory_pack",
                    "arguments": {"query": spec.prompt, "project": spec.task_id},
                },
            },
        ]
        completed = _run_command(
            [sys.executable, "-m", "hippocampus_memory", "mcp-project", "--root", str(workspace)],
            cwd=_repo_root(),
            runner=self.command_runner,
            input_text="\n".join(json.dumps(item) for item in requests) + "\n",
            timeout=120,
        )
        memory_context = _extract_memory_context(completed.stdout)
        probe = {
            "mode": "codex",
            "memory_mcp_called": completed.returncode == 0,
            "returncode": completed.returncode,
            "memory_seed_present": bool(memory_seed.strip()),
            "memory_context_nonempty": bool(memory_context.strip()),
            "memory_context_chars": len(memory_context),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        probe["memory_context"] = memory_context
        (condition_root / "memory_probe.json").write_text(
            json.dumps(probe, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return probe

    def _run_codex(
        self,
        prompt: str,
        condition_root: Path,
        workspace: Path,
    ) -> subprocess.CompletedProcess[str]:
        last_message_path = workspace / ".codex_last_message.md"
        model = _benchmark_codex_model()
        command = [
                _codex_executable(),
                "exec",
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "--ignore-user-config",
                "--ignore-rules",
                *( ["--model", model] if model else [] ),
                "--json",
                "--color",
                "never",
                "-C",
                str(workspace),
                "--output-last-message",
                str(last_message_path),
                "-",
            ]
        env = _codex_environment(_prepare_codex_home(condition_root))
        if self.command_runner is subprocess.run:
            return _run_command_to_files(
                command,
                cwd=workspace,
                input_text=prompt,
                timeout=self.codex_timeout_seconds,
                env=env,
                stdout_path=condition_root / "codex.jsonl",
                stderr_path=condition_root / "codex_stderr.log",
            )

        completed = _run_command(
            command,
            cwd=workspace,
            runner=self.command_runner,
            input_text=prompt,
            timeout=self.codex_timeout_seconds,
            env=env,
        )
        (condition_root / "codex.jsonl").write_text(completed.stdout, encoding="utf-8")
        (condition_root / "codex_stderr.log").write_text(completed.stderr, encoding="utf-8")
        return completed


    def _run_test_command(
        self,
        command: list[str],
        cwd: Path,
        task_dir: Path,
        *,
        generated_check: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if generated_check is None and any("{generated_check}" in part for part in command):
            raise ValueError("test command uses {generated_check} but no check was generated")
        replacements = {
            "{task_dir}": str(task_dir),
            "{generated_check}": str(generated_check or ""),
        }
        expanded = []
        for part in command:
            expanded_part = part
            for placeholder, value in replacements.items():
                expanded_part = expanded_part.replace(placeholder, value)
            expanded.append(expanded_part)
        return _run_command(expanded, cwd=cwd, runner=self.command_runner, timeout=120)


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    runner: CommandRunner,
    input_text: str | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    expanded = [sys.executable if part == "{python}" else part for part in command]
    try:
        return runner(
            expanded,
            cwd=cwd,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            expanded,
            124,
            _timeout_stream_to_text(exc.output),
            _timeout_stream_to_text(exc.stderr)
            or f"command timed out after {exc.timeout} seconds",
        )


def _run_command_to_files(
    command: list[str],
    *,
    cwd: Path,
    input_text: str | None,
    timeout: int,
    env: dict[str, str] | None,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    expanded = [sys.executable if part == "{python}" else part for part in command]
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file:
        with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
            process = subprocess.Popen(
                expanded,
                cwd=cwd,
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            try:
                process.communicate(input=input_text, timeout=timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process.pid)
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                stderr_file.write(f"\ncommand timed out after {timeout} seconds\n")
                returncode = 124
    return subprocess.CompletedProcess(
        expanded,
        returncode,
        stdout_path.read_text(encoding="utf-8", errors="replace"),
        stderr_path.read_text(encoding="utf-8", errors="replace"),
    )


def _terminate_process_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return
    try:
        os.kill(pid, 15)
    except OSError:
        return


def _timeout_stream_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _extract_memory_context(stdout: str) -> str:
    context_parts: list[str] = []

    def append_context(text: str) -> None:
        normalized = text.strip()
        if normalized and normalized not in context_parts:
            context_parts.append(normalized)

    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            continue
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            structured_memory_context = _structured_memory_context(structured)
            if structured_memory_context:
                append_context(structured_memory_context)
                continue
            text = structured.get("text")
            if isinstance(text, str):
                append_context(text)
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    append_context(item["text"])
    return "\n".join(context_parts).strip()


def _structured_memory_context(structured: dict[str, Any]) -> str:
    memories = structured.get("selected_memories") or structured.get("retrieved_memories") or []
    if not isinstance(memories, list):
        return ""
    lines = ["Memory Pack:", "Structured retrieved memories:"]
    included = 0
    for memory in memories:
        if not isinstance(memory, dict):
            continue
        content = str(memory.get("content") or "").strip()
        if not content:
            continue
        memory_id = str(memory.get("memory_id") or memory.get("id") or "memory")
        score = memory.get("score")
        score_text = f" score={score:.3f}" if isinstance(score, int | float) else ""
        lines.append(f"- {memory_id}{score_text}: {content}")
        included += 1
    return "\n".join(lines) if included else ""


def _codex_executable() -> str:
    if sys.platform == "win32":
        return shutil.which("codex.cmd") or shutil.which("codex.exe") or "codex.cmd"
    return shutil.which("codex") or "codex"


def _prepare_codex_home(condition_root: Path) -> Path:
    source_home = Path.home() / ".codex"
    codex_home = condition_root / "codex_home"
    if codex_home.exists():
        shutil.rmtree(codex_home)
    codex_home.mkdir(parents=True, exist_ok=True)
    for name in ["auth.json", "config.toml"]:
        source = source_home / name
        if source.exists():
            shutil.copy2(source, codex_home / name)
    return codex_home


def _benchmark_codex_model() -> str | None:
    configured = os.environ.get("HIPPO_BENCHMARK_CODEX_MODEL", "").strip()
    if configured:
        return configured
    config_path = Path.home() / ".codex" / "config.toml"
    if not config_path.exists():
        return None
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    model = config.get("model")
    return model.strip() if isinstance(model, str) and model.strip() else None


def _codex_environment(codex_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    env["UV_CACHE_DIR"] = str(codex_home / "uv-cache")
    env["PIP_CACHE_DIR"] = str(codex_home / "pip-cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _score_result(
    *,
    returncode: int,
    output: str,
    patch_text: str,
    codex_executed: bool,
    codex_returncode: int | None,
    memory_mcp_called: bool,
) -> dict[str, Any]:
    passed, failed = _parse_test_counts(output, returncode)
    total = passed + failed
    pass_rate = passed / total if total else (1.0 if returncode == 0 else 0.0)
    changed_files = _changed_files_from_patch(patch_text)
    diff_lines = len([line for line in patch_text.splitlines() if line.startswith(('+', '-'))])
    run_valid = not codex_executed or codex_returncode == 0
    correctness = 10.0 if returncode == 0 else round(pass_rate * 10, 2)
    scope_score = max(0.0, 10.0 - len(changed_files) * 1.5 - diff_lines * 0.03)
    regression_score = 2.0 if _patch_changes_tests(changed_files) else 10.0
    overall = round(
        correctness * 0.45
        + pass_rate * 10 * 0.25
        + scope_score * 0.15
        + regression_score * 0.15,
        2,
    )
    if not run_valid:
        correctness = 0.0
        scope_score = 0.0
        regression_score = 0.0
        overall = 0.0
    return {
        "codex_executed": codex_executed,
        "codex_returncode": codex_returncode,
        "run_valid": run_valid,
        "invalid_reason": "codex_returncode_nonzero" if not run_valid else None,
        "memory_mcp_called": memory_mcp_called,
        "test_returncode": returncode,
        "tests_passed": passed,
        "tests_failed": failed,
        "test_pass_rate": round(pass_rate, 4),
        "changed_files": changed_files,
        "changed_file_count": len(changed_files),
        "diff_line_count": diff_lines,
        "scores": {
            "correctness": correctness,
            "change_scope": round(scope_score, 2),
            "regression_risk": regression_score,
            "overall": overall,
        },
    }


def _parse_test_counts(output: str, returncode: int) -> tuple[int, int]:
    passed = _sum_pytest_count(output, "passed")
    failed = _sum_pytest_count(output, "failed") + _sum_pytest_count(output, "error")
    if passed == 0 and failed == 0:
        return (1, 0) if returncode == 0 else (0, 1)
    return passed, failed


def _sum_pytest_count(output: str, label: str) -> int:
    return sum(int(match) for match in re.findall(rf"(\d+)\s+{label}", output))


def _changed_files_from_patch(patch_text: str) -> list[str]:
    files = []
    for line in patch_text.splitlines():
        if line.startswith("--- "):
            path = line.removeprefix("--- ").strip()
            if path != "/dev/null":
                files.append(path.removeprefix("a/"))
    return sorted(set(files))


def _patch_changes_tests(changed_files: list[str]) -> bool:
    return any(
        "test" in Path(path).name.casefold() or "tests" in Path(path).parts
        for path in changed_files
    )


def _condition_prompt(spec: TaskSpec, condition: Condition, memory_context: str = "") -> str:
    header = "# A) Baseline" if condition == "A_baseline" else "# B) Memory-enhanced"
    memory_note = (
        "Do not use external memory. Solve from the current workspace only."
        if condition == "A_baseline"
        else "Use the supplied memory context before changing code."
    )
    memory_section = ""
    if condition == "B_memory":
        context_text = memory_context or "(no relevant memory returned)"
        memory_section = f"\n## Memory Context\n\n{context_text}\n"
    return (
        f"{header}\n\n{memory_note}\n\n"
        "Keep changes minimal. Do not edit tests unless the task explicitly asks for it. "
        "Run the task test command before finishing.\n"
        f"{memory_section}\n## Task\n\n{spec.prompt}\n"
    )


def _diff_trees(original: Path, modified: Path) -> str:
    original_files = _collect_files(original)
    modified_files = _collect_files(modified)
    all_relative = sorted(set(original_files) | set(modified_files))
    chunks: list[str] = []
    for relative in all_relative:
        before = original_files.get(relative, [])
        after = modified_files.get(relative, [])
        if before == after:
            continue
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
                lineterm="",
            )
        )
    return "\n".join(chunks) + ("\n" if chunks else "")


def _collect_files(root: Path) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or _is_ignored(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            files[relative] = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
    return files


def _ignore_fixture_paths(_directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        path = Path(name)
        if name in _IGNORE_NAMES or path.suffix in _IGNORE_SUFFIXES:
            ignored.add(name)
        if name in _IGNORE_SETUP_FILES:
            ignored.add(name)
        if name == "runs":
            ignored.add(name)
    return ignored


def _is_ignored(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        any(part in _IGNORE_NAMES for part in relative.parts)
        or path.name in _IGNORE_SETUP_FILES
        or path.suffix in _IGNORE_SUFFIXES
    )


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Coding Task A/B Benchmark Report",
        "",
        f"Run ID: `{summary['run_id']}`",
        f"Mode: `{summary['mode']}`",
        f"Codex executed: `{summary['codex_executed']}`",
        "",
        "| Task | Category | A overall | B overall | A pass | B pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in summary["tasks"]:
        a = task["conditions"]["A_baseline"]
        b = task["conditions"]["B_memory"]
        lines.append(
            "| {task_id} | {category} | {a_overall} | {b_overall} | {a_pass} | {b_pass} |".format(
                task_id=task["task_id"],
                category=task["category"],
                a_overall=a["scores"]["overall"],
                b_overall=b["scores"]["overall"],
                a_pass=a["test_pass_rate"],
                b_pass=b["test_pass_rate"],
            )
        )
    lines.extend(
        [
            "",
            "Dry-run mode validates artifact generation only. Codex mode records "
            "Codex logs and B-condition memory probes.",
        ]
    )
    return "\n".join(lines) + "\n"


def _merge_existing_task_results(
    run_root: Path,
    new_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scores_path = run_root / "scores.json"
    if not scores_path.exists():
        return new_results
    existing = json.loads(scores_path.read_text(encoding="utf-8"))
    merged_by_id = {
        task["task_id"]: task
        for task in existing.get("tasks", [])
        if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    }
    for task in new_results:
        merged_by_id[task["task_id"]] = task
    return sorted(merged_by_id.values(), key=lambda task: task["task_id"])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the coding-task A/B benchmark harness.")
    parser.add_argument("--benchmark-root", default="benchmarks/coding_tasks")
    parser.add_argument("--runs-root")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--timestamp")
    parser.add_argument("--mode", choices=["dry_run", "codex"], default="dry_run")
    parser.add_argument("--codex-timeout-seconds", type=int, default=900)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args(argv)

    harness = CodingBenchmarkHarness(
        benchmark_root=Path(args.benchmark_root),
        runs_root=Path(args.runs_root) if args.runs_root else None,
        codex_timeout_seconds=args.codex_timeout_seconds,
    )
    summary = harness.run(
        task_ids=args.tasks,
        timestamp=args.timestamp,
        mode=args.mode,
        append=args.append,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
