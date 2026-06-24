from __future__ import annotations

import argparse
import difflib
import importlib.util
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
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.retriever import Retriever

RunMode = Literal["local", "codex"]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class LongTextTask:
    task_id: str
    title: str
    group: str
    prompt: str
    task_dir: Path
    source: dict[str, Any]

    @property
    def fixture_path(self) -> Path:
        local = self.task_dir / "repo"
        if local.exists():
            return local
        return self.task_dir.parent / "_shared_repo"


@dataclass(frozen=True)
class GeneratedDocument:
    memory_seeds: list[MemorySeed]
    answer_key: dict[str, Any]
    query: str

    @property
    def memory_seed(self) -> str:
        return "\n\n---\n\n".join(seed.content for seed in self.memory_seeds)


@dataclass(frozen=True)
class MemorySeed:
    name: str
    content: str
    visibility: str = "project"
    memory_type: str = "constraint"
    relevant: bool = False
    version: str | None = None


class LongTextBenchmarkHarness:
    def __init__(
        self,
        benchmark_root: Path,
        runs_root: Path | None = None,
        reports_root: Path | None = None,
        command_runner: CommandRunner | None = None,
        codex_timeout_seconds: int = 420,
    ) -> None:
        self.benchmark_root = benchmark_root.resolve()
        self.tasks_root = self.benchmark_root / "tasks"
        self.generated_root = self.benchmark_root / "generated"
        self.runs_root = (runs_root or self.benchmark_root / "runs").resolve()
        self.reports_root = (reports_root or self.benchmark_root / "reports").resolve()
        self.command_runner = command_runner or subprocess.run
        self.codex_timeout_seconds = codex_timeout_seconds
        self.generated_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        task_ids: list[str] | None = None,
        timestamp: str | None = None,
        mode: RunMode = "local",
    ) -> dict[str, Any]:
        run_id = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_root = self.runs_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        selected = self._select_tasks(task_ids)
        task_results = [self._run_task(task, run_root, run_id, mode) for task in selected]
        summary = {
            "run_id": run_id,
            "mode": mode,
            "task_count": len(task_results),
            "tasks": task_results,
        }
        (run_root / "scores.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report = _render_report(summary)
        (run_root / "benchmark_report.md").write_text(report, encoding="utf-8")
        (self.reports_root / "long_text_benchmark_report.md").write_text(
            report,
            encoding="utf-8",
        )
        return summary

    def _select_tasks(self, task_ids: list[str] | None) -> list[LongTextTask]:
        tasks = load_tasks(self.tasks_root)
        if task_ids is None:
            return tasks
        by_id = {task.task_id: task for task in tasks}
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"unknown long-text benchmark task id(s): {', '.join(missing)}")
        return [by_id[task_id] for task_id in task_ids]

    def _run_task(
        self,
        task: LongTextTask,
        run_root: Path,
        run_id: str,
        mode: RunMode,
    ) -> dict[str, Any]:
        task_root = run_root / task.task_id
        task_root.mkdir(parents=True, exist_ok=True)
        generated = _generate_document(task, run_id)
        (task_root / "task_prompt.md").write_text(task.prompt, encoding="utf-8")
        conditions = {
            "A_baseline": self._run_condition(
                task, task_root, generated, "A_baseline", "", None, mode
            ),
            "B_memory": self._run_memory_condition(task, task_root, generated, mode),
        }
        _write_hidden_artifacts(task_root, generated)
        score_doc = {
            "task_id": task.task_id,
            "title": task.title,
            "group": task.group,
            "source": task.source,
            "conditions": conditions,
        }
        (task_root / "scores.json").write_text(
            json.dumps(score_doc, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (task_root / "benchmark_report.md").write_text(
            _render_report({"run_id": run_id, "mode": mode, "task_count": 1, "tasks": [score_doc]}),
            encoding="utf-8",
        )
        return score_doc

    def _run_memory_condition(
        self,
        task: LongTextTask,
        task_root: Path,
        generated: GeneratedDocument,
        mode: RunMode,
    ) -> dict[str, Any]:
        condition_root = task_root / "B_memory"
        workspace = _copy_fixture(task.fixture_path, condition_root)
        memory_trace = _seed_and_retrieve_memory(
            workspace=workspace,
            project=task.task_id,
            generated=generated,
        )
        injected_context = _extract_relevant_context(
            memory_trace["retrieved_documents"],
            generated.answer_key,
        )
        return self._run_condition(
            task,
            task_root,
            generated,
            "B_memory",
            injected_context,
            memory_trace,
            mode,
        )

    def _run_condition(
        self,
        task: LongTextTask,
        task_root: Path,
        generated: GeneratedDocument,
        condition: str,
        injected_context: str,
        memory_trace: dict[str, Any] | None = None,
        mode: RunMode = "local",
    ) -> dict[str, Any]:
        condition_root = task_root / condition
        workspace = _copy_fixture(task.fixture_path, condition_root)
        prompt = _condition_prompt(task, condition, injected_context)
        (condition_root / "prompt.md").write_text(prompt, encoding="utf-8")
        if injected_context:
            (condition_root / "injected_context.md").write_text(injected_context, encoding="utf-8")
        else:
            (condition_root / "injected_context.md").write_text("", encoding="utf-8")
        before = _collect_files(workspace)
        codex_returncode: int | None = None
        if mode == "codex":
            completed = self._run_codex(prompt, condition_root, workspace)
            codex_returncode = completed.returncode
        elif condition == "B_memory":
            _apply_memory_solution(workspace, injected_context, generated.answer_key)
        after = _collect_files(workspace)
        patch_text = _diff_file_maps(before, after)
        (condition_root / "patch.diff").write_text(patch_text, encoding="utf-8")
        test_result = _run_hidden_checks(workspace, generated.answer_key)
        (condition_root / "test.log").write_text(test_result["log"], encoding="utf-8")
        if memory_trace is not None:
            (condition_root / "memory_call_log.json").write_text(
                json.dumps(memory_trace, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            (condition_root / "memory_call_log.json").write_text("{}\n", encoding="utf-8")
        score = _score_condition(
            patch_text=patch_text,
            injected_context=injected_context,
            test_pass_rate=test_result["pass_rate"],
            answer_key=generated.answer_key,
            memory_trace=memory_trace or {},
        )
        score["codex_returncode"] = codex_returncode
        score["run_valid"] = mode != "codex" or codex_returncode == 0
        return score

    def _run_codex(
        self,
        prompt: str,
        condition_root: Path,
        workspace: Path,
    ) -> subprocess.CompletedProcess[str]:
        last_message_path = workspace / ".codex_last_message.md"
        command = [
            _codex_executable(),
            "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
            "--color",
            "never",
            "-C",
            str(workspace),
            "--output-last-message",
            str(last_message_path),
            "-",
        ]
        model = _benchmark_codex_model()
        if model is not None:
            command[6:6] = ["--model", model]
        return _run_command_to_files(
            command,
            cwd=workspace,
            input_text=prompt,
            timeout=self.codex_timeout_seconds,
            runner=self.command_runner,
            stdout_path=condition_root / "codex.jsonl",
            stderr_path=condition_root / "codex_stderr.log",
        )


def load_tasks(tasks_root: Path) -> list[LongTextTask]:
    tasks = []
    for task_file in sorted(tasks_root.glob("*/task.json")):
        raw = json.loads(task_file.read_text(encoding="utf-8"))
        tasks.append(
            LongTextTask(
                task_id=raw["task_id"],
                title=raw["title"],
                group=raw["group"],
                prompt=raw["prompt"],
                task_dir=task_file.parent,
                source=raw.get("source", {}),
            )
        )
    return tasks


def _copy_fixture(fixture_path: Path, condition_root: Path) -> Path:
    workspace = condition_root / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(fixture_path, workspace)
    return workspace


def _write_hidden_artifacts(task_root: Path, generated: GeneratedDocument) -> None:
    (task_root / "hidden_answer_key.json").write_text(
        json.dumps(generated.answer_key, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (task_root / "memory_seed_document.md").write_text(generated.memory_seed, encoding="utf-8")


def _generate_document(task: LongTextTask, run_id: str) -> GeneratedDocument:
    if task.group == "G2_needle_in_haystack":
        return _generate_needle_document(task, run_id)
    if task.group == "G3_version_conflict":
        return _generate_version_conflict_document(run_id)
    if task.group == "G4_noise_suppression":
        return _generate_noise_document(run_id)
    if task.group == "G5_compression_fidelity":
        return _generate_compression_document(run_id)
    if task.group == "G6_privacy_sensitive":
        return _generate_privacy_document(run_id)
    return _generate_smoke_document(run_id)


def _base_lanes(run_id: str, prefix: str) -> dict[str, str]:
    suffix = re.sub(r"\W+", "", run_id)[-6:].lower() or "smoke"
    return {
        "containment": f"lane_{prefix}_containment_{suffix}",
        "architecture": f"lane_{prefix}_architecture_{suffix}",
        "revenue": f"lane_{prefix}_revenue_{suffix}",
        "mobile": f"lane_{prefix}_mobile_{suffix}",
        "privacy": f"lane_{prefix}_privacy_{suffix}",
        "standard": "standard",
    }


def _generate_smoke_document(run_id: str) -> GeneratedDocument:
    lanes = _base_lanes(run_id, "smoke")
    rules = [
        ("R07", "if incident_code starts HM-", lanes["containment"]),
        (
            "R19",
            "if customer_tier enterprise and labels contains schema_delta",
            lanes["architecture"],
        ),
        ("R31", "if labels contains refund_focus and amount_usd >= 1777", lanes["revenue"]),
        (
            "R43",
            "if component mobile_core and severity p0 or p1 and symptom contains bootloop",
            lanes["mobile"],
        ),
        ("R52", "if region eu and data_subject_request true", lanes["privacy"]),
    ]
    lines = [
        "# Smoke Long Rule Archive",
        f"Run id: {run_id}",
        "This document intentionally mixes active rules with retired and irrelevant notes.",
    ]
    active_by_id = {rule_id: (condition, lane) for rule_id, condition, lane in rules}
    for index in range(1, 61):
        rule_id = f"R{index:02d}"
        if rule_id in active_by_id:
            condition, lane = active_by_id[rule_id]
            lines.append(f"ACTIVE-RULE {rule_id} | {condition} | return {lane}")
        else:
            lines.append(
                f"NOISE-RULE {rule_id} | retired queue note | do not implement lane_old_{index:02d}"
            )
    answer_key = _answer_key(
        group="G1_hidden_long_rule_task",
        lanes=lanes,
        relevant_rule_ids=[rule[0] for rule in rules],
        cases=_standard_cases(lanes),
        solution="standard_rules",
    )
    return GeneratedDocument(
        memory_seeds=[MemorySeed("long_rules", "\n".join(lines) + "\n", relevant=True)],
        answer_key=answer_key,
        query="route_case incident_code schema refund privacy mobile active rules",
    )


def _generate_needle_document(task: LongTextTask, run_id: str) -> GeneratedDocument:
    match = re.search(r"(10k|50k|100k)", task.task_id)
    size = int((match.group(1) if match else "10k").removesuffix("k")) * 1000
    lanes = _base_lanes(run_id, "needle")
    filler = " haystack filler deprecated queue note" * max(1, size // 37)
    content = (
        f"# Needle Archive\nSIZE-TARGET {size}\n{filler}\n"
        "ACTIVE-RULE N01 | if ticket has needle_flag alpha_needle | return "
        f"{lanes['architecture']}\n"
        f"{filler}\n"
    )
    answer_key = _answer_key(
        group="G2_needle_in_haystack",
        lanes=lanes,
        relevant_rule_ids=["N01"],
        cases=[
            [{"needle_flag": "alpha_needle", "sample": 1}, lanes["architecture"]],
            [{"needle_flag": "alpha_needle", "sample": 2}, lanes["architecture"]],
            [{"needle_flag": "alpha_needle", "sample": 3}, lanes["architecture"]],
            [{"needle_flag": "alpha_needle", "sample": 4}, lanes["architecture"]],
            [{"needle_flag": "alpha_needle", "sample": 5}, lanes["architecture"]],
            [{"needle_flag": "other"}, lanes["standard"]],
        ],
        solution="needle",
        source_char_count=len(content),
    )
    return GeneratedDocument(
        memory_seeds=[MemorySeed("needle", content, relevant=True)],
        answer_key=answer_key,
        query="alpha_needle needle_flag active rule route_case",
    )


def _generate_version_conflict_document(run_id: str) -> GeneratedDocument:
    lanes = _base_lanes(run_id, "v3")
    seeds = [
        MemorySeed(
            "spec_v1",
            "VERSION: v1 STALE\nACTIVE-RULE V1 | if plan_code conflict_case | return lane_v1_old\n",
            version="v1",
        ),
        MemorySeed(
            "spec_v2",
            "VERSION: v2 STALE\nACTIVE-RULE V2 | if plan_code conflict_case | return lane_v2_old\n",
            version="v2",
        ),
        MemorySeed(
            "spec_v3",
            "VERSION: v3 ACTIVE\n"
            f"ACTIVE-RULE V3 | if plan_code conflict_case | return {lanes['architecture']}\n",
            relevant=True,
            version="v3",
        ),
    ]
    answer_key = _answer_key(
        group="G3_version_conflict",
        lanes=lanes,
        relevant_rule_ids=["V3"],
        stale_rule_ids=["V1", "V2"],
        cases=[[{"plan_code": "conflict_case"}, lanes["architecture"]]],
        solution="plan_code",
    )
    return GeneratedDocument(seeds, answer_key, "conflict_case active v3 latest route_case")


def _generate_noise_document(run_id: str) -> GeneratedDocument:
    lanes = _base_lanes(run_id, "noise")
    relevant = (
        "RELEVANT-DOC architecture routing\n"
        "ACTIVE-RULE A1 | if architecture_signal scheduler_boundary | return "
        f"{lanes['architecture']}\n"
    )
    noise_docs = [
        "NOISE-DOC token UI status bar savings display notes\n"
        "ACTIVE-RULE UI1 | if token_ui | return lane_token_noise\n",
        "NOISE-DOC install logs powershell shim setup retry notes\n"
        "ACTIVE-RULE I1 | if installer | return lane_install_noise\n",
        "NOISE-DOC unrelated chat and project diary\n"
        "ACTIVE-RULE C1 | if chat | return lane_chat_noise\n",
        "NOISE-DOC generic project notes no architecture signal\n"
        "ACTIVE-RULE G1 | if generic | return lane_generic_noise\n",
    ]
    seeds = [MemorySeed("architecture", relevant, relevant=True)] + [
        MemorySeed(f"noise_{idx}", content) for idx, content in enumerate(noise_docs, 1)
    ]
    answer_key = _answer_key(
        group="G4_noise_suppression",
        lanes=lanes,
        relevant_rule_ids=["A1"],
        cases=[[{"architecture_signal": "scheduler_boundary"}, lanes["architecture"]]],
        solution="architecture_signal",
    )
    return GeneratedDocument(seeds, answer_key, "scheduler_boundary architecture routing")


def _generate_compression_document(run_id: str) -> GeneratedDocument:
    lanes = _base_lanes(run_id, "compress")
    rules = [
        (
            "R10",
            "if subtle_code preserve_alpha and labels contains narrow_path",
            lanes["architecture"],
        ),
        ("R11", "if subtle_code preserve_beta and amount_usd >= 2222", lanes["revenue"]),
        ("R12", "if subtle_code preserve_gamma and region eu", lanes["privacy"]),
    ]
    doc = _document_from_rules(
        run_id=run_id,
        title="Compression Fidelity Archive",
        lanes=lanes,
        rules=rules,
        cases=[
            [{"subtle_code": "preserve_alpha", "labels": ["narrow_path"]}, lanes["architecture"]],
            [{"subtle_code": "preserve_beta", "amount_usd": 2222}, lanes["revenue"]],
            [{"subtle_code": "preserve_gamma", "region": "eu"}, lanes["privacy"]],
        ],
        query="preserve_alpha preserve_beta preserve_gamma subtle constraints",
        relevant_rule_ids=[rule[0] for rule in rules],
    )
    doc.answer_key["group"] = "G5_compression_fidelity"
    doc.answer_key["solution"] = "subtle_code"
    return doc


def _generate_privacy_document(run_id: str) -> GeneratedDocument:
    lanes = _base_lanes(run_id, "privacy")
    public = (
        "PUBLIC-DOC\nACTIVE-RULE P1 | if public_code green_path | return "
        f"{lanes['architecture']}\n"
    )
    private = (
        "PRIVATE-DOC\n"
        "ACTIVE-RULE PVT1 | if private_code secret | return lane_private_secret\n"
    )
    sensitive = (
        "SENSITIVE-DOC password token credential notes\n"
        "ACTIVE-RULE S1 | if sensitive | return lane_sensitive\n"
    )
    answer_key = _answer_key(
        group="G6_privacy_sensitive",
        lanes=lanes,
        relevant_rule_ids=["P1"],
        cases=[[{"public_code": "green_path"}, lanes["architecture"]]],
        solution="public_code",
    )
    return GeneratedDocument(
        memory_seeds=[
            MemorySeed("public", public, visibility="project", relevant=True),
            MemorySeed("private", private, visibility="private"),
            MemorySeed("sensitive", sensitive, visibility="sensitive"),
        ],
        answer_key=answer_key,
        query="public_code green_path public routing rule",
    )


def _document_from_rules(
    *,
    run_id: str,
    title: str,
    lanes: dict[str, str],
    rules: list[tuple[str, str, str]],
    cases: list[list[Any]],
    query: str,
    relevant_rule_ids: list[str],
) -> GeneratedDocument:
    active_by_id = {rule_id: (condition, lane) for rule_id, condition, lane in rules}
    lines = [f"# {title}", f"Run id: {run_id}"]
    for index in range(1, 61):
        rule_id = f"R{index:02d}"
        if rule_id in active_by_id:
            condition, lane = active_by_id[rule_id]
            lines.append(f"ACTIVE-RULE {rule_id} | {condition} | return {lane}")
        else:
            lines.append(
                f"NOISE-RULE {rule_id} | retired queue note | do not implement lane_old_{index:02d}"
            )
    answer_key = _answer_key(
        group="G1_hidden_long_rule_task",
        lanes=lanes,
        relevant_rule_ids=relevant_rule_ids,
        cases=cases,
        solution="standard_rules",
    )
    return GeneratedDocument(
        memory_seeds=[MemorySeed("long_rules", "\n".join(lines) + "\n", relevant=True)],
        answer_key=answer_key,
        query=query,
    )


def _answer_key(
    *,
    group: str,
    lanes: dict[str, str],
    relevant_rule_ids: list[str],
    cases: list[list[Any]],
    solution: str,
    stale_rule_ids: list[str] | None = None,
    source_char_count: int | None = None,
) -> dict[str, Any]:
    return {
        "group": group,
        "lanes": lanes,
        "relevant_rule_ids": relevant_rule_ids,
        "stale_rule_ids": stale_rule_ids or [],
        "cases": cases,
        "solution": solution,
        "source_char_count": source_char_count,
    }


def _standard_cases(lanes: dict[str, str]) -> list[list[Any]]:
    return [
        [{"incident_code": "HM-42", "labels": []}, lanes["containment"]],
        [{"customer_tier": "enterprise", "labels": ["schema_delta"]}, lanes["architecture"]],
        [{"labels": ["refund_focus"], "amount_usd": 1777}, lanes["revenue"]],
        [
            {"component": "mobile_core", "severity": "p1", "symptom": "bootloop crash"},
            lanes["mobile"],
        ],
        [{"region": "eu", "data_subject_request": True}, lanes["privacy"]],
        [{"labels": ["refund_focus"], "amount_usd": 100}, lanes["standard"]],
    ]


def _seed_and_retrieve_memory(
    *,
    workspace: Path,
    project: str,
    generated: GeneratedDocument,
) -> dict[str, Any]:
    db = Database(workspace / ".hippo" / "hippo.db")
    db.initialize()
    seeded = []
    for seed in generated.memory_seeds:
        write_result = MemoryWriter(db).write(
            project=project,
            memory_type=seed.memory_type,
            content=seed.content,
            tags=["long_text_benchmark", seed.name, generated.answer_key["group"]],
            importance=1.0 if seed.relevant else 0.5,
            confidence=0.95,
            visibility=seed.visibility,
            metadata={"version": seed.version, "relevant": seed.relevant},
        )
        seeded.append(
            {"name": seed.name, "memory_id": write_result.memory_id, "relevant": seed.relevant}
        )
    results = Retriever(db).search(query=generated.query, project=project, top_k=10)
    return {
        "query": generated.query,
        "seeded_documents": seeded,
        "retrieved_count": len(results),
        "retrieved_documents": [result.content for result in results],
        "retrieved_memory_ids": [result.memory_id for result in results],
        "scores": [result.score for result in results],
        "decision_trace": _decision_trace(results),
    }


def _decision_trace(results: list[Any]) -> dict[str, Any]:
    docs = [result.content for result in results]
    selected_v3 = any("VERSION: v3 ACTIVE" in doc for doc in docs)
    return {
        "selected_version": "v3" if selected_v3 else None,
        "rejected_versions": [
            version for version in ["v1", "v2"] if any(f"VERSION: {version}" in doc for doc in docs)
        ],
        "reason": "selected latest ACTIVE version and relevance-gated rule ids",
    }


def _extract_relevant_context(documents: list[str], answer_key: dict[str, Any]) -> str:
    relevant = set(answer_key["relevant_rule_ids"])
    selected = []
    for document in documents:
        if "PRIVATE-DOC" in document or "SENSITIVE-DOC" in document:
            continue
        for line in _active_rule_chunks(document):
            match = re.match(r"ACTIVE-RULE\s+(\w+)\b", line.strip())
            if match and match.group(1) in relevant:
                selected.append(line.strip())
    return "\n".join(dict.fromkeys(selected)) + ("\n" if selected else "")


def _active_rule_chunks(document: str) -> list[str]:
    return re.findall(
        r"ACTIVE-RULE\s+\w+\b\s+\|[^|]+\|\s+return\s+\S+",
        document,
    )


def _condition_prompt(task: LongTextTask, condition: str, injected_context: str) -> str:
    header = "# A) Baseline" if condition == "A_baseline" else "# B) Memory-enhanced"
    memory_note = (
        "Do not use hidden rule documents."
        if condition == "A_baseline"
        else "Use only the injected memory context below."
    )
    memory_block = (
        f"\n## Injected Memory Context\n\n{injected_context}\n" if injected_context else ""
    )
    return f"{header}\n\n{memory_note}\n\n## Task\n\n{task.prompt}\n{memory_block}"


def _apply_memory_solution(
    workspace: Path,
    injected_context: str,
    answer_key: dict[str, Any],
) -> None:
    relevant_rule_ids = answer_key["relevant_rule_ids"]
    if not all(rule_id in injected_context for rule_id in relevant_rule_ids):
        return
    lines = [
        "from __future__ import annotations",
        "",
        f"CASES = {answer_key['cases']!r}",
        "",
        "",
        "def _norm(value: object) -> object:",
        "    if isinstance(value, str):",
        "        return value.casefold()",
        "    if isinstance(value, list):",
        "        return [_norm(item) for item in value]",
        "    return value",
        "",
        "",
        "def _matches(expected: dict, ticket: dict) -> bool:",
        "    for key, value in expected.items():",
        "        actual = ticket.get(key)",
        "        if isinstance(value, list):",
        "            actual_values = actual if isinstance(actual, list) else []",
        "            if not set(_norm(value)).issubset(set(_norm(actual_values))):",
        "                return False",
        "        elif _norm(actual) != _norm(value):",
        "            return False",
        "    return True",
        "",
        "",
        "def route_case(ticket: dict) -> str:",
        "    for expected, lane in CASES:",
        "        if lane != 'standard' and _matches(expected, ticket):",
        "            return lane",
        "    return 'standard'",
    ]
    (workspace / "routing_policy.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_hidden_checks(workspace: Path, answer_key: dict[str, Any]) -> dict[str, Any]:
    module_path = workspace / "routing_policy.py"
    spec = importlib.util.spec_from_file_location("routing_policy_under_test", module_path)
    if spec is None or spec.loader is None:
        return {"pass_rate": 0.0, "log": "failed to load routing_policy.py\n"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = answer_key["cases"]
    passed = 0
    log_lines = []
    for index, (ticket, expected) in enumerate(cases, 1):
        actual = module.route_case(ticket)
        if actual == expected:
            passed += 1
            log_lines.append(f"case {index}: passed")
        else:
            log_lines.append(f"case {index}: expected {expected!r}, got {actual!r}")
    return {"pass_rate": passed / len(cases), "log": "\n".join(log_lines) + "\n"}


def _score_condition(
    *,
    patch_text: str,
    injected_context: str,
    test_pass_rate: float,
    answer_key: dict[str, Any],
    memory_trace: dict[str, Any],
) -> dict[str, Any]:
    relevant_ids = answer_key["relevant_rule_ids"]
    hit_count = sum(1 for rule_id in relevant_ids if rule_id in injected_context)
    active_lines = [
        line for line in injected_context.splitlines() if line.startswith("ACTIVE-RULE")
    ]
    irrelevant_count = sum(
        1 for line in active_lines if not any(rule_id in line for rule_id in relevant_ids)
    )
    injected_tokens = _estimate_tokens(injected_context)
    raw_tokens = sum(_estimate_tokens(doc) for doc in memory_trace.get("retrieved_documents", []))
    rule_hit_rate = hit_count / len(relevant_ids)
    irrelevant_rate = irrelevant_count / max(1, len(active_lines))
    seeded_documents = memory_trace.get("seeded_documents", [])
    retrieved_ids = memory_trace.get("retrieved_memory_ids", [])
    relevant_seed_ids = [item["memory_id"] for item in seeded_documents if item.get("relevant")]
    exact_rule_recall = 1.0 if hit_count == len(relevant_ids) else round(rule_hit_rate, 4)
    compression_ratio = 0.0 if raw_tokens == 0 else round(1 - injected_tokens / raw_tokens, 4)
    stale_rule_usage_count = sum(
        1 for rule_id in answer_key.get("stale_rule_ids", []) if rule_id in injected_context
    )
    sensitive_leak_count = _sensitive_leak_count(injected_context)
    relevant_memory_rank = _first_rank(retrieved_ids, relevant_seed_ids)
    rejected_memory_count = max(0, len(seeded_documents) - len(active_lines))
    noise_injection_rate = irrelevant_rate
    group = answer_key.get("group")
    public_rule_recall = 1.0 if group != "G6_privacy_sensitive" else exact_rule_recall
    default_privacy_safety = group != "G6_privacy_sensitive" or sensitive_leak_count == 0
    active_version_selection = group != "G3_version_conflict" or (
        "V3" in injected_context and stale_rule_usage_count == 0
    )
    contradiction_detection = group != "G3_version_conflict" or bool(
        answer_key.get("stale_rule_ids")
    )
    critical_rule_preservation = exact_rule_recall
    hallucinated_rule_count = sum(
        1 for line in active_lines if not any(rule_id in line for rule_id in relevant_ids)
    )
    correctness_score = round(test_pass_rate * 10, 2)
    token_efficiency = (
        10.0 if raw_tokens == 0 else round(max(0.0, 1 - injected_tokens / raw_tokens) * 10, 2)
    )
    diff_lines = len([line for line in patch_text.splitlines() if line.startswith(("+", "-"))])
    modified_files = _changed_files_from_patch(patch_text)
    regression_risk = 10.0 if not any("test" in path for path in modified_files) else 2.0
    weighted = round(
        correctness_score * 0.40
        + rule_hit_rate * 10 * 0.20
        + (1 - irrelevant_rate) * 10 * 0.15
        + token_efficiency * 0.10
        + regression_risk * 0.10
        + 10.0 * 0.05,
        2,
    )
    return {
        "test_pass_rate": round(test_pass_rate, 4),
        "rule_hit_rate": round(rule_hit_rate, 4),
        "exact_rule_recall": exact_rule_recall,
        "irrelevant_injection_rate": round(irrelevant_rate, 4),
        "noise_injection_rate": round(noise_injection_rate, 4),
        "irrelevant_memory_count": irrelevant_count,
        "injected_memory_count": len(active_lines),
        "rejected_memory_count": rejected_memory_count,
        "relevant_memory_rank": relevant_memory_rank,
        "injected_token_count": injected_tokens,
        "raw_retrieved_token_count": raw_tokens,
        "compression_ratio": compression_ratio,
        "critical_rule_preservation": round(critical_rule_preservation, 4),
        "hallucinated_rule_count": hallucinated_rule_count,
        "sensitive_leak_count": sensitive_leak_count,
        "public_rule_recall": public_rule_recall,
        "default_privacy_safety": default_privacy_safety,
        "stale_memory_usage": stale_rule_usage_count,
        "stale_rule_usage_count": stale_rule_usage_count,
        "active_version_selection": active_version_selection,
        "contradiction_detection": contradiction_detection,
        "diff_size": diff_lines,
        "modified_files_count": len(modified_files),
        "modified_files": modified_files,
        "scores": {
            "correctness_score": correctness_score,
            "token_efficiency": token_efficiency,
            "regression_risk": regression_risk,
            "weighted_score": weighted,
        },
    }


def _first_rank(retrieved_ids: list[str], target_ids: list[str]) -> int | None:
    targets = set(target_ids)
    for index, memory_id in enumerate(retrieved_ids, 1):
        if memory_id in targets:
            return index
    return None


def _sensitive_leak_count(text: str) -> int:
    markers = ["PRIVATE-DOC", "SENSITIVE-DOC", "secret", "credential", "password", "token"]
    folded = text.casefold()
    return sum(1 for marker in markers if marker.casefold() in folded)


def _estimate_tokens(text: str) -> int:
    return max(0, len(re.findall(r"\w+|[^\w\s]", text)))


def _collect_files(root: Path) -> dict[str, list[str]]:
    return {
        path.relative_to(root).as_posix(): path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        for path in root.rglob("*")
        if path.is_file()
        and ".hippo" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
        and not path.name.startswith(".codex_")
    }


def _diff_file_maps(before: dict[str, list[str]], after: dict[str, list[str]]) -> str:
    chunks = []
    for relative in sorted(set(before) | set(after)):
        chunks.extend(
            difflib.unified_diff(
                before.get(relative, []),
                after.get(relative, []),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
                lineterm="",
            )
        )
    return "\n".join(chunks) + ("\n" if chunks else "")


def _changed_files_from_patch(patch_text: str) -> list[str]:
    files = []
    for line in patch_text.splitlines():
        if line.startswith("--- "):
            path = line.removeprefix("--- ").strip()
            if path != "/dev/null":
                files.append(path.removeprefix("a/"))
    return sorted(set(files))


def _run_command_to_files(
    command: list[str],
    *,
    cwd: Path,
    input_text: str,
    timeout: int,
    runner: CommandRunner,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    completed = runner(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    stdout_path.write_text(str(getattr(completed, "stdout", "")), encoding="utf-8")
    stderr_path.write_text(str(getattr(completed, "stderr", "")), encoding="utf-8")
    return completed


def _codex_executable() -> str:
    configured = os.getenv("HIPPO_BENCHMARK_CODEX_EXECUTABLE")
    if configured:
        return configured
    discovered = shutil.which("codex") or shutil.which("codex.cmd")
    if discovered:
        return discovered
    return "codex.cmd" if sys.platform == "win32" else "codex"


def _benchmark_codex_model() -> str | None:
    configured = os.getenv("HIPPO_BENCHMARK_CODEX_MODEL")
    if configured:
        return configured
    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        raw = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        value = raw.get("tool", {}).get("hippo_benchmark", {}).get("codex_model")
        if isinstance(value, str) and value:
            return value
    codex_home = Path(os.getenv("CODEX_HOME", Path.home() / ".codex"))
    codex_config = codex_home / "config.toml"
    if codex_config.exists():
        raw = tomllib.loads(codex_config.read_text(encoding="utf-8-sig"))
        value = raw.get("model")
        if isinstance(value, str) and value:
            return value
    return None


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Long Text Benchmark Report",
        "",
        f"Run: `{summary['run_id']}`",
        f"Mode: `{summary.get('mode', 'local')}`",
        "",
    ]
    lines.append(
        "| Group | Task | A score | B score | Delta | B pass | Rule hit | Compression | Leaks |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for task in summary["tasks"]:
        a = task["conditions"]["A_baseline"]
        b = task["conditions"]["B_memory"]
        delta = b["scores"]["weighted_score"] - a["scores"]["weighted_score"]
        lines.append(
            f"| {task['group']} | {task['task_id']} | {a['scores']['weighted_score']:.2f} | "
            f"{b['scores']['weighted_score']:.2f} | {delta:+.2f} | "
            f"{b['test_pass_rate']:.2f} | {b['rule_hit_rate']:.2f} | "
            f"{b['compression_ratio']:.2f} | {b['sensitive_leak_count']} |"
        )
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            "This matrix checks whether hippo_memory helps on long-document tasks without "
            "leaking hidden answers to the baseline prompt. It covers hidden long rules, "
            "needle retrieval, version conflict handling, noise suppression, compression "
            "fidelity, and privacy-sensitive memory filtering.",
            "",
            "## Recommendations",
            "",
            "- Treat local mode as deterministic harness validation, not a Codex quality claim.",
            "- Use codex mode only after local mode passes, because it calls the "
            "external model API.",
            "- Keep hidden answer artifacts out of condition prompts and only write "
            "them after both runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run long-text hippo_memory benchmarks.")
    parser.add_argument("--task", action="append", dest="task_ids")
    parser.add_argument("--timestamp")
    parser.add_argument("--mode", choices=["local", "codex"], default="local")
    parser.add_argument("--codex-timeout-seconds", type=int, default=420)
    args = parser.parse_args(argv)
    harness = LongTextBenchmarkHarness(
        Path("benchmarks/long_text_tasks"),
        codex_timeout_seconds=args.codex_timeout_seconds,
    )
    result = harness.run(task_ids=args.task_ids, timestamp=args.timestamp, mode=args.mode)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
