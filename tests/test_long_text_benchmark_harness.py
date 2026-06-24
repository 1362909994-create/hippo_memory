from __future__ import annotations

import json
from pathlib import Path

from benchmarks.long_text_tasks.scripts.harness import LongTextBenchmarkHarness, load_tasks


def test_long_text_catalog_contains_all_required_groups() -> None:
    tasks = load_tasks(Path("benchmarks/long_text_tasks/tasks"))
    groups = {task.group for task in tasks}

    assert {
        "G1_hidden_long_rule_task",
        "G2_needle_in_haystack",
        "G3_version_conflict",
        "G4_noise_suppression",
        "G5_compression_fidelity",
        "G6_privacy_sensitive",
    }.issubset(groups)
    assert len(tasks) >= 8


def test_smoke_long_text_harness_generates_required_artifacts(tmp_path: Path) -> None:
    harness = LongTextBenchmarkHarness(
        benchmark_root=Path("benchmarks/long_text_tasks"),
        runs_root=tmp_path / "runs",
        reports_root=tmp_path / "reports",
    )

    result = harness.run(task_ids=["G1_smoke_hidden_long_rules"], timestamp="20260623T150000Z")

    run_root = tmp_path / "runs" / "20260623T150000Z" / "G1_smoke_hidden_long_rules"
    required = [
        "task_prompt.md",
        "A_baseline/prompt.md",
        "B_memory/prompt.md",
        "hidden_answer_key.json",
        "memory_seed_document.md",
        "A_baseline/patch.diff",
        "B_memory/patch.diff",
        "A_baseline/test.log",
        "B_memory/test.log",
        "B_memory/memory_call_log.json",
        "B_memory/injected_context.md",
        "scores.json",
        "benchmark_report.md",
    ]
    assert result["task_count"] == 1
    assert all((run_root / relative).exists() for relative in required)
    assert (tmp_path / "reports" / "long_text_benchmark_report.md").exists()


def test_smoke_long_text_memory_beats_baseline_and_suppresses_noise(tmp_path: Path) -> None:
    harness = LongTextBenchmarkHarness(
        benchmark_root=Path("benchmarks/long_text_tasks"),
        runs_root=tmp_path / "runs",
        reports_root=tmp_path / "reports",
    )

    result = harness.run(task_ids=["G1_smoke_hidden_long_rules"], timestamp="20260623T151000Z")
    task = result["tasks"][0]
    a_score = task["conditions"]["A_baseline"]["scores"]["weighted_score"]
    b_score = task["conditions"]["B_memory"]["scores"]["weighted_score"]

    assert b_score - a_score >= 4.0
    assert task["conditions"]["B_memory"]["test_pass_rate"] == 1.0
    assert task["conditions"]["B_memory"]["rule_hit_rate"] == 1.0
    assert task["conditions"]["B_memory"]["irrelevant_injection_rate"] <= 0.1


def test_smoke_long_text_baseline_prompt_does_not_leak_hidden_rules(tmp_path: Path) -> None:
    harness = LongTextBenchmarkHarness(
        benchmark_root=Path("benchmarks/long_text_tasks"),
        runs_root=tmp_path / "runs",
        reports_root=tmp_path / "reports",
    )

    harness.run(task_ids=["G1_smoke_hidden_long_rules"], timestamp="20260623T152000Z")
    run_root = tmp_path / "runs" / "20260623T152000Z" / "G1_smoke_hidden_long_rules"
    baseline_prompt = (run_root / "A_baseline" / "prompt.md").read_text(encoding="utf-8")
    injected_context = (run_root / "B_memory" / "injected_context.md").read_text(encoding="utf-8")
    key = json.loads((run_root / "hidden_answer_key.json").read_text(encoding="utf-8"))

    assert "ACTIVE-RULE" not in baseline_prompt
    assert all(lane not in baseline_prompt for lane in key["lanes"].values())
    assert "ACTIVE-RULE" in injected_context
    assert all(rule_id in injected_context for rule_id in key["relevant_rule_ids"])


def test_full_local_long_text_benchmark_covers_required_group_metrics(tmp_path: Path) -> None:
    harness = LongTextBenchmarkHarness(
        benchmark_root=Path("benchmarks/long_text_tasks"),
        runs_root=tmp_path / "runs",
        reports_root=tmp_path / "reports",
    )

    result = harness.run(timestamp="20260623T170000Z")
    by_group = {task["group"]: task for task in result["tasks"]}

    assert by_group["G1_hidden_long_rule_task"]["conditions"]["B_memory"]["test_pass_rate"] == 1.0
    for task in result["tasks"]:
        b = task["conditions"]["B_memory"]
        a_score = task["conditions"]["A_baseline"]["scores"]["weighted_score"]
        assert b["scores"]["weighted_score"] - a_score >= 4.0
        assert b["irrelevant_injection_rate"] <= 0.1
        assert b["injected_token_count"] < b["raw_retrieved_token_count"]

    for task in result["tasks"]:
        if task["group"] == "G2_needle_in_haystack":
            assert task["conditions"]["B_memory"]["exact_rule_recall"] == 1.0

    conflict = by_group["G3_version_conflict"]["conditions"]["B_memory"]
    assert conflict["active_version_selection"] is True
    assert conflict["stale_rule_usage_count"] == 0
    assert conflict["contradiction_detection"] is True

    noise = by_group["G4_noise_suppression"]["conditions"]["B_memory"]
    assert noise["noise_injection_rate"] <= 0.1
    assert noise["relevant_memory_rank"] == 1
    assert noise["rejected_memory_count"] >= 4

    compression = by_group["G5_compression_fidelity"]["conditions"]["B_memory"]
    assert compression["compression_ratio"] >= 0.5
    assert compression["critical_rule_preservation"] >= 0.9
    assert compression["hallucinated_rule_count"] == 0

    privacy = by_group["G6_privacy_sensitive"]["conditions"]["B_memory"]
    assert privacy["sensitive_leak_count"] == 0
    assert privacy["public_rule_recall"] == 1.0
    assert privacy["default_privacy_safety"] is True


def test_codex_mode_does_not_write_hidden_artifacts_before_baseline_exec(tmp_path: Path) -> None:
    seen_baseline = False

    def fake_runner(command: list[str], **kwargs: object) -> object:
        nonlocal seen_baseline
        cwd = Path(kwargs["cwd"])  # type: ignore[index]
        condition_root = cwd.parent
        task_root = condition_root.parent
        if condition_root.name == "A_baseline":
            seen_baseline = True
            assert not (task_root / "hidden_answer_key.json").exists()
            assert not (task_root / "memory_seed_document.md").exists()
        if "--output-last-message" in command:
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text("ok", encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "{}\n", "stderr": ""})()

    harness = LongTextBenchmarkHarness(
        benchmark_root=Path("benchmarks/long_text_tasks"),
        runs_root=tmp_path / "runs",
        reports_root=tmp_path / "reports",
        command_runner=fake_runner,
    )

    harness.run(
        task_ids=["G1_smoke_hidden_long_rules"],
        timestamp="20260623T171000Z",
        mode="codex",
    )

    assert seen_baseline is True
