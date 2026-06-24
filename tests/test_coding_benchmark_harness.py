from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from benchmarks.coding_tasks.harness import (
    CodingBenchmarkHarness,
    _build_dynamic_long_policy,
    _diff_trees,
    _extract_memory_context,
    _run_command_to_files,
    load_task_specs,
)


def test_catalog_contains_persistent_tasks_split_across_abc_and_long_context() -> None:
    specs = load_task_specs(Path("benchmarks/coding_tasks/tasks"))

    assert len(specs) == 8
    assert [spec.category for spec in specs].count("A_negative_control") == 2
    assert [spec.category for spec in specs].count("B_project_context") == 2
    assert [spec.category for spec in specs].count("C_memory_dependent") == 4
    assert any(spec.task_id == "C03_long_context_routing_memory" for spec in specs)
    assert any(spec.task_id == "C04_dynamic_long_context_memory" for spec in specs)
    assert all(spec.fixture_path.exists() for spec in specs)
    assert all(spec.test_command for spec in specs)


def test_test_command_expands_task_dir_placeholder(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def fake_runner(
        command: list[str],
        *,
        cwd: Path,
        input: str | None = None,
        capture_output: bool = True,
        text: bool = True,
        encoding: str = "utf-8",
        errors: str = "replace",
        timeout: int = 120,
        check: bool = False,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, input, capture_output, text, encoding, errors, timeout, check, kwargs
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, "1 passed in 0.01s\n", "")

    harness = CodingBenchmarkHarness(
        benchmark_root=Path("benchmarks/coding_tasks"),
        runs_root=tmp_path / "runs",
        command_runner=fake_runner,
    )

    harness.run(
        task_ids=["C03_long_context_routing_memory"],
        timestamp="20260623T030000Z",
        mode="dry_run",
    )

    assert any("hidden_check.py" in " ".join(command) for command in seen)
    assert all("{task_dir}" not in part for command in seen for part in command)


def test_dynamic_long_context_task_uses_generated_check_and_memory(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def fake_runner(
        command: list[str],
        *,
        cwd: Path,
        input: str | None = None,
        capture_output: bool = True,
        text: bool = True,
        encoding: str = "utf-8",
        errors: str = "replace",
        timeout: int = 120,
        check: bool = False,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, input, capture_output, text, encoding, errors, timeout, check, kwargs
        seen.append(command)
        command_text = " ".join(command)
        assert "{generated_check}" not in command_text
        if "generated_hidden_check.py" in command_text:
            generated_check = Path(command[-1])
            assert generated_check.exists()
            check_text = generated_check.read_text(encoding="utf-8")
            assert "lane_" in check_text
        return subprocess.CompletedProcess(command, 0, "6 passed in 0.01s\n", "")

    harness = CodingBenchmarkHarness(
        benchmark_root=Path("benchmarks/coding_tasks"),
        runs_root=tmp_path / "runs",
        command_runner=fake_runner,
    )

    harness.run(
        task_ids=["C04_dynamic_long_context_memory"],
        timestamp="20260623T040000Z",
        mode="dry_run",
    )

    b_root = tmp_path / "runs" / "20260623T040000Z" / "B_memory" / "C04_dynamic_long_context_memory"
    memory_context = (b_root / "memory_context.md").read_text(encoding="utf-8")
    assert "Dynamic long-context routing archive" in memory_context
    assert "Section 25 - current policy table begins" in memory_context
    assert "lane_" in memory_context
    assert any("generated_hidden_check.py" in " ".join(command) for command in seen)


def test_dynamic_hidden_check_can_import_workspace_module(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    condition_root = tmp_path / "condition"
    workspace.mkdir()
    condition_root.mkdir()
    policy = _build_dynamic_long_policy("C04_dynamic_long_context_memory", "import-test")
    check_path = condition_root / "generated_hidden_check.py"
    check_path.write_text(policy.hidden_check, encoding="utf-8")
    (workspace / "routing_policy.py").write_text(
        "from __future__ import annotations\n\n"
        "def route_case(ticket: dict) -> str:\n"
        "    return 'standard'\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(check_path)],
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert "ModuleNotFoundError" not in completed.stderr


def test_dry_run_generates_ab_artifacts_without_codex_execution(tmp_path: Path) -> None:
    harness = CodingBenchmarkHarness(
        benchmark_root=Path("benchmarks/coding_tasks"),
        runs_root=tmp_path / "runs",
    )

    result = harness.run(
        task_ids=["A01_exercism_isogram"],
        timestamp="20260623T000000Z",
        mode="dry_run",
    )
    run_root = tmp_path / "runs" / "20260623T000000Z"

    assert result["mode"] == "dry_run"
    assert result["codex_executed"] is False
    assert (run_root / "A_baseline" / "A01_exercism_isogram" / "workspace").exists()
    assert (run_root / "B_memory" / "A01_exercism_isogram" / "workspace").exists()

    for condition in ["A_baseline", "B_memory"]:
        task_root = run_root / condition / "A01_exercism_isogram"
        assert (task_root / "prompt.md").exists()
        assert (task_root / "patch.diff").exists()
        assert (task_root / "pytest.log").exists()
        assert (task_root / "score.json").exists()

    assert (run_root / "scores.json").exists()
    assert (run_root / "benchmark_report.md").exists()
    scores = json.loads((run_root / "scores.json").read_text(encoding="utf-8"))
    assert scores["tasks"][0]["task_id"] == "A01_exercism_isogram"
    assert scores["tasks"][0]["conditions"]["A_baseline"]["codex_executed"] is False
    assert scores["tasks"][0]["conditions"]["B_memory"]["memory_mcp_called"] is False


def test_append_mode_preserves_existing_task_scores(tmp_path: Path) -> None:
    harness = CodingBenchmarkHarness(
        benchmark_root=Path("benchmarks/coding_tasks"),
        runs_root=tmp_path / "runs",
    )

    harness.run(
        task_ids=["A01_exercism_isogram"],
        timestamp="20260623T020000Z",
        mode="dry_run",
    )
    result = harness.run(
        task_ids=["A02_exercism_raindrops"],
        timestamp="20260623T020000Z",
        mode="dry_run",
        append=True,
    )

    run_root = tmp_path / "runs" / "20260623T020000Z"
    scores = json.loads((run_root / "scores.json").read_text(encoding="utf-8"))
    task_ids = [task["task_id"] for task in scores["tasks"]]
    assert task_ids == ["A01_exercism_isogram", "A02_exercism_raindrops"]
    assert [task["task_id"] for task in result["tasks"]] == task_ids
    assert (run_root / "A_baseline" / "A01_exercism_isogram" / "score.json").exists()
    assert (run_root / "A_baseline" / "A02_exercism_raindrops" / "score.json").exists()


def test_codex_mode_generates_logs_patch_and_memory_probe_with_fake_runner(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        *,
        cwd: Path,
        input: str | None = None,
        capture_output: bool = True,
        text: bool = True,
        encoding: str = "utf-8",
        errors: str = "replace",
        timeout: int = 120,
        check: bool = False,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        del input, capture_output, text, encoding, errors, check
        calls.append(command)
        command_text = " ".join(command)
        if "mcp-project" in command_text:
            payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "structuredContent": {
                        "text": "Memory Pack:\n- fake memory context",
                        "execution_trace": [{"node_id": "rank_memories"}],
                    }
                },
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload) + "\n", "")
        if Path(command[0]).name.casefold().startswith("codex") and command[1] == "exec":
            env = kwargs.get("env")
            assert env is not None
            assert Path(str(env.get("CODEX_HOME", ""))).name == "codex_home"
            assert Path(str(env.get("UV_CACHE_DIR", ""))).parent.name == "codex_home"
            assert Path(str(env.get("PIP_CACHE_DIR", ""))).parent.name == "codex_home"
            assert env.get("PYTHONDONTWRITEBYTECODE") == "1"
            assert "--dangerously-bypass-approvals-and-sandbox" in command
            assert "--ignore-user-config" in command
            assert "--ignore-rules" in command
            assert "-s" not in command
            assert timeout == 123
            output_index = command.index("--output-last-message") + 1
            output_path = Path(command[output_index])
            assert output_path.parent == cwd
            (cwd / "isogram.py").write_text(
                "def is_isogram(text: str) -> bool:\n"
                "    letters = [char.lower() for char in text if char.isalpha()]\n"
                "    return len(letters) == len(set(letters))\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command,
                0,
                '{"type":"agent_message","message":"done"}\n',
                "",
            )
        if "pytest" in command_text:
            return subprocess.CompletedProcess(command, 0, "5 passed in 0.01s\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    harness = CodingBenchmarkHarness(
        benchmark_root=Path("benchmarks/coding_tasks"),
        runs_root=tmp_path / "runs",
        command_runner=fake_runner,
        codex_timeout_seconds=123,
    )

    result = harness.run(
        task_ids=["A01_exercism_isogram"],
        timestamp="20260623T010000Z",
        mode="codex",
    )
    run_root = tmp_path / "runs" / "20260623T010000Z"
    b_root = run_root / "B_memory" / "A01_exercism_isogram"

    assert result["codex_executed"] is True
    assert result["tasks"][0]["conditions"]["A_baseline"]["codex_executed"] is True
    assert result["tasks"][0]["conditions"]["B_memory"]["memory_mcp_called"] is True
    assert (b_root / "memory_probe.json").exists()
    assert (b_root / "memory_context.md").read_text(encoding="utf-8").startswith("Memory Pack")
    assert (b_root / "codex.jsonl").exists()
    assert "isogram.py" in (b_root / "patch.diff").read_text(encoding="utf-8")
    assert any(
        Path(command[0]).name.casefold().startswith("codex") and command[1] == "exec"
        for command in calls
    )
    if sys.platform == "win32":
        assert any(Path(command[0]).suffix.casefold() == ".cmd" for command in calls)
    assert any("mcp-project" in " ".join(command) for command in calls)


def test_codex_timeout_is_recorded_as_failed_condition(tmp_path: Path) -> None:
    def fake_runner(
        command: list[str],
        *,
        cwd: Path,
        input: str | None = None,
        capture_output: bool = True,
        text: bool = True,
        encoding: str = "utf-8",
        errors: str = "replace",
        timeout: int = 120,
        check: bool = False,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, input, capture_output, text, encoding, errors, check, kwargs
        command_text = " ".join(command)
        if Path(command[0]).name.casefold().startswith("codex") and command[1] == "exec":
            raise subprocess.TimeoutExpired(
                cmd=command,
                timeout=timeout,
                output='{"type":"agent_message","message":"started"}\n',
                stderr="timeout while waiting for tool execution",
            )
        if "mcp-project" in command_text:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "pytest" in command_text:
            return subprocess.CompletedProcess(command, 1, "1 failed in 0.01s\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    harness = CodingBenchmarkHarness(
        benchmark_root=Path("benchmarks/coding_tasks"),
        runs_root=tmp_path / "runs",
        command_runner=fake_runner,
        codex_timeout_seconds=5,
    )

    result = harness.run(
        task_ids=["A01_exercism_isogram"],
        timestamp="20260623T011500Z",
        mode="codex",
    )

    a_root = tmp_path / "runs" / "20260623T011500Z" / "A_baseline" / "A01_exercism_isogram"
    assert result["tasks"][0]["conditions"]["A_baseline"]["codex_returncode"] == 124
    assert result["tasks"][0]["conditions"]["B_memory"]["memory_mcp_called"] is True
    assert (a_root / "codex.jsonl").read_text(encoding="utf-8").strip()
    assert "timeout" in (a_root / "codex_stderr.log").read_text(encoding="utf-8")
    assert (a_root / "score.json").exists()


def test_codex_mode_passes_configured_benchmark_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_commands: list[list[str]] = []
    monkeypatch.setenv("HIPPO_BENCHMARK_CODEX_MODEL", "gpt-test-supported")

    def fake_runner(
        command: list[str],
        *,
        cwd: Path,
        input: str | None = None,
        capture_output: bool = True,
        text: bool = True,
        encoding: str = "utf-8",
        errors: str = "replace",
        timeout: int = 120,
        check: bool = False,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, input, capture_output, text, encoding, errors, timeout, check, kwargs
        command_text = " ".join(command)
        if Path(command[0]).name.casefold().startswith("codex") and command[1] == "exec":
            codex_commands.append(command)
            return subprocess.CompletedProcess(command, 0, "{}\n", "")
        if "mcp-project" in command_text:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "pytest" in command_text:
            return subprocess.CompletedProcess(command, 0, "5 passed in 0.01s\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    harness = CodingBenchmarkHarness(
        benchmark_root=Path("benchmarks/coding_tasks"),
        runs_root=tmp_path / "runs",
        command_runner=fake_runner,
    )

    harness.run(
        task_ids=["A01_exercism_isogram"],
        timestamp="20260623T014500Z",
        mode="codex",
    )

    assert codex_commands
    for command in codex_commands:
        model_index = command.index("--model") + 1
        assert command[model_index] == "gpt-test-supported"


def test_codex_nonzero_returncode_invalidates_condition_score(tmp_path: Path) -> None:
    def fake_runner(
        command: list[str],
        *,
        cwd: Path,
        input: str | None = None,
        capture_output: bool = True,
        text: bool = True,
        encoding: str = "utf-8",
        errors: str = "replace",
        timeout: int = 120,
        check: bool = False,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, input, capture_output, text, encoding, errors, timeout, check, kwargs
        command_text = " ".join(command)
        if Path(command[0]).name.casefold().startswith("codex") and command[1] == "exec":
            return subprocess.CompletedProcess(command, 1, "", "model refresh failed")
        if "mcp-project" in command_text:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "pytest" in command_text:
            return subprocess.CompletedProcess(command, 0, "5 passed in 0.01s\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    harness = CodingBenchmarkHarness(
        benchmark_root=Path("benchmarks/coding_tasks"),
        runs_root=tmp_path / "runs",
        command_runner=fake_runner,
    )

    result = harness.run(
        task_ids=["A01_exercism_isogram"],
        timestamp="20260623T013000Z",
        mode="codex",
    )

    score = result["tasks"][0]["conditions"]["A_baseline"]
    assert score["test_pass_rate"] == 1.0
    assert score["codex_returncode"] == 1
    assert score["run_valid"] is False
    assert score["invalid_reason"] == "codex_returncode_nonzero"
    assert score["scores"]["overall"] == 0.0


def test_memory_context_extraction_deduplicates_mcp_content() -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [{"type": "text", "text": "Memory Pack:\n- same"}],
            "structuredContent": {"text": "Memory Pack:\n- same"},
        },
    }

    context = _extract_memory_context(json.dumps(payload) + "\n")

    assert context == "Memory Pack:\n- same"


def test_memory_context_extraction_includes_structured_retrieved_memories() -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "structuredContent": {
                "text": "Memory Pack:\nNo strong memory found for this query.",
                "retrieved_memories": [
                    {
                        "memory_id": "mem_seed",
                        "content": "Prior decision: keep scheduler boundaries stable.",
                        "score": 0.65,
                    }
                ],
            }
        },
    }

    context = _extract_memory_context(json.dumps(payload) + "\n")

    assert "No strong memory found" not in context
    assert "mem_seed" in context
    assert "Prior decision: keep scheduler boundaries stable." in context


def test_diff_ignores_uv_cache_artifacts(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    (before / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (after / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    uv_cache = after / ".uv-cache" / "archive-v0"
    uv_cache.mkdir(parents=True)
    (uv_cache / "dependency.py").write_text("generated cache\n", encoding="utf-8")

    assert _diff_trees(before, after) == ""


def test_file_backed_command_does_not_wait_for_grandchild_stdout_handles(
    tmp_path: Path,
) -> None:
    script = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)']); "
        "print('parent done')"
    )

    started = time.monotonic()
    completed = _run_command_to_files(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        input_text=None,
        timeout=1,
        env=None,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == 0
    assert elapsed < 1.5
    assert "parent done" in completed.stdout
