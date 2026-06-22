from __future__ import annotations

import sys

from typer.testing import CliRunner

from hippocampus_memory.cli import app


def test_cli_acceptance_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "cli.db"))
    runner = CliRunner()

    init_result = runner.invoke(app, ["init"])
    assert init_result.exit_code == 0

    constraint = runner.invoke(
        app,
        [
            "write",
            "--project",
            "glasses",
            "--type",
            "constraint",
            "--content",
            "用户不接受 3-4 cm 焦距的光学结构。",
        ],
    )
    assert constraint.exit_code == 0

    task_state = runner.invoke(
        app,
        [
            "write",
            "--project",
            "glasses",
            "--type",
            "task_state",
            "--content",
            "当前目标是先让 STM32 点亮 TFT 屏幕。",
        ],
    )
    assert task_state.exit_code == 0

    search = runner.invoke(app, ["search", "继续上次那个屏幕项目", "--project", "glasses"])
    assert search.exit_code == 0
    assert "STM32" in search.output

    pack = runner.invoke(
        app,
        ["pack", "继续上次那个 STM32 点亮 TFT 的项目", "--project", "glasses"],
    )
    assert pack.exit_code == 0
    assert "Memory Pack:" in pack.output
    assert "当前目标是先让 STM32 点亮 TFT 屏幕。" in pack.output

    profile = runner.invoke(app, ["project-profile", "--project", "glasses"])
    assert profile.exit_code == 0
    assert "Project Profile:" in profile.output

    impact = runner.invoke(
        app,
        ["impact", "change search ranking", "--project", "glasses"],
    )
    assert impact.exit_code == 0
    assert "Code Impact Pack:" in impact.output

    context_print = runner.invoke(
        app,
        [
            "run",
            "--project",
            "glasses",
            "--intent",
            "change search ranking",
            "--inject",
            "print",
            "--no-code-map",
        ],
    )
    assert context_print.exit_code == 0
    assert "Hippocampus Context Bundle" in context_print.output

    context_path = tmp_path / "context.md"
    env_result = runner.invoke(
        app,
        [
            "run",
            "--project",
            "glasses",
            "--intent",
            "change search ranking",
            "--inject",
            "env",
            "--context-file",
            str(context_path),
            "--",
            sys.executable,
            "-c",
            (
                "import os, pathlib; "
                "print(os.environ['HIPPO_PROJECT']); "
                "print('Hippocampus Context Bundle' in os.environ['HIPPO_CONTEXT']); "
                "print(pathlib.Path(os.environ['HIPPO_CONTEXT_FILE']).exists())"
            ),
        ],
    )
    assert env_result.exit_code == 0
    assert "glasses" in env_result.output
    assert "True" in env_result.output

    stdin_result = runner.invoke(
        app,
        [
            "run",
            "--project",
            "glasses",
            "--intent",
            "change search ranking",
            "--inject",
            "stdin",
            "--context-file",
            str(tmp_path / "stdin-context.md"),
            "--",
            sys.executable,
            "-c",
            "import sys; print('Hippocampus Context Bundle' in sys.stdin.read())",
        ],
    )
    assert stdin_result.exit_code == 0
    assert "True" in stdin_result.output

    arg_result = runner.invoke(
        app,
        [
            "run",
            "--project",
            "glasses",
            "--intent",
            "change search ranking",
            "--inject",
            "arg",
            "--context-file",
            str(tmp_path / "arg-context.md"),
            "--",
            sys.executable,
            "-c",
            "import sys; print('Hippocampus Context Bundle' in sys.argv[-1])",
        ],
    )
    assert arg_result.exit_code == 0
    assert "True" in arg_result.output


def test_summarize_session_write_requires_yes(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "session.db"))
    chat = tmp_path / "chat.txt"
    chat.write_text("用户：当前目标是先让 STM32 点亮 TFT 屏幕。\n", encoding="utf-8")
    runner = CliRunner()

    blocked = runner.invoke(
        app,
        ["summarize-session", str(chat), "--project", "glasses", "--write"],
    )
    assert blocked.exit_code != 0

    written = runner.invoke(
        app,
        ["summarize-session", str(chat), "--project", "glasses", "--write", "--yes"],
    )
    assert written.exit_code == 0
    assert "write_result" in written.output


def test_candidate_queue_accept_and_discard(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "candidates.db"))
    chat = tmp_path / "chat.txt"
    chat.write_text("用户：当前目标是先让 STM32 点亮 TFT 屏幕。\n", encoding="utf-8")
    runner = CliRunner()

    queued = runner.invoke(app, ["queue-session", str(chat), "--project", "glasses"])
    assert queued.exit_code == 0
    assert "candidate_ids" in queued.output

    listed = runner.invoke(app, ["candidate-list", "--project", "glasses"])
    assert listed.exit_code == 0
    assert "cand_" in listed.output
    candidate_id = listed.output.split("cand_", 1)[1].split("'", 1)[0]
    candidate_id = "cand_" + candidate_id

    accepted = runner.invoke(app, ["candidate-accept", candidate_id])
    assert accepted.exit_code == 0
    assert "memory_id" in accepted.output


def test_cli_token_ledger_is_project_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "tokens.db"))
    runner = CliRunner()

    write = runner.invoke(
        app,
        [
            "write",
            "--project",
            "alpha",
            "--type",
            "task_state",
            "--content",
            "Current task is to measure token savings.",
        ],
    )
    assert write.exit_code == 0

    report = runner.invoke(app, ["token-report", "measure token savings", "--project", "alpha"])
    assert report.exit_code == 0
    assert "ledger_ids" in report.output

    ledger = runner.invoke(app, ["token-ledger", "--project", "alpha"])
    assert ledger.exit_code == 0
    assert "entry_count" in ledger.output
    assert "alpha" in ledger.output


def test_cli_explain_memory_outputs_score_details(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "explain.db"))
    runner = CliRunner()

    write = runner.invoke(
        app,
        [
            "write",
            "--project",
            "alpha",
            "--type",
            "constraint",
            "--content",
            "Search must explain why a memory was recalled.",
            "--importance",
            "0.9",
        ],
    )
    assert write.exit_code == 0
    memory_id = write.output.strip()

    explained = runner.invoke(
        app,
        ["explain", memory_id, "--project", "alpha", "--query", "explain recalled memory"],
    )

    assert explained.exit_code == 0
    assert "why_recalled" in explained.output
    assert "score_details" in explained.output
    assert "project_match" in explained.output


def test_cli_pack_prints_run_and_total_token_savings(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "pack-token-stats.db"))
    runner = CliRunner()

    relevant = runner.invoke(
        app,
        [
            "write",
            "--project",
            "alpha",
            "--type",
            "task_state",
            "--content",
            "Current task is to show token savings in the CLI.",
        ],
    )
    assert relevant.exit_code == 0
    for index in range(8):
        filler = runner.invoke(
            app,
            [
                "write",
                "--project",
                "alpha",
                "--type",
                "technical_fact",
                "--content",
                (
                    f"Background note {index}: "
                    "this long unrelated implementation context is kept only "
                    "to make the naive baseline larger than the focused pack. "
                    "It should not be recalled for the token savings query. "
                ),
            ],
        )
        assert filler.exit_code == 0

    pack = runner.invoke(app, ["pack", "token savings CLI", "--project", "alpha"])
    assert pack.exit_code == 0
    assert "Token savings:" in pack.output
    assert "this run saved" in pack.output
    assert "project total saved" in pack.output
    assert "Memory Pack:" in pack.output

    ledger = runner.invoke(app, ["token-ledger", "--project", "alpha"])
    assert ledger.exit_code == 0
    assert "'entry_count': 1" in ledger.output
    assert "'saved_tokens':" in ledger.output


def test_cli_auto_context_metadata_includes_token_savings(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "auto-context-token-stats.db"))
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "write",
            "--project",
            "alpha",
            "--type",
            "task_state",
            "--content",
            "Current task is to continue the automatic context scheduler.",
        ],
    )

    result = runner.invoke(
        app,
        ["auto-context", "continue", "--project", "alpha", "--metadata"],
    )

    assert result.exit_code == 0
    assert "Token savings:" in result.output
    assert "'token_savings':" in result.output
    assert "'total_saved_tokens':" in result.output


def test_cli_callback_tracks_seen_memory_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "callback.db"))
    runner = CliRunner()
    write = runner.invoke(
        app,
        [
            "write",
            "--project",
            "alpha",
            "--type",
            "task_state",
            "--content",
            "Callback CLI memory should be injected once.",
        ],
    )
    assert write.exit_code == 0

    first = runner.invoke(
        app,
        [
            "callback",
            "callback cli memory",
            "--project",
            "alpha",
            "--session",
            "s1",
            "--metadata",
        ],
    )
    second = runner.invoke(
        app,
        [
            "callback",
            "callback cli memory",
            "--project",
            "alpha",
            "--session",
            "s1",
            "--metadata",
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "included_memory_ids" in first.output
    assert "excluded_memory_ids" in second.output


def test_cli_code_intelligence_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "code-intel.db"))
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text(
        "from b import helper\n\ndef main():\n    return helper()\n",
        encoding="utf-8",
    )
    (root / "b.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    runner = CliRunner()

    indexed = runner.invoke(app, ["index-project", str(root), "--project", "demo"])
    symbols = runner.invoke(app, ["code-symbols", "--project", "demo", "--query", "helper"])
    references = runner.invoke(app, ["code-references", "helper", "--project", "demo"])
    impact = runner.invoke(
        app,
        ["code-intelligence", "change helper behavior", "--project", "demo"],
    )

    assert indexed.exit_code == 0
    assert symbols.exit_code == 0
    assert "helper" in symbols.output
    assert references.exit_code == 0
    assert "a.py" in references.output
    assert impact.exit_code == 0
    assert "helper [function] b.py:1" in impact.output


def test_cli_code_diagnostics_refresh_handles_missing_checker(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "diagnostics.db"))
    root = tmp_path / "project"
    root.mkdir()
    runner = CliRunner()
    runner.invoke(app, ["project-init", "demo", "--root", str(root)])

    result = runner.invoke(
        app,
        [
            "code-diagnostics",
            "--project",
            "demo",
            "--path",
            str(root),
            "--checker",
            "definitely_missing_pyright_tool",
            "--refresh",
        ],
    )

    assert result.exit_code == 0
    assert "'available': False" in result.output
