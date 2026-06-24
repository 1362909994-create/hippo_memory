from __future__ import annotations

import ast
import json
import re
import sys

from typer.testing import CliRunner

from hippocampus_memory.cli import app

CLI_COMMANDS = [
    "init",
    "project-init",
    "serve",
    "daemon",
    "write",
    "search",
    "pack",
    "auto-store",
    "auto-context",
    "callback",
    "callback-reset",
    "project-profile",
    "code-map",
    "code-graph",
    "code-symbols",
    "code-references",
    "code-intelligence",
    "code-diagnostics",
    "impact",
    "run",
    "mcp",
    "mcp-project",
    "mcp-codex",
    "codex-deploy",
    "doctor",
    "eval",
    "token-report",
    "token-ledger",
    "mcp-config",
    "daemon-script",
    "browser",
    "index-project",
    "summarize-session",
    "candidate-list",
    "candidate-accept",
    "candidate-discard",
    "conflict-list",
    "conflict-resolve",
    "queue-session",
    "consolidate",
    "forget",
    "memory-supersede",
    "project-summary",
    "stats",
]


def _invoke_ok(runner: CliRunner, args: list[str]):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def _parse_mapping(output: str) -> dict:
    return ast.literal_eval(output.strip())


def _parse_first_candidate_id(output: str) -> str:
    match = re.search(r"cand_[A-Za-z0-9_]+", output)
    assert match, output
    return match.group(0)


def _parse_first_conflict_id(output: str) -> str:
    match = re.search(r"cfl_[A-Za-z0-9_]+", output)
    assert match, output
    return match.group(0)


def test_cli_all_commands_render_help():
    runner = CliRunner()

    for command in CLI_COMMANDS:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, f"{command} help failed:\n{result.output}"
        assert command.split("-", 1)[0] in result.output


def test_cli_core_project_workflow_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "workflow.db"))
    runner = CliRunner()
    root = tmp_path / "demo"
    src = root / "src"
    src.mkdir(parents=True)
    (src / "app.py").write_text(
        "from helper import helper\n\n"
        "def main():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    (src / "helper.py").write_text(
        "def helper():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    _invoke_ok(runner, ["project-init", "demo", "--root", str(root)])
    indexed = _invoke_ok(runner, ["index-project", str(root), "--project", "demo"])
    assert "indexed_files" in indexed.output

    decision = _invoke_ok(
        runner,
        [
            "write",
            "--project",
            "demo",
            "--type",
            "decision",
            "--content",
            "Decision: use SQLite as the local durable memory store.",
            "--entity",
            "SQLite",
            "--tag",
            "storage",
        ],
    )
    constraint = _invoke_ok(
        runner,
        [
            "write",
            "--project",
            "demo",
            "--type",
            "constraint",
            "--content",
            "Constraint: do not recall private memories by default.",
            "--entity",
            "privacy",
            "--tag",
            "privacy",
        ],
    )
    assert decision.output.strip().startswith("mem_")
    assert constraint.output.strip().startswith("mem_")

    search = _invoke_ok(
        runner,
        ["search", "local memory store", "--project", "demo", "--entity", "SQLite"],
    )
    assert "SQLite" in search.output

    pack = _invoke_ok(runner, ["pack", "storage decision", "--project", "demo"])
    assert "Token savings:" in pack.output
    assert "Memory Pack:" in pack.output

    auto_context = _invoke_ok(
        runner,
        ["auto-context", "modify helper behavior", "--project", "demo", "--metadata"],
    )
    assert "token_savings" in auto_context.output

    callback = _invoke_ok(
        runner,
        [
            "callback",
            "continue helper behavior",
            "--project",
            "demo",
            "--session",
            "s1",
            "--metadata",
        ],
    )
    assert "included_memory_ids" in callback.output
    reset = _invoke_ok(runner, ["callback-reset", "--project", "demo", "--session", "s1"])
    assert "reset" in reset.output

    for args, expected in [
        (["project-profile", "--project", "demo"], "Project Profile:"),
        (["code-map", "--project", "demo"], "helper.py"),
        (["code-graph", "--project", "demo"], "helper"),
        (["code-symbols", "--project", "demo", "--query", "helper"], "helper"),
        (["code-references", "helper", "--project", "demo"], "app.py"),
        (["code-intelligence", "change helper", "--project", "demo"], "helper"),
        (["impact", "change helper", "--project", "demo"], "Code Impact Pack:"),
        (["project-summary", "--project", "demo"], "Constraint:"),
        (["stats"], "memories"),
    ]:
        result = _invoke_ok(runner, args)
        assert expected in result.output

    token_report = _invoke_ok(
        runner,
        ["token-report", "continue storage decision", "--project", "demo"],
    )
    assert "estimated_tokens_saved" in token_report.output
    ledger = _invoke_ok(runner, ["token-ledger", "--project", "demo"])
    assert "entry_count" in ledger.output

    context_file = tmp_path / "context.md"
    run_result = _invoke_ok(
        runner,
        [
            "run",
            "--project",
            "demo",
            "--intent",
            "change helper",
            "--inject",
            "file",
            "--context-file",
            str(context_file),
            "--no-code-map",
            "--",
            sys.executable,
            "-c",
            "import os, pathlib; print(pathlib.Path(os.environ['HIPPO_CONTEXT_FILE']).exists())",
        ],
    )
    assert "True" in run_result.output
    assert context_file.exists()


def test_cli_queue_conflict_forget_and_artifact_workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "advanced.db"))
    runner = CliRunner()

    dry_run = _invoke_ok(
        runner,
        [
            "auto-store",
            "--project",
            "demo",
            "--mode",
            "preview",
            "--text",
            "Decision: rank exact project memories above generic code chunks.",
        ],
    )
    assert "previewed" in dry_run.output

    queued = _invoke_ok(
        runner,
        [
            "auto-store",
            "--project",
            "demo",
            "--mode",
            "queue",
            "--text",
            "Decision: keep Codex MCP context short and architecture-focused.",
        ],
    )
    assert "'queued': 1" in queued.output
    candidate_id = _parse_first_candidate_id(queued.output)
    listed = _invoke_ok(runner, ["candidate-list", "--project", "demo"])
    assert candidate_id in listed.output
    accepted = _invoke_ok(runner, ["candidate-accept", candidate_id])
    assert "memory_id" in accepted.output

    old = _invoke_ok(
        runner,
        [
            "write",
            "--project",
            "demo",
            "--type",
            "constraint",
            "--content",
            "Display must use SPI interface.",
            "--entity",
            "Display",
        ],
    ).output.strip()
    new = _invoke_ok(
        runner,
        [
            "write",
            "--project",
            "demo",
            "--type",
            "constraint",
            "--content",
            "Display must use RGB interface.",
            "--entity",
            "Display",
        ],
    ).output.strip()
    superseded = _invoke_ok(runner, ["memory-supersede", old, new])
    assert "'updated': True" in superseded.output

    consolidated = _invoke_ok(runner, ["consolidate", "--project", "demo"])
    assert "conflict_count" in consolidated.output
    conflicts = _invoke_ok(runner, ["conflict-list", "--project", "demo"])
    conflict_id = _parse_first_conflict_id(conflicts.output)
    resolved = _invoke_ok(
        runner,
        ["conflict-resolve", conflict_id, "--resolution", "Use RGB", "--status", "resolved"],
    )
    assert "'updated': True" in resolved.output

    mcp_config = tmp_path / "mcp.json"
    daemon_script = tmp_path / "daemon.ps1"
    browser = tmp_path / "browser.html"
    _invoke_ok(runner, ["mcp-config", "--output", str(mcp_config), "--command", "python"])
    _invoke_ok(runner, ["daemon-script", "--output", str(daemon_script), "--port", "9876"])
    _invoke_ok(runner, ["browser", "--output", str(browser), "--project", "demo"])
    assert json.loads(mcp_config.read_text(encoding="utf-8"))["mcpServers"]
    assert "9876" in daemon_script.read_text(encoding="utf-8")
    assert "Codex" in browser.read_text(encoding="utf-8")

    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "query": "Codex MCP context wording",
                "project": "demo",
                "expected_contains": ["architecture-focused"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evaluated = _invoke_ok(runner, ["eval", str(benchmark)])
    assert "'hit_rate': 1.0" in evaluated.output

    forgotten = _invoke_ok(runner, ["forget", new])
    assert "'deleted': 1" in forgotten.output
    hard_deleted = _invoke_ok(runner, ["forget", "--project", "demo", "--hard"])
    deleted_count = _parse_mapping(hard_deleted.output)["deleted"]
    assert deleted_count >= 1


def test_cli_bad_inputs_are_clean_user_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "bad-inputs.db"))
    runner = CliRunner()

    invalid_mode = runner.invoke(
        app,
        ["auto-store", "--mode", "nope", "--text", "Decision: keep tests strict."],
    )
    assert invalid_mode.exit_code != 0
    assert "mode must be one of" in invalid_mode.output
    assert "Traceback" not in invalid_mode.output

    missing_command = runner.invoke(
        app,
        ["run", "--project", "demo", "--intent", "test", "--inject", "file"],
    )
    assert missing_command.exit_code != 0
    assert "a command is required unless --inject print is used" in missing_command.output
    assert "Traceback" not in missing_command.output

    invalid_inject = runner.invoke(
        app,
        ["run", "--project", "demo", "--intent", "test", "--inject", "nope"],
    )
    assert invalid_inject.exit_code != 0
    assert "unsupported inject mode: nope" in invalid_inject.output
    assert "Traceback" not in invalid_inject.output

    invalid_strategy = runner.invoke(
        app,
        [
            "run",
            "--project",
            "demo",
            "--intent",
            "test",
            "--inject",
            "print",
            "--bundle-strategy",
            "nope",
        ],
    )
    assert invalid_strategy.exit_code != 0
    assert "strategy must be one of: auto, full, lean, pack" in invalid_strategy.output
    assert "Traceback" not in invalid_strategy.output

    missing_project_path = runner.invoke(
        app,
        ["index-project", str(tmp_path / "missing-project"), "--project", "demo"],
    )
    assert missing_project_path.exit_code != 0
    assert "project path does not exist or is not a directory" in missing_project_path.output
    assert "Traceback" not in missing_project_path.output

    missing_project_path_without_name = runner.invoke(
        app,
        ["index-project", str(tmp_path / "missing-project")],
    )
    assert missing_project_path_without_name.exit_code != 0
    assert (
        "project path does not exist or is not a directory"
        in missing_project_path_without_name.output
    )
    assert "Traceback" not in missing_project_path_without_name.output
