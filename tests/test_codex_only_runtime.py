from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hippocampus_memory.cli import app
from hippocampus_memory.db import Database
from hippocampus_memory.deploy import codex_project_memory_block, deploy_codex
from hippocampus_memory.mcp_server import HippoMcpServer
from hippocampus_memory.memory_writer import MemoryWriter


def test_cli_exposes_codex_deploy_without_reasonix_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "codex-deploy" in result.output
    assert "reasonix" not in result.output.casefold()


def test_codex_project_memory_block_has_no_reasonix_ui_contract() -> None:
    block = codex_project_memory_block("demo")

    assert 'session_key="codex"' in block
    assert "hippo_memory_context_auto" in block
    assert "hippo_memory_memory_auto_store" in block
    assert "Reasonix" not in block
    assert "token_savings_text" not in block


def test_codex_deploy_writes_project_local_memory_prompt(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")

    result = deploy_codex(root, project="demo", index_project=False)

    agents = root / "AGENTS.md"
    assert result["codex_project_memory"] == str(agents)
    assert agents.exists()
    text = agents.read_text(encoding="utf-8")
    assert 'session_key="codex"' in text
    assert "Reasonix" not in text
    assert (root / ".hippo" / "hippo.db").exists()
    assert (root / ".hippo" / "hippo-mcp.ps1").exists()


def test_mcp_pack_survives_unwritable_scheduler_state(tmp_path: Path) -> None:
    db_path = tmp_path / "hippo.db"
    db = Database(db_path)
    db.initialize()
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content="Decision: Codex runtime should keep scheduler state failures non-fatal.",
        entities=["Codex", "MemoryScheduler"],
    )
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    bad_state_path = readonly_dir / "scheduler.json"
    bad_state_path.mkdir()

    server = HippoMcpServer(
        db,
        safe_tool_names=False,
        default_project="demo",
        scheduler_state_path=bad_state_path,
    )

    result = server.call_tool(
        "memory.pack",
        {"query": "Codex scheduler fallback", "project": "demo"},
    )

    assert "Memory Pack:" in str(result.get("text") or result.get("injected_context"))
    trace_text = str(result.get("execution_trace"))
    assert "scheduler_state_save_failed" in trace_text
