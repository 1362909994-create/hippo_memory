from __future__ import annotations

from pathlib import Path

import pytest

from hippocampus_memory.db import Database
from hippocampus_memory.deploy import deploy_codex
from hippocampus_memory.mcp_server import HippoMcpServer
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.turn_orchestrator import TurnOrchestrator


def test_corrupt_scheduler_state_file_falls_back_without_crashing(tmp_path: Path) -> None:
    db = _db(tmp_path, "corrupt")
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content="Decision: corrupt scheduler state must fall back for Codex turns.",
        tags=["scheduler", "resilience"],
    )
    state_path = tmp_path / "scheduler.json"
    state_path.write_text("{not valid json", encoding="utf-8")

    result = TurnOrchestrator(db, scheduler_state_path=state_path).run_turn(
        "Codex corrupt scheduler state fallback",
        context={"project": "demo", "operation": "memory_pack", "writeback": False},
    )

    scheduler = result.turn_context.context_budget["memory_scheduler_report"]
    assert "Memory Pack:" in result.injected_context
    assert scheduler["state_version"]
    assert scheduler["persistence_report"]["event"] == "scheduler_state_saved"


def test_invalid_user_input_returns_safe_fallback_not_exception(tmp_path: Path) -> None:
    db = _db(tmp_path, "invalid")

    result = TurnOrchestrator(db, scheduler_state_path=tmp_path / "scheduler.json").run_turn(
        "@@@invalid memory command###",
        context={"project": "demo", "operation": "memory_pack", "writeback": False},
    )

    assert result.injected_context
    assert result.trace
    assert result.turn_context.context_budget["memory_scheduler_report"]


def test_mcp_unknown_tool_returns_jsonrpc_error_without_crashing(tmp_path: Path) -> None:
    server = HippoMcpServer(_db(tmp_path, "mcp"), default_project="demo")

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "does_not_exist", "arguments": {}},
        }
    )

    assert response["id"] == 7
    assert response["error"]["code"] == -32000
    assert "unknown tool" in response["error"]["message"].casefold()


def test_codex_deploy_rejects_file_root_without_partial_project_setup(tmp_path: Path) -> None:
    file_root = tmp_path / "not-a-directory.txt"
    file_root.write_text("not a project", encoding="utf-8")

    with pytest.raises(ValueError, match="not exist or is not a directory"):
        deploy_codex(file_root, project="bad", index_project=False)

    assert not (tmp_path / ".hippo.toml").exists()
    assert not (tmp_path / ".hippo").exists()


def _db(tmp_path: Path, name: str) -> Database:
    db = Database(tmp_path / f"{name}.db")
    db.initialize()
    return db
