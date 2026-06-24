from __future__ import annotations

from pathlib import Path

from hippocampus_memory.db import Database
from hippocampus_memory.mcp_server import HippoMcpServer
from hippocampus_memory.orchestrator.turn_orchestrator import TurnOrchestrator


def _make_non_empty_project(root: Path) -> None:
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'legacy'\n", encoding="utf-8")
    (root / "src" / "app.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")


def test_old_project_without_structure_memory_gets_codegraph_bootstrap_suggestion(
    db: Database, tmp_path: Path
) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    _make_non_empty_project(root)
    db.insert_or_update_project("legacy", root_path=str(root))

    result = TurnOrchestrator(db).run_turn(
        "understand this existing project",
        context={"project": "legacy", "operation": "auto_context", "writeback": False},
        mode="preview",
    )

    payload = result.runtime_payload()
    suggestion = payload["context_budget"]["codegraph_bootstrap"]
    assert suggestion["recommended"] is True
    assert suggestion["requires_user_approval"] is True
    assert suggestion["project_state"] == "existing_project_without_structure_memory"
    assert suggestion["tool"] == "codegraph_bootstrap"
    assert db.list_candidates(project="legacy") == []
    assert db.list_memories(project="legacy") == []


def test_empty_project_does_not_get_codegraph_bootstrap_suggestion(
    db: Database, tmp_path: Path
) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    db.insert_or_update_project("empty", root_path=str(root))

    result = TurnOrchestrator(db).run_turn(
        "start a new project",
        context={"project": "empty", "operation": "auto_context", "writeback": False},
        mode="preview",
    )

    payload = result.runtime_payload()
    assert "codegraph_bootstrap" not in payload["context_budget"]
    assert db.list_candidates(project="empty") == []


def test_codegraph_bootstrap_tool_preview_does_not_write(db: Database, tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    _make_non_empty_project(root)
    db.insert_or_update_project("legacy", root_path=str(root))
    server = HippoMcpServer(db, safe_tool_names=True, default_project="legacy")

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "codegraph_bootstrap",
                "arguments": {
                    "mode": "preview",
                    "codegraph_summary": (
                        "Project architecture: CLI routes to MCP. "
                        "Entry points: cli.py and mcp_server.py. "
                        "Scheduler and policy layers stay separated. Tests use pytest."
                    ),
                },
            },
        }
    )

    payload = response["result"]["structuredContent"]
    assert payload["mode"] == "preview"
    assert payload["previewed"] >= 1
    assert payload["queued"] == 0
    assert payload["written"] == 0
    assert payload["items"][0]["memory_type"] == "project_context"
    assert db.list_candidates(project="legacy") == []
    assert db.list_memories(project="legacy") == []


def test_codegraph_bootstrap_tool_queues_structural_memory_candidates(
    db: Database, tmp_path: Path
) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    _make_non_empty_project(root)
    db.insert_or_update_project("legacy", root_path=str(root))
    server = HippoMcpServer(db, safe_tool_names=True, default_project="legacy")

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "codegraph_bootstrap",
                "arguments": {
                    "mode": "queue",
                    "codegraph_summary": (
                        "Project architecture: CLI, MCP, and API call TurnOrchestrator. "
                        "Module boundaries: scheduler handles lifecycle, policy handles "
                        "routing, semantic handles meaning. Tests: pytest and ruff verify "
                        "compatibility."
                    ),
                },
            },
        }
    )

    payload = response["result"]["structuredContent"]
    candidates = db.list_candidates(project="legacy")
    assert payload["mode"] == "queue"
    assert payload["queued"] == len(candidates)
    assert payload["queued"] >= 2
    assert payload["written"] == 0
    assert {candidate["memory_type"] for candidate in candidates} >= {
        "project_context",
        "constraint",
    }
    assert all("CodeGraph bootstrap" in candidate["content"] for candidate in candidates)
