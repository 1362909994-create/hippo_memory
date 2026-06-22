from __future__ import annotations

import ast
from pathlib import Path

from typer.testing import CliRunner

from hippocampus_memory.api import create_app
from hippocampus_memory.cli import app as cli_app
from hippocampus_memory.config import Settings
from hippocampus_memory.db import Database
from hippocampus_memory.mcp_server import HippoMcpServer
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.schemas import AutoContextRequest

ENTRYPOINTS = [
    Path("hippocampus_memory/cli.py"),
    Path("hippocampus_memory/mcp_server.py"),
    Path("hippocampus_memory/api.py"),
]

FORBIDDEN_RUNTIME_IMPORTS = {
    "hippocampus_memory.callback": {"callback_pack"},
    "hippocampus_memory.context_bundle": {"ContextBundleBuilder"},
    "hippocampus_memory.memory_policy": {"auto_store_memories"},
    "hippocampus_memory.packer": {"MemoryPacker"},
    "hippocampus_memory.ranker": {"RANKER_VERSION", "explain_memory_score"},
    "hippocampus_memory.recall_policy": {"build_auto_context", "decide_recall"},
    "hippocampus_memory.retriever": {"Retriever"},
}

FORBIDDEN_RUNTIME_CALLS = {
    "callback_pack",
    "ContextBundleBuilder",
    "auto_store_memories",
    "MemoryPacker",
    "explain_memory_score",
    "build_auto_context",
    "decide_recall",
    "Retriever",
}


def test_cli_mcp_api_route_runtime_calls_through_turn_orchestrator() -> None:
    for path in ENTRYPOINTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_orchestrator = False
        violations: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                if node.module == "hippocampus_memory.orchestrator":
                    imported_orchestrator = "TurnOrchestrator" in names
                forbidden = FORBIDDEN_RUNTIME_IMPORTS.get(node.module or "")
                if forbidden:
                    bad_names = sorted(names & forbidden)
                    for name in bad_names:
                        violations.append(f"forbidden import {node.module}.{name}")
            elif isinstance(node, ast.Call):
                call_name = _call_name(node.func)
                if call_name in FORBIDDEN_RUNTIME_CALLS:
                    violations.append(f"forbidden call {call_name}()")

        assert imported_orchestrator, f"{path} must import TurnOrchestrator"
        assert not violations, f"{path} bypasses TurnOrchestrator: {violations}"


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_cli_auto_context_metadata_exposes_turn_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "cli-routing.db"))
    runner = CliRunner()
    write = runner.invoke(
        cli_app,
        [
            "write",
            "--project",
            "demo",
            "--type",
            "task_state",
            "--content",
            "Current task is to route CLI through the orchestrator.",
        ],
    )
    assert write.exit_code == 0

    result = runner.invoke(
        cli_app,
        ["auto-context", "continue routing", "--project", "demo", "--metadata"],
    )

    assert result.exit_code == 0
    assert "execution_trace" in result.output
    assert "retrieved_memories" in result.output
    assert "selected_memories" in result.output
    assert "injected_context" in result.output


def test_mcp_auto_context_exposes_turn_result(db) -> None:
    MemoryWriter(db).write(
        project="demo",
        memory_type="task_state",
        content="Current task is to route MCP through the orchestrator.",
    )
    server = HippoMcpServer(db, safe_tool_names=True, default_project="demo")

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "context_auto",
                "arguments": {"intent": "continue routing", "session_key": "routing"},
            },
        }
    )

    payload = response["result"]["structuredContent"]
    assert payload["execution_trace"]
    assert payload["retrieved_memories"]
    assert payload["selected_memories"]
    assert payload["injected_context"] == payload["text"]


def test_api_auto_context_exposes_turn_result(tmp_path) -> None:
    settings = Settings(db_path=tmp_path / "api-routing.db")
    db = Database(settings=settings)
    db.initialize()
    MemoryWriter(db).write(
        project="demo",
        memory_type="task_state",
        content="Current task is to route API through the orchestrator.",
    )
    app = create_app(settings)
    endpoint = next(
        route.endpoint for route in app.routes if getattr(route, "path", None) == "/context/auto"
    )

    payload = endpoint(
        AutoContextRequest(intent="continue routing", project="demo", session_key="routing")
    )

    assert payload["execution_trace"]
    assert payload["retrieved_memories"]
    assert payload["selected_memories"]
    assert payload["injected_context"] == payload["text"]
