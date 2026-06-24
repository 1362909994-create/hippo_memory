from __future__ import annotations

from typer.testing import CliRunner

from hippocampus_memory.api import create_app
from hippocampus_memory.cli import app as cli_app
from hippocampus_memory.config import Settings
from hippocampus_memory.db import Database
from hippocampus_memory.mcp_server import HippoMcpServer
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.schemas import AutoContextRequest, MemoryPackRequest, MemorySearchRequest


def test_cli_commands_still_expose_existing_user_facing_entries() -> None:
    result = CliRunner().invoke(cli_app, ["--help"])

    assert result.exit_code == 0
    for command in [
        "write",
        "search",
        "pack",
        "auto-context",
        "auto-store",
        "mcp",
        "mcp-project",
        "doctor",
        "token-report",
        "codex-deploy",
    ]:
        assert command in result.output


def test_cli_auto_context_regression_keeps_metadata_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "cli.db"))
    runner = CliRunner()
    write = runner.invoke(
        cli_app,
        [
            "write",
            "--project",
            "regression",
            "--type",
            "task_state",
            "--content",
            "Current task is regression testing orchestrator routing.",
        ],
    )
    assert write.exit_code == 0

    result = runner.invoke(
        cli_app,
        ["auto-context", "continue regression testing", "--project", "regression", "--metadata"],
    )

    assert result.exit_code == 0
    assert "execution_trace" in result.output
    assert "retrieved_memories" in result.output
    assert "selected_memories" in result.output
    assert "injected_context" in result.output


def test_mcp_tool_response_shape_is_backward_compatible(db) -> None:
    MemoryWriter(db).write(
        project="regression",
        memory_type="task_state",
        content="Current task is MCP regression testing.",
    )
    server = HippoMcpServer(db, safe_tool_names=True, default_project="regression")

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "context_auto",
                "arguments": {"intent": "continue MCP regression", "session_key": "reg"},
            },
        }
    )
    payload = response["result"]["structuredContent"]

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert payload["text"] == payload["injected_context"]
    assert payload["execution_trace"]
    assert payload["retrieved_memories"]


def test_api_memory_endpoints_keep_legacy_fields_and_add_turn_metadata(tmp_path) -> None:
    settings = Settings(db_path=tmp_path / "api.db")
    db = Database(settings=settings)
    db.initialize()
    MemoryWriter(db).write(
        project="regression",
        memory_type="task_state",
        content="Current task is API regression testing.",
    )
    app = create_app(settings)
    routes = {getattr(route, "path", None): route.endpoint for route in app.routes}

    auto_payload = routes["/context/auto"](
        AutoContextRequest(
            intent="continue API regression",
            project="regression",
            session_key="reg",
        )
    )
    search_payload = routes["/memory/search"](
        MemorySearchRequest(query="API regression", project="regression")
    )
    pack_payload = routes["/memory/pack"](
        MemoryPackRequest(query="API regression", project="regression")
    )

    assert auto_payload["text"] == auto_payload["injected_context"]
    assert auto_payload["execution_trace"]
    assert search_payload.results
    assert search_payload.execution_trace
    assert pack_payload.pack
    assert pack_payload.injected_context == pack_payload.pack
