from __future__ import annotations

import json
import re

from hippocampus_memory.code_graph import CodeGraphBuilder
from hippocampus_memory.consolidator import Consolidator
from hippocampus_memory.evaluator import evaluate_retrieval
from hippocampus_memory.lsp_diagnostics import CodeDiagnostic
from hippocampus_memory.mcp_server import HippoMcpServer
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.project_indexer import ProjectIndexer
from hippocampus_memory.report import render_memory_browser
from hippocampus_memory.session_recorder import record_run_session


def test_sensitive_content_defaults_to_sensitive_visibility(db):
    result = MemoryWriter(db).write(
        project="demo",
        memory_type="technical_fact",
        content="api_key=secret-value",
    )
    memory = db.get_memory(result.memory_id)
    assert memory is not None
    assert memory.visibility == "sensitive"


def test_consolidate_archives_expired_memory(db):
    result = MemoryWriter(db).write(
        project="demo",
        memory_type="task_state",
        content="Temporary state",
        ttl_days=1,
    )
    with db.connect() as conn:
        conn.execute(
            "UPDATE memories SET expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", result.memory_id),
        )

    summary = Consolidator(db).consolidate("demo")

    assert summary["archived_count"] >= 1
    assert db.get_memory(result.memory_id).status == "archived"


def test_consolidate_archives_missing_file_source_chunks(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    source = root / "app.py"
    source.write_text("def removed():\n    return True\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")
    with db.connect() as conn:
        memory_id = conn.execute(
            "SELECT memory_id FROM chunks WHERE project = 'demo'"
        ).fetchone()["memory_id"]

    source.unlink()
    ProjectIndexer(db).index_project(root, "demo")
    summary = Consolidator(db).consolidate("demo")

    assert summary["missing_source_chunk_count"] == 1
    assert db.get_memory(memory_id).status == "archived"


def test_code_graph_infers_cross_file_calls(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("from b import helper\n\ndef main():\n    return helper()\n")
    (root / "b.py").write_text("def helper():\n    return 1\n")
    ProjectIndexer(db).index_project(root, "demo")

    graph = CodeGraphBuilder(db).build("demo")

    assert "a.py --calls helper--> b.py" in graph


def test_mcp_server_can_write_and_search_memory(db):
    server = HippoMcpServer(db)
    tools = server.handle({"jsonrpc": "2.0", "id": 0, "method": "tools/list"})
    assert tools["result"]["tools"][0]["inputSchema"]

    write_response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "memory.write",
                "arguments": {
                    "project": "demo",
                    "memory_type": "task_state",
                    "content": "MCP can write memories.",
                },
            },
        }
    )
    assert write_response["result"]["structuredContent"]["created"] is True

    search_response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "memory.search",
                "arguments": {"query": "MCP write", "project": "demo"},
            },
        }
    )
    assert search_response["result"]["structuredContent"]["results"]
    assert search_response["result"]["content"][0]["type"] == "text"
    assert search_response["result"]["content"][0]["text"].startswith("Memory Search Results:")
    assert '"memory_id"' not in search_response["result"]["content"][0]["text"]


def test_mcp_server_can_expose_deepseek_safe_tool_names(db):
    server = HippoMcpServer(db, safe_tool_names=True)
    tools = server.handle({"jsonrpc": "2.0", "id": 0, "method": "tools/list"})
    names = [tool["name"] for tool in tools["result"]["tools"]]

    assert "memory_write" in names
    assert "context_callback" in names
    assert "memory.write" not in names
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in names)

    write_response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "memory_write",
                "arguments": {
                    "project": "demo",
                    "memory_type": "task_state",
                    "content": "Safe MCP names can write memories.",
                },
            },
        }
    )

    assert write_response["result"]["structuredContent"]["created"] is True


def test_mcp_search_text_dedupes_repeated_summaries(db):
    writer = MemoryWriter(db)
    writer.write(
        project="demo",
        memory_type="source_chunk",
        content="architecture module first chunk with callback search context",
        summary="server.py:L1-120: architecture module callback search context",
    )
    writer.write(
        project="demo",
        memory_type="source_chunk",
        content="architecture module second chunk with callback search context",
        summary="server.py:L1-120: architecture module callback search context",
    )
    server = HippoMcpServer(db)

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "memory.search",
                "arguments": {
                    "query": "architecture callback search",
                    "project": "demo",
                    "top_k": 5,
                    "dedupe_results": False,
                },
            },
        }
    )

    text = response["result"]["content"][0]["text"]
    assert text.count("server.py:L1-120") == 1
    assert "duplicate/extra" in text


def test_mcp_server_can_generate_callback_pack(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="task_state",
        content="MCP callback should remember injected memories.",
        importance=0.9,
    )
    server = HippoMcpServer(db)

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "context.callback",
                "arguments": {
                    "project": "demo",
                    "intent": "callback memory",
                    "session_key": "mcp-test",
                },
            },
        }
    )

    payload = response["result"]["structuredContent"]
    assert payload["included_memory_ids"]
    assert "MCP callback should remember" in response["result"]["content"][0]["text"]


def test_mcp_server_can_query_code_intelligence(db, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text(
        "from b import helper\n\ndef main():\n    return helper()\n",
        encoding="utf-8",
    )
    (root / "b.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    ProjectIndexer(db).index_project(root, "demo")
    server = HippoMcpServer(db)

    symbols = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "code.symbols",
                "arguments": {"project": "demo", "query": "helper"},
            },
        }
    )
    references = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "code.references",
                "arguments": {"project": "demo", "symbol": "helper"},
            },
        }
    )
    impact = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "code.intelligence",
                "arguments": {"project": "demo", "intent": "change helper behavior"},
            },
        }
    )

    assert symbols["result"]["structuredContent"]["symbols"]
    assert references["result"]["structuredContent"]["references"]
    assert "helper [function] b.py:1" in impact["result"]["content"][0]["text"]


def test_mcp_server_can_query_code_diagnostics(db):
    db.replace_code_diagnostics(
        project="demo",
        source="basedpyright",
        diagnostics=[
            CodeDiagnostic(
                relative_path="app.py",
                severity="error",
                message="Undefined name",
                line=2,
                column=4,
                rule="reportUndefinedVariable",
                source="basedpyright",
            )
        ],
    )
    server = HippoMcpServer(db)

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "code.diagnostics",
                "arguments": {"project": "demo"},
            },
        }
    )

    diagnostics = response["result"]["structuredContent"]["diagnostics"]
    assert diagnostics[0]["relative_path"] == "app.py"


def test_record_run_session_stores_event_and_optional_memory(db):
    result = record_run_session(
        db,
        project="demo",
        intent="test run",
        command=["python", "--version"],
        returncode=0,
        context_file=None,
        stdout="ok",
        stderr="",
        write_memory=True,
    )

    assert result["event_id"]
    assert result["memory_id"]
    with db.connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'session.run'"
        ).fetchone()[0]
    assert count == 1


def test_evaluate_retrieval_jsonl(db, tmp_path):
    MemoryWriter(db).write(
        project="demo",
        memory_type="task_state",
        content="The display task is to light a TFT screen.",
    )
    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "query": "display task",
                "project": "demo",
                "expected_contains": ["TFT screen"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_retrieval(db, benchmark)

    assert result["hit_rate"] == 1.0


def test_evaluate_pack_jsonl_supports_forbidden_and_budget(db, tmp_path):
    writer = MemoryWriter(db)
    writer.write(
        project="demo",
        memory_type="task_state",
        content="The display task is to light a TFT screen.",
    )
    writer.write(
        project="demo",
        memory_type="technical_fact",
        content="password=secret",
    )
    benchmark = tmp_path / "pack-bench.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "mode": "pack",
                "query": "display task",
                "project": "demo",
                "expected_contains": ["TFT screen"],
                "forbidden_contains": ["password=secret"],
                "max_tokens": 300,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_retrieval(db, benchmark)

    assert result["hit_rate"] == 1.0


def test_browser_report_excludes_sensitive_memory(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="technical_fact",
        content="Public architecture note.",
    )
    MemoryWriter(db).write(
        project="demo",
        memory_type="technical_fact",
        content="password=secret",
    )

    html = render_memory_browser(db, "demo")

    assert "Public architecture note." in html
    assert "password=secret" not in html


def test_browser_report_shows_candidates_and_conflicts(db):
    db.insert_candidate(
        project="demo",
        content="Candidate task state",
        memory_type="task_state",
        confidence=0.7,
        importance=0.6,
        source="test",
    )
    writer = MemoryWriter(db)
    writer.write(
        project="demo",
        memory_type="technical_fact",
        content="TFT interface is SPI.",
        entities=["TFT"],
    )
    writer.write(
        project="demo",
        memory_type="technical_fact",
        content="TFT interface is RGB.",
        entities=["TFT"],
    )
    from hippocampus_memory.conflict_detector import ConflictDetector

    ConflictDetector(db).detect_for_project("demo")

    html = render_memory_browser(db, "demo")

    assert "Candidate task state" in html
    assert "Open Conflicts" in html


def test_conflict_resolve_updates_status(db):
    conflict_id = "cfl_test"
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO conflicts
                (id, project, entity, attribute, memory_ids, description,
                 resolution, status, created_at, updated_at)
            VALUES (?, 'demo', 'TFT', 'content', '[]', 'conflict', NULL, 'open', ?, ?)
            """,
            ("cfl_test", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        )

    assert db.update_conflict(conflict_id, resolution="Use SPI")
    assert db.list_conflicts("demo") == []
