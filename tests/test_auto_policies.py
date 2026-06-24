from __future__ import annotations

from typer.testing import CliRunner

from hippocampus_memory.cli import app
from hippocampus_memory.mcp_server import HippoMcpServer
from hippocampus_memory.memory_policy import auto_store_memories, plan_memory_admission
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.turn_orchestrator import TurnOrchestrator
from hippocampus_memory.recall_policy import build_auto_context, decide_recall


def test_memory_admission_writes_high_confidence_decision(db):
    result = auto_store_memories(
        db,
        (
            "Decision: we will use SQLite FTS as the default retrieval backend.\n"
            "Thanks for checking."
        ),
        project="demo",
    )

    assert result["written"] == 1
    assert result["queued"] == 0
    memories = db.list_memories(project="demo")
    assert memories[0].memory_type == "decision"
    assert "SQLite FTS" in memories[0].content


def test_memory_admission_enriches_architecture_runtime_memories():
    decisions = plan_memory_admission(
        "\n".join(
            [
                "Architectural decision: TurnOrchestrator should remain the single "
                "CLI/MCP/API entry point, while MemoryScheduler owns lifecycle "
                "scheduling only.",
                "Constraint: scheduler, policy, semantic, and world-model layers must "
                "stay decoupled and communicate through reports.",
            ]
        ),
        project="hippo",
    )

    assert {decision.memory_type for decision in decisions} >= {"decision", "constraint"}
    assert all("architecture" in decision.tags for decision in decisions)
    assert any("orchestrator" in decision.tags for decision in decisions)
    assert any("scheduler" in decision.tags for decision in decisions)
    assert any("policy" in decision.tags for decision in decisions)
    assert any("semantic" in decision.tags for decision in decisions)
    assert any("world_model" in decision.tags for decision in decisions)
    assert any("TurnOrchestrator" in decision.entities for decision in decisions)
    assert any("MemoryScheduler" in decision.entities for decision in decisions)
    assert all("architecture runtime" in decision.reason for decision in decisions)


def test_auto_store_architecture_memory_improves_next_architecture_recall(db):
    store = auto_store_memories(
        db,
        "\n".join(
            [
                "Architectural decision: TurnOrchestrator should remain the single "
                "CLI/MCP/API entry point, while MemoryScheduler owns lifecycle "
                "scheduling only.",
                "Constraint: scheduler, policy, semantic, and world-model layers must "
                "stay decoupled and communicate through reports.",
            ]
        ),
        project="hippo",
    )

    result = TurnOrchestrator(db).run_turn(
        "Refactor memory_scheduler policy semantic world model orchestrator boundaries.",
        context={"project": "hippo", "operation": "memory_search", "writeback": False},
        mode="preview",
    )

    assert store["written"] == 2
    assert "TurnOrchestrator" in result.injected_context
    assert "MemoryScheduler" in result.injected_context
    relevance = result.turn_context.context_budget["task_relevance"]
    assert relevance["detected_task_intent"] == "architecture_refactor"
    assert relevance["boosted_memories"]


def test_auto_store_architecture_memory_persists_runtime_profile(db):
    store = auto_store_memories(
        db,
        "\n".join(
            [
                "Architectural decision: TurnOrchestrator remains the CLI/MCP/API "
                "entry point and MemoryScheduler owns lifecycle scheduling.",
                "Constraint: policy, semantic, and world-model layers communicate "
                "through reports instead of direct ownership.",
            ]
        ),
        project="hippo",
    )

    memory_ids = [item["memory_id"] for item in store["items"] if item["memory_id"]]
    profiles = [
        db.get_memory(memory_id).metadata["architecture_runtime_profile"]
        for memory_id in memory_ids
    ]

    assert store["written"] == 2
    assert any("orchestrator" in profile["layers"] for profile in profiles)
    assert any("scheduler" in profile["layers"] for profile in profiles)
    assert any("policy" in profile["layers"] for profile in profiles)
    assert any("semantic" in profile["layers"] for profile in profiles)
    assert any("world_model" in profile["layers"] for profile in profiles)
    assert any("CLI" in profile["interfaces"] for profile in profiles)
    assert any("MCP" in profile["interfaces"] for profile in profiles)
    assert any("API" in profile["interfaces"] for profile in profiles)
    assert any("entry_point" in profile["boundary_signals"] for profile in profiles)
    assert any("ownership" in profile["boundary_signals"] for profile in profiles)


def test_memory_admission_queues_sensitive_memory_by_default(db):
    result = auto_store_memories(
        db,
        "Technical fact: api_key=secret-value is used by the test server.",
        project="demo",
    )

    assert result["written"] == 0
    assert result["queued"] == 1
    assert db.list_memories(project="demo", include_sensitive=True) == []
    candidates = db.list_candidates(project="demo")
    assert candidates[0]["memory_type"] == "technical_fact"


def test_memory_admission_skips_low_value_lines():
    decisions = plan_memory_admission("ok\nthanks\nWall time: 0.8 seconds")

    assert decisions == []


def test_auto_store_skips_near_duplicate_existing_memory(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="task_state",
        content="Current task is to finish automatic memory scheduling.",
        importance=0.9,
    )

    result = auto_store_memories(
        db,
        "Current task is to finish automatic memory scheduling with tests.",
        project="demo",
    )

    assert result["written"] == 0
    assert result["queued"] == 0
    assert result["duplicates"] == 1
    assert result["items"][0]["outcome"] == "near_duplicate"
    assert len(db.list_memories(project="demo")) == 1


def test_auto_context_skips_small_talk(db):
    context = build_auto_context(db, intent="thanks", project="demo")

    assert context["decision"]["action"] == "none"
    assert "No external memory recall recommended" in context["text"]


def test_auto_context_recalls_continuation_with_session_dedupe(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="task_state",
        content="Current task is to finish auto memory scheduling.",
        importance=0.9,
    )

    first = build_auto_context(db, intent="continue", project="demo", session_key="s1")
    second = build_auto_context(db, intent="continue", project="demo", session_key="s1")

    assert first["decision"]["action"] == "callback_pack"
    assert first["included_memory_ids"]
    assert "auto memory scheduling" in first["text"]
    assert second["excluded_memory_ids"] == first["included_memory_ids"]


def test_auto_context_uses_bundle_for_coding_change(db):
    decision = decide_recall("fix search ranking bug", project="demo")

    assert decision.action == "context_bundle"
    assert decision.strategy == "lean"


def test_auto_context_uses_impact_for_explicit_risk_question(db):
    decision = decide_recall("what is the impact of modifying retrieval ranking?", project="demo")

    assert decision.action == "impact_pack"


def test_cli_auto_store_and_auto_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_DB_PATH", str(tmp_path / "auto.db"))
    runner = CliRunner()

    stored = runner.invoke(
        app,
        [
            "auto-store",
            "--project",
            "demo",
            "--text",
            "Current task is to wire automatic memory recall.",
        ],
    )
    context = runner.invoke(
        app,
        ["auto-context", "continue", "--project", "demo", "--metadata"],
    )

    assert stored.exit_code == 0
    assert "'written': 1" in stored.output
    assert context.exit_code == 0
    assert "callback_pack" in context.output
    assert "automatic memory recall" in context.output


def test_mcp_auto_store_and_auto_context(db):
    server = HippoMcpServer(db, safe_tool_names=True, default_project="demo")
    initialized = server.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize"})
    assert "token_savings_text" not in initialized["result"]["instructions"]
    assert "Codex" not in initialized["result"]["instructions"]

    tools = server.handle({"jsonrpc": "2.0", "id": 0, "method": "tools/list"})
    names = [tool["name"] for tool in tools["result"]["tools"]]
    assert "memory_auto_store" in names
    assert "context_auto" in names

    stored = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "memory_auto_store",
                "arguments": {
                    "text": "Next step is to test the automatic context scheduler.",
                },
            },
        }
    )
    assert stored["result"]["structuredContent"]["written"] == 1

    recalled = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "context_auto",
                "arguments": {
                    "intent": "continue",
                    "session_key": "mcp-auto",
                },
            },
        }
    )
    assert recalled["result"]["structuredContent"]["decision"]["action"] == "callback_pack"
    assert "token_savings_text" not in recalled["result"]["structuredContent"]
    content_text = recalled["result"]["content"][0]["text"]
    assert "Show this token savings line to the user:" not in content_text
    assert "automatic context scheduler" in content_text
