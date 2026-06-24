from __future__ import annotations

import json
import math
from pathlib import Path

from hippocampus_memory.api import create_app
from hippocampus_memory.config import Settings
from hippocampus_memory.db import Database
from hippocampus_memory.deploy import codex_doctor, deploy_codex
from hippocampus_memory.mcp_server import HippoMcpServer
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.turn_orchestrator import TurnOrchestrator
from hippocampus_memory.schemas import AutoContextRequest, MemoryPackRequest, MemorySearchRequest
from hippocampus_memory.utils import estimate_tokens


def test_01_codex_project_deploy_creates_project_local_mcp_entry(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")

    result = deploy_codex(root, project="demo", index_project=False)
    config_path = root / ".hippo" / "codex-mcp-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert result["project"] == "demo"
    assert (root / ".hippo" / "hippo.db").exists()
    assert (root / ".hippo.toml").exists()
    assert config["mcpServers"]["hippo_memory"]["command"] == "hippo"
    assert config["mcpServers"]["hippo_memory"]["args"] == [
        "mcp-project",
        "--root",
        str(root.resolve()),
    ]
    assert 'session_key="codex"' in (root / "AGENTS.md").read_text(encoding="utf-8")


def test_02_scheduler_state_write_failure_is_non_fatal_and_traced(tmp_path: Path) -> None:
    db = _db(tmp_path, "sandbox")
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content="Decision: Codex scheduler persistence failure must not break packs.",
        tags=["scheduler", "codex"],
    )
    bad_state_path = tmp_path / "state-as-directory"
    bad_state_path.mkdir()

    server = HippoMcpServer(
        db,
        safe_tool_names=False,
        default_project="demo",
        scheduler_state_path=bad_state_path,
    )
    result = server.call_tool("memory.pack", {"query": "Codex scheduler persistence"})
    trace = result["execution_trace"]

    assert "Memory Pack:" in result["text"]
    assert any(event["decision"] == "state_save_failed" for event in trace)
    scheduler = result["context_budget"]["memory_scheduler_report"]
    assert scheduler["persistence_report"]["event"] == "scheduler_state_save_failed"


def test_03_task_aware_ab_retrieval_beats_empty_baseline_without_ui_noise(tmp_path: Path) -> None:
    query = "Refactor scheduler policy semantic world model boundaries for Codex runtime"
    baseline = _pack(_db(tmp_path, "baseline"), query)
    db = _db(tmp_path, "enhanced")
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content=(
            "Architectural decision: MemoryScheduler owns lifecycle scheduling, "
            "PolicyArbiter owns policy arbitration, and semantic/world-model layers "
            "communicate through reports."
        ),
        tags=["scheduler", "policy", "semantic", "world_model", "architecture"],
        entities=["MemoryScheduler", "PolicyArbiter", "MemoryWorldModel"],
        importance=0.9,
    )
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content="Token savings status bar UI history should not steer architecture refactors.",
        tags=["ui-history", "token-ui"],
        importance=1.0,
    )

    enhanced = _pack(db, query)

    assert baseline["tokens"] < enhanced["tokens"]
    assert enhanced["intent"] == "architecture_refactor"
    assert "MemoryScheduler owns lifecycle scheduling" in enhanced["text"]
    assert "Token savings" not in enhanced["text"]


def test_04_project_and_session_isolation_are_independent(tmp_path: Path) -> None:
    db = _db(tmp_path, "isolation")
    alpha_id = MemoryWriter(db).write(
        project="alpha",
        memory_type="decision",
        content="Decision: alpha project uses scheduler boundary alpha-only-token.",
        tags=["scheduler", "alpha-only-token"],
    ).memory_id
    MemoryWriter(db).write(
        project="beta",
        memory_type="decision",
        content="Decision: beta project uses storage boundary beta-only-token.",
        tags=["storage", "beta-only-token"],
    )
    orchestrator = TurnOrchestrator(db, scheduler_state_path=tmp_path / "scheduler.json")

    alpha = orchestrator.run_turn(
        "alpha-only-token scheduler boundary",
        context={"project": "alpha", "operation": "memory_pack", "writeback": False},
    )
    beta = orchestrator.run_turn(
        "alpha-only-token scheduler boundary",
        context={"project": "beta", "operation": "memory_pack", "writeback": False},
    )
    first = orchestrator.run_turn(
        "alpha-only-token scheduler boundary",
        context={
            "project": "alpha",
            "operation": "context_callback",
            "session_key": "s1",
            "writeback": False,
        },
    )
    second_same_session = orchestrator.run_turn(
        "alpha-only-token scheduler boundary",
        context={
            "project": "alpha",
            "operation": "context_callback",
            "session_key": "s1",
            "writeback": False,
        },
    )
    first_other_session = orchestrator.run_turn(
        "alpha-only-token scheduler boundary",
        context={
            "project": "alpha",
            "operation": "context_callback",
            "session_key": "s2",
            "writeback": False,
        },
    )

    assert "alpha-only-token" in alpha.injected_context
    assert alpha_id not in beta.recall_payload.get("included_memory_ids", [])
    assert "alpha project uses scheduler boundary" not in beta.injected_context
    assert alpha_id in first.recall_payload["included_memory_ids"]
    assert alpha_id in second_same_session.recall_payload["excluded_memory_ids"]
    assert alpha_id not in second_same_session.recall_payload["included_memory_ids"]
    assert alpha_id in first_other_session.recall_payload["included_memory_ids"]


def test_05_privacy_filters_and_noise_suppression_hold_for_codex_pack(tmp_path: Path) -> None:
    db = _db(tmp_path, "privacy")
    writer = MemoryWriter(db)
    writer.write(
        project="demo",
        memory_type="decision",
        content="Decision: public scheduler policy boundary is safe to recall.",
        tags=["scheduler", "policy", "architecture"],
        visibility="project",
    )
    writer.write(
        project="demo",
        memory_type="technical_fact",
        content="PRIVATE_SECRET_DO_NOT_RECALL",
        visibility="private",
        importance=1.0,
    )
    writer.write(
        project="demo",
        memory_type="technical_fact",
        content="SENSITIVE_SECRET_DO_NOT_RECALL",
        visibility="sensitive",
        importance=1.0,
    )
    writer.write(
        project="demo",
        memory_type="decision",
        content="Token savings status bar UI history should be suppressed for architecture work.",
        tags=["ui-history", "token-ui"],
        importance=1.0,
    )

    result = TurnOrchestrator(db, scheduler_state_path=tmp_path / "scheduler.json").run_turn(
        "Refactor scheduler policy architecture boundary",
        context={"project": "demo", "operation": "memory_pack", "writeback": False},
    )

    assert "public scheduler policy boundary" in result.injected_context
    assert "PRIVATE_SECRET_DO_NOT_RECALL" not in result.injected_context
    assert "SENSITIVE_SECRET_DO_NOT_RECALL" not in result.injected_context
    assert "Token savings" not in result.injected_context


def test_06_long_run_policy_scheduler_state_remains_finite(tmp_path: Path) -> None:
    db = _db(tmp_path, "longrun")
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content="Decision: Codex long-run scheduler policy semantic reports stay stable.",
        tags=["scheduler", "policy", "semantic"],
    )
    orchestrator = TurnOrchestrator(db, scheduler_state_path=tmp_path / "scheduler.json")

    last = None
    for index in range(20):
        last = orchestrator.run_turn(
            f"Codex long-run scheduler policy turn {index}",
            context={"project": "demo", "writeback": False, "min_rank_confidence": 0.0},
        )

    assert last is not None
    budget = last.turn_context.context_budget
    scheduler = budget["memory_scheduler_report"]
    policy_history = budget["multi_policy_decision_history"]
    assert scheduler["persistence_report"]["event"] == "scheduler_state_saved"
    assert scheduler["state_version"]
    assert policy_history
    assert _all_numbers_finite(budget)


def test_07_install_helper_and_codex_doctor_are_codex_only(tmp_path: Path) -> None:
    script = Path("install-codex-hippo.ps1")
    root = tmp_path / "doctor"
    root.mkdir()
    deploy_codex(root, project="doctor", index_project=False)

    report = codex_doctor(root)
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert "codex-deploy" in script_text
    assert "reasonix" not in script_text.casefold()
    assert report["diagnostic"] == "hippo_codex"
    assert report["ready"] is True
    assert report["recommendations"] == []


def test_08_api_and_mcp_clients_keep_turn_result_compatibility(tmp_path: Path) -> None:
    settings = Settings(db_path=tmp_path / "api-mcp.db")
    db = Database(settings=settings)
    db.initialize()
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content="Decision: API and MCP compatibility returns turn metadata.",
        tags=["api", "mcp", "compatibility"],
    )
    server = HippoMcpServer(db, safe_tool_names=True, default_project="demo")
    app = create_app(settings)
    routes = {getattr(route, "path", None): route.endpoint for route in app.routes}

    initialized = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    mcp_pack = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "memory_pack",
                "arguments": {"query": "API MCP compatibility"},
            },
        }
    )
    api_search = routes["/memory/search"](
        MemorySearchRequest(query="API MCP compatibility", project="demo")
    )
    api_pack = routes["/memory/pack"](
        MemoryPackRequest(query="API MCP compatibility", project="demo")
    )
    api_auto = routes["/context/auto"](
        AutoContextRequest(intent="API MCP compatibility", project="demo", session_key="codex")
    )

    assert initialized["result"]["serverInfo"]["name"] == "hippocampus-memory"
    assert "memory_pack" in [tool["name"] for tool in tools["result"]["tools"]]
    mcp_payload = mcp_pack["result"]["structuredContent"]
    assert mcp_payload["execution_trace"]
    assert mcp_payload["retrieved_memories"]
    assert mcp_payload["selected_memories"]
    assert api_search.execution_trace
    assert api_pack.execution_trace
    assert api_pack.injected_context == api_pack.pack
    assert api_auto["execution_trace"]
    assert api_auto["injected_context"] == api_auto["text"]


def _db(tmp_path: Path, name: str) -> Database:
    db = Database(tmp_path / f"{name}.db")
    db.initialize()
    return db


def _pack(db: Database, query: str) -> dict[str, object]:
    result = TurnOrchestrator(db).run_turn(
        query,
        context={"project": "demo", "operation": "memory_pack", "writeback": False},
    )
    relevance = result.turn_context.context_budget.get("task_relevance", {})
    return {
        "text": result.injected_context,
        "tokens": estimate_tokens(result.injected_context),
        "intent": relevance.get("detected_task_intent"),
    }


def _all_numbers_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int | float):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_all_numbers_finite(item) for item in value)
    return True
