from __future__ import annotations

from hippocampus_memory.db import Database
from hippocampus_memory.memory_policy import auto_store_memories
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.turn_orchestrator import TurnOrchestrator
from hippocampus_memory.utils import estimate_tokens


def test_ab_matrix_architecture_refactor_improves_context(tmp_path):
    query = (
        "Refactor memory_scheduler policy semantic world model orchestrator boundaries "
        "architecture refactor constraints"
    )
    baseline = _pack(_db(tmp_path, "a_arch"), query)
    db = _db(tmp_path, "b_arch")
    auto_store_memories(
        db,
        "\n".join(
            [
                "Architectural decision: TurnOrchestrator remains the single CLI/MCP/API "
                "entry point and owns turn orchestration for compatibility.",
                "Constraint: MemoryScheduler owns lifecycle scheduling, while policy, "
                "semantic, and world-model layers stay decoupled and communicate through "
                "reports.",
            ]
        ),
        project="demo",
    )
    _write_noise(db)

    enhanced = _pack(db, query)

    assert baseline["tokens"] < enhanced["tokens"]
    assert enhanced["intent"] == "architecture_refactor"
    assert "Architecture Runtime Profile:" in enhanced["text"]
    assert "Token savings" not in enhanced["text"]


def test_ab_matrix_bug_fix_recalls_failure_not_architecture(tmp_path):
    query = "Fix YAML config loader crash when empty values produce parsing error"
    baseline = _pack(_db(tmp_path, "a_bug"), query)
    db = _db(tmp_path, "b_bug")
    _write(
        db,
        content="Error trace: YAML config loader failed when parsing empty values.",
        memory_type="failure",
        tags=["error", "yaml"],
    )
    _write_architecture_memory(db)

    enhanced = _pack(db, query)

    assert baseline["tokens"] < enhanced["tokens"]
    assert enhanced["intent"] == "bug_fix"
    assert "YAML config loader failed" in enhanced["text"]
    assert "Architecture Runtime Profile" not in enhanced["text"]


def test_ab_matrix_code_understanding_recalls_semantic_world_model(tmp_path):
    query = "Explain how the semantic graph and world model influence memory recall"
    baseline = _pack(_db(tmp_path, "a_understanding"), query)
    db = _db(tmp_path, "b_understanding")
    _write(
        db,
        content=(
            "Semantic graph connects memories to concepts; MemoryWorldModel tracks "
            "entities, concepts, decisions, events, and patterns for recall explanations."
        ),
        memory_type="technical_fact",
        tags=["semantic", "world_model", "graph"],
        entities=["MemoryWorldModel"],
    )
    _write_noise(db)

    enhanced = _pack(db, query)

    assert baseline["tokens"] < enhanced["tokens"]
    assert enhanced["intent"] == "code_understanding"
    assert "Semantic graph connects" in enhanced["text"]
    assert "Token savings" not in enhanced["text"]


def test_ab_matrix_multi_file_refactor_recalls_entrypoint_constraints(tmp_path):
    query = "Refactor CLI MCP API routing while preserving TurnOrchestrator run_turn compatibility"
    baseline = _pack(_db(tmp_path, "a_multi"), query)
    db = _db(tmp_path, "b_multi")
    _write(
        db,
        content=(
            "Constraint: CLI, MCP, and API entrypoints must preserve external payload "
            "shape while routing through TurnOrchestrator.run_turn."
        ),
        memory_type="constraint",
        tags=["cli", "mcp", "api", "orchestrator", "architecture"],
        entities=["CLI", "MCP", "API", "TurnOrchestrator"],
    )
    _write(
        db,
        content=(
            "Decision: keep routing glue thin and avoid direct calls from entrypoints "
            "into recall_policy, ranker, memory_policy, or scheduler."
        ),
        memory_type="decision",
        tags=["routing", "orchestrator"],
    )

    enhanced = _pack(db, query)

    assert baseline["tokens"] < enhanced["tokens"]
    assert enhanced["intent"] == "architecture_refactor"
    assert "external payload shape" in enhanced["text"]
    assert "routing glue thin" in enhanced["text"]


def test_ab_matrix_regression_avoidance_recalls_privacy_constraint(tmp_path):
    query = "Change search ranking but avoid leaking private sensitive memories"
    baseline = _pack(_db(tmp_path, "a_regression"), query)
    db = _db(tmp_path, "b_regression")
    _write(
        db,
        content=(
            "Constraint: search and ranking changes must preserve private and sensitive "
            "memory filters."
        ),
        memory_type="constraint",
        tags=["regression", "privacy", "search"],
    )

    enhanced = _pack(db, query)

    assert baseline["tokens"] < enhanced["tokens"]
    assert enhanced["intent"] == "bug_fix"
    assert "preserve private and sensitive memory filters" in enhanced["text"]


def test_ab_matrix_pollution_control_suppresses_irrelevant_memory(tmp_path):
    query = "Fix YAML parser bug"
    baseline = _pack(_db(tmp_path, "a_pollution"), query)
    db = _db(tmp_path, "b_pollution")
    _write_noise(db)
    _write_architecture_memory(db)

    enhanced = _pack(db, query)

    assert baseline["tokens"] == enhanced["tokens"]
    assert enhanced["intent"] == "bug_fix"
    assert "Token savings" not in enhanced["text"]
    assert "Architecture Runtime Profile" not in enhanced["text"]
    assert "MemoryScheduler" not in enhanced["text"]


def test_ab_matrix_cost_efficiency_keeps_compact_pack_small(tmp_path):
    query = "Refactor scheduler policy semantic world model boundaries"
    baseline = _pack(_db(tmp_path, "a_cost"), query)
    db = _db(tmp_path, "b_cost")
    for index in range(20):
        _write(
            db,
            content=(
                f"Architecture note {index}: scheduler policy semantic world_model "
                "boundary should remain concise and decoupled."
            ),
            memory_type="decision",
            tags=["architecture", "scheduler", "policy"],
            importance=0.7,
        )

    enhanced = _pack(db, query)

    assert baseline["tokens"] < enhanced["tokens"] <= 650
    assert enhanced["intent"] == "architecture_refactor"


def test_ab_matrix_persistence_reuses_auto_stored_memory_next_turn(tmp_path):
    query = "Refactor memory scheduler architecture boundaries"
    db = _db(tmp_path, "persist")
    baseline = _pack(db, query)
    store = auto_store_memories(
        db,
        (
            "Architectural decision: MemoryScheduler owns lifecycle scheduling and "
            "PolicyArbiter owns policy conflict resolution."
        ),
        project="demo",
    )

    enhanced = _pack(db, query)

    assert store["written"] >= 1
    assert baseline["tokens"] < enhanced["tokens"]
    assert enhanced["intent"] == "architecture_refactor"
    assert "MemoryScheduler owns lifecycle scheduling" in enhanced["text"]


def _db(tmp_path, name: str) -> Database:
    db = Database(tmp_path / f"{name}.db")
    db.initialize()
    return db


def _pack(db: Database, query: str) -> dict[str, object]:
    result = TurnOrchestrator(db).run_turn(
        query,
        context={
            "project": "demo",
            "operation": "memory_pack",
            "writeback": False,
            "compact": True,
            "include_code_map": False,
        },
        mode="preview",
    )
    relevance = result.turn_context.context_budget.get("task_relevance", {})
    return {
        "text": result.injected_context,
        "intent": relevance.get("detected_task_intent"),
        "tokens": estimate_tokens(result.injected_context),
    }


def _write(
    db: Database,
    *,
    content: str,
    memory_type: str,
    tags: list[str] | None = None,
    entities: list[str] | None = None,
    importance: float = 0.85,
    metadata: dict | None = None,
) -> None:
    MemoryWriter(db).write(
        project="demo",
        memory_type=memory_type,
        content=content,
        tags=tags or [],
        entities=entities or [],
        importance=importance,
        confidence=0.9,
        metadata=metadata,
    )


def _write_noise(db: Database) -> None:
    _write(
        db,
        content="Token savings status bar history and UI display issue.",
        memory_type="decision",
        tags=["ui-history", "token-ui"],
        importance=1.0,
    )


def _write_architecture_memory(db: Database) -> None:
    _write(
        db,
        content="MemoryScheduler policy semantic world_model architecture boundary.",
        memory_type="decision",
        tags=["architecture", "scheduler", "policy"],
        entities=["MemoryScheduler"],
        importance=1.0,
        metadata={
            "architecture_runtime_profile": {
                "layers": ["scheduler", "policy"],
                "interfaces": [],
                "boundary_signals": ["ownership"],
                "canonical_entities": ["MemoryScheduler"],
            }
        },
    )

