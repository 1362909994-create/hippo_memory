from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hippocampus_memory.consolidator import Consolidator
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.memory_scheduler import MemoryScheduler
from hippocampus_memory.orchestrator.turn_orchestrator import TurnOrchestrator


def test_full_memory_lifecycle_creation_recall_decay_promotion_demotion_eviction_compression(
    db, tmp_path
) -> None:
    writer = MemoryWriter(db)
    recalled = writer.write(
        project="life",
        memory_type="decision",
        content="Decision: scheduler recall should increase usage weight.",
        confidence=0.95,
        importance=0.9,
    ).memory_id
    stale = writer.write(
        project="life",
        memory_type="technical_fact",
        content="Old weak fact that should decay and be considered for eviction.",
        confidence=0.15,
        importance=0.1,
    ).memory_id
    task_state = writer.write(
        project="life",
        memory_type="task_state",
        content="Temporary task state should demote after aging.",
        confidence=0.6,
        importance=0.4,
    ).memory_id
    large = writer.write(
        project="life",
        memory_type="source_chunk",
        content="Long lifecycle detail. " * 220,
        confidence=0.8,
        importance=0.7,
    ).memory_id
    _age_memory(db, stale, days=120, usage_count=0)
    _age_memory(db, task_state, days=30, usage_count=0)
    _age_memory(db, large, days=20, usage_count=1)

    orchestrator = TurnOrchestrator(db)
    for _ in range(3):
        orchestrator.run_turn(
            "continue scheduler recall usage weight",
            context={"project": "life", "writeback": False},
            mode="preview",
        )
    _age_memory(db, task_state, days=30, usage_count=0)

    assert db.get_memory(recalled).usage_count >= 3

    report = MemoryScheduler(db, state_path=tmp_path / "scheduler.json").run_cycle(
        project="life", selected_memory_ids=[recalled]
    )
    action_ids = {}
    for action in report.lifecycle_actions:
        action_ids.setdefault(action.action_type, set()).add(action.memory_id)

    assert stale in action_ids["decay"]
    assert stale in action_ids["evict"]
    assert recalled in action_ids["promote"]
    assert task_state in action_ids["demote"]
    assert large in action_ids["compress"]
    assert report.hierarchy_report["assignments"][recalled]["tier"] in {"L0", "L2"}


def test_conflicting_memory_resolution_is_reported_without_silent_overwrite(db) -> None:
    writer = MemoryWriter(db)
    positive = writer.write(
        project="life",
        memory_type="technical_fact",
        content="Cache policy accept durable scheduler cache.",
        entities=["scheduler cache"],
        confidence=0.9,
        importance=0.8,
    ).memory_id
    negative = writer.write(
        project="life",
        memory_type="technical_fact",
        content="Cache policy reject durable scheduler cache.",
        entities=["scheduler cache"],
        confidence=0.85,
        importance=0.75,
    ).memory_id

    result = Consolidator(db).consolidate(project="life")
    memories = {memory.id: memory for memory in db.list_memories(project="life")}

    assert result["conflict_count"] >= 1
    assert positive in memories
    assert negative in memories
    assert memories[positive].status == "active"
    assert memories[negative].status == "active"


def _age_memory(db, memory_id: str, *, days: int, usage_count: int) -> None:
    created_at = (datetime.now(UTC) - timedelta(days=days)).replace(microsecond=0).isoformat()
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE memories
            SET created_at = ?, updated_at = ?, last_used_at = NULL, usage_count = ?
            WHERE id = ?
            """,
            (created_at, created_at, usage_count, memory_id),
        )
