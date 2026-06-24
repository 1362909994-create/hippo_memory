from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.memory_scheduler import (
    MemoryScheduler,
    SystemStabilityController,
)


def test_load_aware_scheduler_converges_under_high_memory_pressure(db, tmp_path) -> None:
    writer = MemoryWriter(db)
    for index in range(80):
        memory_id = writer.write(
            project="sched",
            memory_type="technical_fact" if index % 3 else "source_chunk",
            content=(f"Scheduler pressure memory {index}. " * (20 if index % 3 == 0 else 1)),
            confidence=0.25 if index % 5 == 0 else 0.75,
            importance=0.1 if index % 5 == 0 else 0.55,
        ).memory_id
        if index % 5 == 0:
            _age_memory(db, memory_id, days=100, usage_count=0)

    scheduler = MemoryScheduler(db, state_path=tmp_path / "scheduler.json")
    reports = [
        scheduler.run_cycle(
            project="sched",
            system_load={"active_sessions": 4, "cpu_load": 0.9, "low_confidence_state": True},
        )
        for _ in range(3)
    ]

    assert reports[-1].optimization_loop["scheduler_interval_turns"] >= 5.0
    assert reports[-1].optimization_loop["compression_budget"] < 1.0
    assert all(len(report.lifecycle_actions) <= 160 for report in reports)
    assert db.list_memories(project="sched", limit=1000)


def test_scheduler_blocks_rapid_promotion_demotion_loops() -> None:
    controller = SystemStabilityController(convergence_window=3)

    first = controller.allow_transition("m1", "L1", "L2", turn=1)
    second = controller.allow_transition("m1", "L2", "L1", turn=2)
    third = controller.allow_transition("m1", "L2", "L1", turn=5)

    assert first["allowed"] is True
    assert second["allowed"] is False
    assert second["reason"] == "convergence_window"
    assert third["allowed"] is True


def test_conflicting_lifecycle_actions_do_not_delete_everything(db, tmp_path) -> None:
    writer = MemoryWriter(db)
    for index in range(20):
        memory_id = writer.write(
            project="sched",
            memory_type="technical_fact",
            content=f"Weak memory {index} should be reviewed, not hard deleted.",
            confidence=0.1,
            importance=0.1,
        ).memory_id
        _age_memory(db, memory_id, days=120, usage_count=0)

    report = MemoryScheduler(db, state_path=tmp_path / "scheduler.json").run_cycle(project="sched")

    assert any(action.action_type == "evict" for action in report.lifecycle_actions)
    assert len(db.list_memories(project="sched", limit=1000)) == 20
    assert all(
        action.metadata.get("hard_delete") is False
        for action in report.lifecycle_actions
        if action.action_type == "evict"
    )


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
