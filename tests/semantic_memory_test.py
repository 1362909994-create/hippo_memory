from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.memory_scheduler import (
    MemoryScheduler,
    SemanticCompressionEngine,
    SemanticMemoryModel,
)


def test_semantic_type_classification_covers_decision_error_insight_pattern_fact(db) -> None:
    writer = MemoryWriter(db)
    memories = [
        db.get_memory(
            writer.write(
                project="semantic", memory_type="decision", content="Decision: use SQLite."
            ).memory_id
        ),
        db.get_memory(
            writer.write(
                project="semantic", memory_type="failure", content="Error: recall failed."
            ).memory_id
        ),
        db.get_memory(
            writer.write(
                project="semantic",
                memory_type="technical_fact",
                content="Insight: repeated decisions reveal root cause patterns.",
            ).memory_id
        ),
        db.get_memory(
            writer.write(
                project="semantic",
                memory_type="constraint",
                content="Always preserve private memory filters.",
            ).memory_id
        ),
        db.get_memory(
            writer.write(
                project="semantic",
                memory_type="technical_fact",
                content="SQLite stores local memories.",
            ).memory_id
        ),
    ]

    types = [SemanticMemoryModel().profile(memory).semantic_type for memory in memories]

    assert types == ["decision", "error", "insight", "pattern", "fact"]


def test_semantic_compression_preserves_meaning_and_merges_duplicates(db) -> None:
    writer = MemoryWriter(db)
    first = db.get_memory(
        writer.write(
            project="semantic",
            memory_type="decision",
            content="Decision: use TurnOrchestrator as the single entrypoint.",
            confidence=0.9,
            importance=0.8,
        ).memory_id
    )
    second = db.get_memory(
        writer.write(
            project="semantic",
            memory_type="technical_fact",
            content="Use TurnOrchestrator as the single entrypoint.",
            confidence=0.8,
            importance=0.7,
        ).memory_id
    )

    result = SemanticCompressionEngine().compress([first, second], model=SemanticMemoryModel())

    assert result["merged_meanings"]
    assert result["merged_meanings"][0]["meaning_preserved"] is True
    assert "TurnOrchestrator" in result["merged_meanings"][0]["semantic_summary"]
    assert result["semantic_redundancy"] > 0.0


def test_outdated_high_weight_memory_is_detected_in_semantic_report(db, tmp_path) -> None:
    memory_id = (
        MemoryWriter(db)
        .write(
            project="semantic",
            memory_type="decision",
            content="Decision: use the old direct recall path.",
            confidence=0.95,
            importance=0.95,
        )
        .memory_id
    )
    _age_memory(db, memory_id, days=180, usage_count=8)

    report = MemoryScheduler(db, state_path=tmp_path / "scheduler.json").run_cycle(
        project="semantic"
    )

    assert (
        memory_id in report.semantic_report["meaning_consistency"]["outdated_high_weight_memories"]
    )
    assert (
        report.semantic_report["global_semantic_objective"]["minimize"][
            "outdated_reasoning_patterns"
        ]
        > 0.0
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
