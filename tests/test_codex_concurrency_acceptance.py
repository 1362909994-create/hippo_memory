from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hippocampus_memory.db import Database
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.turn_orchestrator import TurnOrchestrator


def test_codex_orchestrator_concurrent_turns_share_database_safely(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.db"
    db = Database(db_path)
    db.initialize()

    def worker(index: int) -> dict[str, object]:
        marker = f"concurrent-marker-{index}"
        local_db = Database(db_path)
        MemoryWriter(local_db).write(
            project="demo",
            memory_type="decision",
            content=(
                f"Decision: Codex concurrent worker {index} owns {marker} "
                "scheduler policy boundary."
            ),
            tags=["codex", "concurrency", marker],
            importance=0.8,
        )
        result = TurnOrchestrator(
            local_db,
            scheduler_state_path=tmp_path / f"scheduler-{index}.json",
        ).run_turn(
            f"Recall Codex concurrency scheduler policy {marker}",
            context={
                "project": "demo",
                "operation": "memory_pack",
                "writeback": False,
                "top_k": 6,
                "min_rank_confidence": 0.0,
            },
        )
        return {
            "marker": marker,
            "text": result.injected_context,
            "trace_count": len(result.trace),
            "scheduler": result.turn_context.context_budget.get(
                "memory_scheduler_report", {}
            ),
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = list(executor.map(worker, range(8)))

    with db.connect() as conn:
        memory_count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE project = 'demo'"
        ).fetchone()[0]

    assert memory_count == 8
    assert all(item["trace_count"] for item in outputs)
    assert all("lifecycle_actions" in item["scheduler"] for item in outputs)
    assert all(item["scheduler"].get("persistence_report") for item in outputs)
    assert all(item["marker"] in item["text"] for item in outputs)
