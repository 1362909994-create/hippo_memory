from __future__ import annotations

from pathlib import Path
from time import perf_counter

from hippocampus_memory.db import Database
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.turn_orchestrator import TurnOrchestrator
from hippocampus_memory.project_indexer import ProjectIndexer
from hippocampus_memory.utils import estimate_tokens


def test_large_project_index_and_pack_stay_bounded(tmp_path: Path) -> None:
    root = tmp_path / "large-project"
    source = root / "src"
    source.mkdir(parents=True)
    for index in range(300):
        (source / f"module_{index:03d}.py").write_text(
            "\n".join(
                [
                    f"def scheduler_policy_boundary_{index}():",
                    f"    return 'codex-scale-{index}'",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    db = Database(tmp_path / "scale.db")
    db.initialize()
    index_start = perf_counter()
    index_report = ProjectIndexer(db).index_project(root, project="scale")
    index_seconds = perf_counter() - index_start

    writer = MemoryWriter(db)
    for index in range(250):
        writer.write(
            project="scale",
            memory_type="decision",
            content=(
                f"Architecture note {index}: MemoryScheduler, PolicyArbiter, "
                "semantic layer, and world model communicate through reports."
            ),
            tags=["architecture", "scheduler", "policy", "semantic"],
            importance=0.7,
        )

    pack_start = perf_counter()
    result = TurnOrchestrator(
        db,
        scheduler_state_path=tmp_path / "scheduler.json",
    ).run_turn(
        "Refactor scheduler policy semantic world model boundaries in a large Codex repo",
        context={
            "project": "scale",
            "operation": "memory_pack",
            "writeback": False,
            "compact": True,
            "top_k": 12,
            "source_chunk_limit": 0,
            "min_rank_confidence": 0.0,
        },
    )
    pack_seconds = perf_counter() - pack_start
    pack_tokens = estimate_tokens(result.injected_context)

    assert index_report["indexed_files"] == 300
    assert index_seconds < 30
    assert pack_seconds < 10
    assert pack_tokens <= 2500
    assert result.turn_context.context_budget["task_relevance"]["detected_task_intent"] == (
        "architecture_refactor"
    )
    assert result.turn_context.context_budget["memory_scheduler_report"]
