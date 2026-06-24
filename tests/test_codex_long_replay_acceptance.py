from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

from hippocampus_memory.db import Database
from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.turn_orchestrator import TurnOrchestrator
from hippocampus_memory.utils import estimate_tokens


def test_120_turn_codex_replay_remains_stable_and_bounded(tmp_path: Path) -> None:
    db = Database(tmp_path / "replay.db")
    db.initialize()
    writer = MemoryWriter(db)
    for index in range(10):
        writer.write(
            project="demo",
            memory_type="decision",
            content=(
                f"Replay decision {index}: Codex scheduler policy semantic "
                "world-model reports must remain bounded across turns."
            ),
            tags=["codex", "scheduler", "policy", "semantic", "replay"],
            importance=0.8,
        )

    orchestrator = TurnOrchestrator(db, scheduler_state_path=tmp_path / "scheduler.json")
    start = perf_counter()
    results = []
    for index in range(120):
        results.append(
            orchestrator.run_turn(
                f"Codex replay turn {index}: scheduler policy semantic stability",
                context={
                    "project": "demo",
                    "operation": "memory_pack",
                    "writeback": False,
                    "compact": True,
                    "top_k": 5,
                    "source_chunk_limit": 0,
                    "min_rank_confidence": 0.0,
                },
            )
        )
    elapsed = perf_counter() - start
    last = results[-1]
    budget = last.turn_context.context_budget
    scheduler = budget["memory_scheduler_report"]

    assert elapsed < 60
    assert all(result.trace for result in results)
    assert all(estimate_tokens(result.injected_context) <= 2000 for result in results)
    assert scheduler["state_version"]
    assert scheduler["semantic_report"]["world_model"]["graph"]["nodes"]
    assert scheduler["semantic_report"]["cognitive_drive"]["generated_goals"]
    assert budget["multi_policy_decision_history"]
    assert _all_numbers_finite(budget)


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
