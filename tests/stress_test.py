from __future__ import annotations

from statistics import mean

from hippocampus_memory.db import Database
from tests.utils.system_simulator import SystemSimulationHarness
from tests.utils.trace_logger import MemoryTraceLogger


def test_long_horizon_500_event_simulation_is_stable(tmp_path) -> None:
    db = Database(tmp_path / "stress.db")
    db.initialize()
    harness = SystemSimulationHarness(db, project="stress", state_dir=tmp_path / "state")
    logger = MemoryTraceLogger()
    events = SystemSimulationHarness.generate_events(500, project="stress")

    steps = harness.run_events(events)
    for step in steps:
        logger.record_step(step)

    failures = logger.failure_trace()
    summary = logger.summary()
    snapshot = harness.snapshot(include_archived=True)
    first_window = [step.duration_ms for step in steps[:50]]
    last_window = [step.duration_ms for step in steps[-50:]]

    assert len(steps) == 500
    assert not failures
    assert summary["performance_metrics"]["max_ms"] < 2000.0
    assert mean(last_window) < max(50.0, mean(first_window) * 10.0)
    assert 1 <= snapshot["memory_count"] <= 500
    assert snapshot["status_counts"]
    assert {"write", "recall", "scheduler", "compress", "policy_feedback"} <= set(
        summary["kind_counts"]
    )


def test_stress_trace_logger_can_emit_memory_graph_snapshot(tmp_path) -> None:
    db = Database(tmp_path / "graph-stress.db")
    db.initialize()
    harness = SystemSimulationHarness(db, project="stress", state_dir=tmp_path / "state")
    harness.run_events(SystemSimulationHarness.generate_events(40, project="stress"))

    graph = MemoryTraceLogger().memory_graph_snapshot(db, project="stress")

    assert graph["memory_count"] > 0
    assert graph["node_count"] >= graph["memory_count"]
    assert graph["edge_count"] >= 0
