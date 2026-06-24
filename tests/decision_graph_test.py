from __future__ import annotations

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.turn_orchestrator import TurnDecisionGraph, TurnOrchestrator


def test_decision_graph_node_order_for_normal_recall_path(db) -> None:
    MemoryWriter(db).write(
        project="graph",
        memory_type="constraint",
        content="Ranking tests must preserve orchestrator trace output.",
        confidence=0.9,
        importance=0.9,
    )

    result = TurnOrchestrator(db).run_turn(
        "fix ranking tests",
        context={"project": "graph", "writeback": False},
        mode="preview",
    )

    assert [event.node_id for event in result.trace] == [
        "start",
        "should_recall",
        "use_cache",
        "rank_memories",
        "skip_memory",
        "use_full_bundle",
        "execute",
        "writeback",
        "complete",
    ]
    assert result.trace[1].decision == "recall"
    assert result.trace[4].decision == "inject"


def test_decision_graph_skip_memory_goes_to_lightweight_fallback(db) -> None:
    result = TurnOrchestrator(db).run_turn(
        "thanks",
        context={"project": "graph", "writeback": False},
        mode="preview",
    )

    trace = {event.node_id: event for event in result.trace}

    assert trace["should_recall"].decision == "skip"
    assert trace["skip_memory"].decision == "policy_skip"
    assert trace["execute"].decision == "executed"
    assert result.turn_context.selected_memories == []
    assert "No external memory recall recommended" in result.injected_context


def test_decision_graph_low_confidence_ranking_triggers_fallback(db) -> None:
    MemoryWriter(db).write(
        project="graph",
        memory_type="technical_fact",
        content="Unrelated low confidence note about kitchen inventory.",
        confidence=0.1,
        importance=0.0,
    )

    result = TurnOrchestrator(db).run_turn(
        "fix scheduler ranking bug",
        context={"project": "graph", "writeback": False},
        mode="preview",
    )
    skip = [event for event in result.trace if event.node_id == "skip_memory"][-1]

    assert skip.decision == "skip"
    assert skip.output_state["skip_reason"] == "low_rank_confidence"
    assert result.turn_context.selected_memories == []


def test_decision_graph_exposes_required_static_edges() -> None:
    graph = TurnDecisionGraph.default()

    assert graph.next_node("should_recall", "recall") == "use_cache"
    assert graph.next_node("should_recall", "skip") == "skip_memory"
    assert graph.next_node("rank_memories", "recall_failed") == "fallback_lightweight"
    assert graph.next_node("skip_memory", "inject") == "use_full_bundle"
    assert graph.next_node("fallback_lightweight", "fallback") == "writeback"
