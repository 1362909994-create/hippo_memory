from __future__ import annotations

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.turn_orchestrator import (
    TurnContext,
    TurnDecisionGraph,
    TurnOrchestrator,
)


def _trace_dicts(result):
    return [event.to_dict() for event in result.trace]


def test_turn_decision_graph_exposes_required_runtime_nodes():
    graph = TurnDecisionGraph.default()

    assert set(graph.nodes) >= {
        "should_recall",
        "use_cache",
        "rank_memories",
        "skip_memory",
        "use_full_bundle",
        "execute",
        "fallback_lightweight",
        "writeback",
        "complete",
    }
    assert graph.next_node("should_recall", "recall") == "use_cache"
    assert graph.next_node("rank_memories", "recall_failed") == "fallback_lightweight"
    assert graph.next_node("skip_memory", "skip") == "fallback_lightweight"


def test_turn_orchestrator_builds_context_and_trace(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="constraint",
        content="Search changes must preserve sensitive memory filters.",
        importance=0.9,
    )

    result = TurnOrchestrator(db).run_turn(
        "fix search ranking bug",
        context={"project": "demo", "max_tokens": 1200, "session_key": "s1"},
        mode="preview",
    )

    assert isinstance(result.turn_context, TurnContext)
    assert result.injected_context
    assert (
        "Memory Pack" in result.injected_context
        or "Hippocampus Context Bundle" in result.injected_context
    )
    assert result.turn_context.retrieved_memories
    assert result.turn_context.selected_memories
    assert result.turn_context.context_budget["max_tokens"] == 1200
    trace = _trace_dicts(result)
    assert all(
        {"node_id", "decision", "input_state", "output_state", "next_node"} <= event.keys()
        for event in trace
    )
    assert [entry["node_id"] for entry in trace] == [
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
    assert trace[1]["decision"] == "recall"
    assert trace[4]["decision"] == "inject"


def test_turn_orchestrator_writeback_is_preview_by_default(db):
    result = TurnOrchestrator(db).run_turn(
        "Decision: use the orchestrator as the single future turn entrypoint.",
        context={"project": "demo"},
        mode="preview",
    )

    assert result.memory_writeback is not None
    assert result.memory_writeback["dry_run"] is True
    assert result.memory_writeback["written"] == 0
    assert db.list_memories(project="demo") == []
    writeback_event = [event for event in _trace_dicts(result) if event["node_id"] == "writeback"]
    assert writeback_event[0]["decision"] == "preview"


def test_turn_orchestrator_can_skip_writeback(db):
    result = TurnOrchestrator(db).run_turn(
        "thanks",
        context={"project": "demo", "writeback": False},
        mode="preview",
    )

    assert result.memory_writeback is None
    assert result.turn_context.selected_memories == []
    assert "No external memory recall recommended" in result.injected_context


def test_decision_graph_skips_low_confidence_memory_injection(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="technical_fact",
        content="Unrelated note about coffee machine maintenance.",
        importance=0.0,
        confidence=0.1,
    )

    result = TurnOrchestrator(db).run_turn(
        "fix ranking bug",
        context={"project": "demo", "writeback": False},
        mode="preview",
    )

    trace = _trace_dicts(result)
    skip_event = [event for event in trace if event["node_id"] == "skip_memory"][0]
    assert skip_event["decision"] == "skip"
    assert skip_event["output_state"]["skip_reason"] == "low_rank_confidence"
    assert result.turn_context.retrieved_memories
    assert result.turn_context.selected_memories == []
    assert "Lightweight Context Fallback" in result.injected_context


def test_decision_graph_falls_back_when_recall_fails(db, monkeypatch):
    MemoryWriter(db).write(
        project="demo",
        memory_type="task_state",
        content="Current task is to test recall fallback routing.",
        importance=0.9,
    )

    def fail_recall(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "hippocampus_memory.orchestrator.turn_orchestrator.build_auto_context",
        fail_recall,
    )

    result = TurnOrchestrator(db).run_turn(
        "continue fallback routing",
        context={"project": "demo", "writeback": False},
        mode="preview",
    )

    trace = _trace_dicts(result)
    execute_event = [event for event in trace if event["node_id"] == "execute"][0]
    assert execute_event["decision"] == "recall_failed"
    assert any(event["node_id"] == "fallback_lightweight" for event in trace)
    assert "Lightweight Context Fallback" in result.injected_context
