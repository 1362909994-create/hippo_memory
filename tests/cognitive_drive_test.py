from __future__ import annotations

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.memory_scheduler import (
    CognitiveConsistencyEngine,
    CognitiveDriveEngine,
    MemoryWorldModel,
    ReasoningPropagationEngine,
    SemanticFusionEngine,
    SemanticMemoryModel,
)


def test_cognitive_drive_prioritizes_conflicts_and_uncertainty(db) -> None:
    world_report, memories = _conflicting_world_report(db)

    drive = CognitiveDriveEngine().evaluate(world_report=world_report, memories=memories)

    assert drive["generated_goals"][0]["goal_id"] == "resolve_cognitive_conflict"
    assert any(
        item["reason"] == "conflict_driven_attention_spike"
        for item in drive["attention_allocation"]
    )
    assert drive["memory_driven_task_selection"]["selected_task"] == "resolve_cognitive_conflict"
    assert any(
        loop["loop_type"] == "consistency_resolution" for loop in drive["self_triggering_loops"]
    )


def test_cognitive_drive_empty_memory_state_stays_bounded() -> None:
    empty_report = {
        "graph": {"profiles": {}, "nodes": [], "edges": [], "concepts": {}, "entities": {}},
        "semantic_fusion": {"redundancy_eliminated": 0, "global_semantic_graph": {}},
        "cognitive_consistency": {
            "contradictions": [],
            "inconsistent_beliefs": [],
            "unstable_reasoning_chains": [],
        },
        "reasoning_propagation": {"propagation_chains": [], "ripple_effects": []},
        "global_cognitive_state": {"belief_state": {}},
    }

    drive = CognitiveDriveEngine().evaluate(world_report=empty_report, memories=[])

    assert drive["generated_goals"][0]["goal_id"] == "maintain_reasoning_continuity"
    assert drive["self_triggering_loops"] == []
    assert len(drive["continuous_cognitive_flow"]["stream"]) <= 4


def test_cognitive_drive_output_is_deterministic_for_same_world_state(db) -> None:
    world_report, memories = _conflicting_world_report(db)
    engine = CognitiveDriveEngine()

    first = engine.evaluate(world_report=world_report, memories=memories)
    second = engine.evaluate(world_report=world_report, memories=memories)

    assert first == second


def _conflicting_world_report(db):
    writer = MemoryWriter(db)
    positive = writer.write(
        project="drive",
        memory_type="decision",
        content="Decision: enable semantic cache.",
        entities=["semantic cache"],
        confidence=0.9,
        importance=0.9,
    ).memory_id
    negative = writer.write(
        project="drive",
        memory_type="decision",
        content="Do not enable semantic cache.",
        entities=["semantic cache"],
        confidence=0.85,
        importance=0.8,
    ).memory_id
    uncertain = writer.write(
        project="drive",
        memory_type="technical_fact",
        content="Semantic cache always improves every task without tradeoffs.",
        entities=["semantic cache"],
        confidence=0.25,
        importance=0.85,
    ).memory_id
    memories = [db.get_memory(positive), db.get_memory(negative), db.get_memory(uncertain)]
    profiles = {memory.id: SemanticMemoryModel().profile(memory) for memory in memories}
    graph = MemoryWorldModel().build(memories=memories, profiles=profiles)
    fusion = SemanticFusionEngine().fuse(graph)
    propagation = ReasoningPropagationEngine().propagate(
        fusion["global_semantic_graph"], selected_memory_ids=[positive]
    )
    consistency = CognitiveConsistencyEngine().evaluate(
        fusion["global_semantic_graph"], propagation
    )
    world_report = {
        "graph": graph,
        "semantic_fusion": fusion,
        "cognitive_consistency": consistency,
        "reasoning_propagation": propagation,
        "global_cognitive_state": MemoryWorldModel().cognitive_state(
            fusion["global_semantic_graph"]
        ),
    }
    return world_report, memories
