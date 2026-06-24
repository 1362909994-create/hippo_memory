from __future__ import annotations

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.memory_scheduler import (
    CognitiveConsistencyEngine,
    MemoryCausalityGraph,
    MemoryWorldModel,
    ReasoningPropagationEngine,
    SemanticFusionEngine,
    SemanticMemoryModel,
)


def test_world_model_entity_graph_and_semantic_fusion_are_consistent(db) -> None:
    writer = MemoryWriter(db)
    first = writer.write(
        project="world",
        memory_type="decision",
        content="Decision: use MemoryScheduler for lifecycle planning.",
        entities=["MemoryScheduler", "lifecycle planning"],
        tags=["scheduler"],
        confidence=0.9,
        importance=0.85,
    ).memory_id
    second = writer.write(
        project="world",
        memory_type="technical_fact",
        content="Memory scheduler coordinates lifecycle planning.",
        entities=["memory scheduler", "lifecycle planning"],
        tags=["scheduler"],
        confidence=0.8,
        importance=0.7,
    ).memory_id
    memories = [db.get_memory(first), db.get_memory(second)]
    profiles = {memory.id: SemanticMemoryModel().profile(memory) for memory in memories}

    graph = MemoryWorldModel().build(memories=memories, profiles=profiles)
    fusion = SemanticFusionEngine().fuse(graph)

    assert graph["nodes"]
    assert graph["edges"]
    assert "lifecycle planning" in graph["entities"]
    assert fusion["merged_concepts"]
    assert fusion["redundancy_eliminated"] >= 1


def test_causality_and_reasoning_propagation_link_memory_influence(db) -> None:
    writer = MemoryWriter(db)
    source = writer.write(
        project="world",
        memory_type="decision",
        content="Decision: recall policy influences scheduler policy.",
        entities=["scheduler policy"],
        confidence=0.9,
        importance=0.9,
    ).memory_id
    target = writer.write(
        project="world",
        memory_type="technical_fact",
        content="Scheduler policy affects long-term task performance.",
        entities=["scheduler policy"],
        confidence=0.75,
        importance=0.75,
    ).memory_id
    memories = [db.get_memory(source), db.get_memory(target)]
    profiles = {memory.id: SemanticMemoryModel().profile(memory) for memory in memories}
    graph = MemoryWorldModel().build(memories=memories, profiles=profiles)

    propagation = ReasoningPropagationEngine().propagate(graph, selected_memory_ids=[source])
    causality = MemoryCausalityGraph().build(
        memories=memories,
        selected_memory_ids=[source],
        trace=[{"node_id": "execute", "decision": "executed"}],
        decision="executed",
        profiles=profiles,
    )

    assert {source, target} <= {chain["from"] for chain in propagation["propagation_chains"]} | {
        chain["to"] for chain in propagation["propagation_chains"]
    }
    assert causality["edges"][0]["from"] == source
    assert causality["impact_chain"][0]["causal_role"] == "decision_memory"


def test_world_model_detects_conflicting_beliefs_without_adding_new_contradictions(db) -> None:
    writer = MemoryWriter(db)
    positive = writer.write(
        project="world",
        memory_type="decision",
        content="Decision: enable semantic cache.",
        confidence=0.9,
        importance=0.9,
    ).memory_id
    negative = writer.write(
        project="world",
        memory_type="decision",
        content="Do not enable semantic cache.",
        confidence=0.85,
        importance=0.8,
    ).memory_id
    memories = [db.get_memory(positive), db.get_memory(negative)]
    profiles = {memory.id: SemanticMemoryModel().profile(memory) for memory in memories}
    graph = MemoryWorldModel().build(memories=memories, profiles=profiles)
    propagation = ReasoningPropagationEngine().propagate(graph, selected_memory_ids=[positive])

    consistency = CognitiveConsistencyEngine().evaluate(graph, propagation)

    assert consistency["contradictions"]
    assert consistency["contradictions"][0]["memory_ids"] == [positive, negative]
    assert consistency["cognitively_consistent"] is False
