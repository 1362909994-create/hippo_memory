from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.memory_scheduler import (
    CognitiveConsistencyEngine,
    CognitiveDriveEngine,
    MemoryCausalityGraph,
    MemoryHierarchy,
    MemoryScheduler,
    MemoryWorldModel,
    ReasoningPropagationEngine,
    SemanticCompressionEngine,
    SemanticFusionEngine,
    SemanticMemoryModel,
    SystemStabilityController,
)
from hippocampus_memory.orchestrator.turn_orchestrator import PolicyArbiter, TurnOrchestrator


def _age_memory(db, memory_id: str, *, days: int, usage_count: int = 0) -> None:
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


def test_memory_scheduler_plans_full_lifecycle(db, tmp_path):
    writer = MemoryWriter(db)
    old_low = writer.write(
        project="demo",
        memory_type="technical_fact",
        content="Old low confidence implementation note.",
        confidence=0.2,
        importance=0.1,
    ).memory_id
    high_value = writer.write(
        project="demo",
        memory_type="decision",
        content="Use TurnOrchestrator as the single memory runtime entrypoint.",
        confidence=0.95,
        importance=0.9,
    ).memory_id
    long_memory = writer.write(
        project="demo",
        memory_type="source_chunk",
        content="Long trace detail. " * 120,
        confidence=0.8,
        importance=0.7,
    ).memory_id
    short_term = writer.write(
        project="demo",
        memory_type="task_state",
        content="Temporary note that has not been reused.",
        confidence=0.65,
        importance=0.45,
    ).memory_id
    _age_memory(db, old_low, days=120)
    _age_memory(db, high_value, days=45, usage_count=8)
    _age_memory(db, long_memory, days=20, usage_count=1)
    _age_memory(db, short_term, days=25, usage_count=0)

    scheduler = MemoryScheduler(db, state_path=tmp_path / "scheduler.json")
    report = scheduler.run_cycle(project="demo")

    actions_by_type = {action.action_type: action for action in report.lifecycle_actions}
    assert actions_by_type["decay"].memory_id == old_low
    assert actions_by_type["evict"].memory_id == old_low
    assert actions_by_type["promote"].memory_id == high_value
    assert actions_by_type["compress"].memory_id == long_memory
    assert actions_by_type["demote"].memory_id == short_term
    assert report.global_objective["maximize"]["memory_usefulness"] >= 0.0


def test_memory_scheduler_persists_cross_turn_state(db, tmp_path):
    scheduler_path = tmp_path / "scheduler.json"
    scheduler = MemoryScheduler(db, state_path=scheduler_path)

    first = scheduler.run_cycle(
        project="demo",
        policy_feedback_signals=[{"signal": "successful_recall", "reward": 1.0}],
    )
    second = scheduler.run_cycle(
        project="demo",
        policy_feedback_signals=[{"signal": "fallback_usage", "reward": -0.2}],
    )
    loaded = MemoryScheduler(db, state_path=scheduler_path)

    assert first.state_version != second.state_version
    assert loaded.state.turn_count == 2
    assert loaded.state.shared_reward_signal < first.shared_reward_signal
    assert loaded.state.lifecycle_history[-1]["project"] == "demo"


def test_memory_scheduler_aligns_policy_weights_from_conflicts(db, tmp_path):
    scheduler = MemoryScheduler(db, state_path=tmp_path / "scheduler.json")
    arbiter = PolicyArbiter()
    safety = [policy for policy in arbiter.policies if policy.name == "safety"][0]
    before = safety.weight

    report = scheduler.run_cycle(
        project="demo",
        policy_conflict_trace=[{"resolution": "safe_mode", "final_decision": "skip"}],
        policy_arbiter=arbiter,
    )

    assert safety.weight > before
    assert report.policy_alignment["shared_reward_signal"] < 0.0
    assert report.policy_alignment["synchronized_policies"]["safety"] > before
    assert report.optimization_loop["policy_conflict_frequency"] == 1.0


def test_turn_orchestrator_runs_memory_scheduler(db):
    MemoryWriter(db).write(
        project="demo",
        memory_type="constraint",
        content="Scheduler integration should not change CLI or API output shape.",
        confidence=0.9,
        importance=0.9,
    )

    result = TurnOrchestrator(db).run_turn(
        "fix scheduler integration",
        context={"project": "demo", "writeback": False},
        mode="preview",
    )

    scheduler_report = result.turn_context.context_budget["memory_scheduler_report"]
    assert scheduler_report["state_version"] == "scheduler-000001"
    assert "global_objective" in scheduler_report
    assert "policy_alignment" in scheduler_report
    assert "optimization_loop" in scheduler_report


def test_memory_hierarchy_assigns_tiers_and_retrieval_weights(db):
    writer = MemoryWriter(db)
    task_state = writer.write(
        project="demo",
        memory_type="task_state",
        content="Current refactor step is scheduler hierarchy.",
        confidence=0.7,
        importance=0.5,
    ).memory_id
    long_term = writer.write(
        project="demo",
        memory_type="decision",
        content="Keep orchestrator as the single memory runtime entrypoint.",
        confidence=0.95,
        importance=0.95,
    ).memory_id
    archival = writer.write(
        project="demo",
        memory_type="technical_fact",
        content="Very old implementation detail kept for audit.",
        confidence=0.2,
        importance=0.1,
    ).memory_id
    _age_memory(db, task_state, days=1, usage_count=0)
    _age_memory(db, long_term, days=40, usage_count=9)
    _age_memory(db, archival, days=180, usage_count=0)

    memories = db.list_memories(project="demo", include_archived=True)
    hierarchy = MemoryHierarchy()
    assignments = hierarchy.assign_tiers(memories, working_memory_ids=[task_state])

    assert assignments[task_state].tier == "L0"
    assert assignments[long_term].tier == "L2"
    assert assignments[archival].tier == "L3"
    assert assignments[task_state].retrieval_weight > assignments[archival].retrieval_weight
    assert hierarchy.promotion_target(assignments[long_term]) == "L2"
    assert hierarchy.demotion_target(assignments[archival]) == "L3"


def test_memory_scheduler_reports_consistency_violations(db, tmp_path):
    writer = MemoryWriter(db)
    first = writer.write(
        project="demo",
        memory_type="decision",
        content="Use Redis cache for session state.",
        confidence=0.9,
        importance=0.8,
    ).memory_id
    duplicate = writer.write(
        project="demo",
        memory_type="technical_fact",
        content="Use Redis cache for session state.",
        confidence=0.8,
        importance=0.6,
    ).memory_id
    writer.write(
        project="demo",
        memory_type="decision",
        content="Do not use Redis cache for session state.",
        confidence=0.85,
        importance=0.7,
    )

    report = MemoryScheduler(db, state_path=tmp_path / "scheduler.json").run_cycle(project="demo")

    consistency = report.consistency_report
    assert consistency["commit_allowed"] is False
    assert {first, duplicate} <= set(consistency["uniqueness_violations"][0]["memory_ids"])
    assert consistency["contradictions"]


def test_load_aware_scheduler_reduces_compression_and_defers_eviction(db, tmp_path):
    writer = MemoryWriter(db)
    long_memory = writer.write(
        project="demo",
        memory_type="source_chunk",
        content="Large context detail. " * 180,
        confidence=0.9,
        importance=0.8,
    ).memory_id
    weak_memory = writer.write(
        project="demo",
        memory_type="technical_fact",
        content="Weak stale memory.",
        confidence=0.2,
        importance=0.1,
    ).memory_id
    _age_memory(db, long_memory, days=20, usage_count=1)
    _age_memory(db, weak_memory, days=120, usage_count=0)

    report = MemoryScheduler(db, state_path=tmp_path / "scheduler.json").run_cycle(
        project="demo",
        system_load={"active_sessions": 3, "cpu_load": 0.92, "low_confidence_state": True},
    )

    actions_by_type = {action.action_type: action for action in report.lifecycle_actions}
    assert actions_by_type["compress"].metadata["deferred"] is True
    assert actions_by_type["evict"].metadata["deferred"] is True
    assert report.optimization_loop["scheduler_interval_turns"] > 1.0
    assert report.optimization_loop["recall_priority"] > 1.0


def test_stability_controller_blocks_rapid_tier_oscillation():
    controller = SystemStabilityController(convergence_window=3)

    assert controller.allow_transition("mem1", "L1", "L2", turn=1)["allowed"] is True
    second = controller.allow_transition("mem1", "L2", "L1", turn=2)

    assert second["allowed"] is False
    assert second["reason"] == "convergence_window"
    assert controller.allow_transition("mem1", "L2", "L1", turn=5)["allowed"] is True


def test_semantic_memory_model_explains_meaning(db):
    memory_id = (
        MemoryWriter(db)
        .write(
            project="demo",
            memory_type="failure",
            content="Error: cache invalidation failed because stale session keys were reused.",
            confidence=0.9,
            importance=0.85,
        )
        .memory_id
    )
    memory = db.get_memory(memory_id)

    profile = SemanticMemoryModel().profile(memory)

    assert profile.semantic_type == "error"
    assert profile.importance_meaning == "prevents repeated failure"
    assert profile.context_role == "failure_avoidance"
    assert profile.reasoning_utility > 0.0


def test_semantic_compression_merges_duplicate_meaning(db):
    writer = MemoryWriter(db)
    first = db.get_memory(
        writer.write(
            project="demo",
            memory_type="decision",
            content="Decision: use TurnOrchestrator as the memory runtime entrypoint.",
            confidence=0.9,
            importance=0.8,
        ).memory_id
    )
    second = db.get_memory(
        writer.write(
            project="demo",
            memory_type="technical_fact",
            content="Use TurnOrchestrator as the memory runtime entrypoint.",
            confidence=0.8,
            importance=0.7,
        ).memory_id
    )
    model = SemanticMemoryModel()

    result = SemanticCompressionEngine().compress([first, second], model=model)

    assert len(result["merged_meanings"]) == 1
    merged = result["merged_meanings"][0]
    assert {first.id, second.id} <= set(merged["memory_ids"])
    assert "TurnOrchestrator" in merged["semantic_summary"]
    assert merged["meaning_preserved"] is True


def test_memory_causality_graph_links_memory_to_decision(db):
    memory_id = (
        MemoryWriter(db)
        .write(
            project="demo",
            memory_type="decision",
            content="Decision: keep scheduler changes non-destructive.",
            confidence=0.9,
            importance=0.9,
        )
        .memory_id
    )
    memory = db.get_memory(memory_id)

    graph = MemoryCausalityGraph().build(
        memories=[memory],
        selected_memory_ids=[memory_id],
        trace=[{"node_id": "execute", "decision": "executed"}],
        decision="executed",
        profiles={memory_id: SemanticMemoryModel().profile(memory)},
    )

    assert graph["edges"][0]["from"] == memory_id
    assert graph["edges"][0]["to"] == "decision:executed"
    assert graph["impact_chain"][0]["causal_role"] == "decision_memory"
    assert graph["explanations"][0]["why_recalled"]


def test_memory_scheduler_outputs_semantic_report(db, tmp_path):
    writer = MemoryWriter(db)
    selected = writer.write(
        project="demo",
        memory_type="decision",
        content="Decision: preserve deterministic fallback behavior.",
        confidence=0.9,
        importance=0.9,
    ).memory_id
    outdated = writer.write(
        project="demo",
        memory_type="decision",
        content="Decision: use the deprecated direct recall path.",
        confidence=0.95,
        importance=0.95,
    ).memory_id
    writer.write(
        project="demo",
        memory_type="decision",
        content="Do not use the deprecated direct recall path.",
        confidence=0.85,
        importance=0.8,
    )
    _age_memory(db, outdated, days=180, usage_count=9)

    report = MemoryScheduler(db, state_path=tmp_path / "scheduler.json").run_cycle(
        project="demo",
        selected_memory_ids=[selected],
        trace=[{"node_id": "execute", "decision": "executed"}],
    )

    semantic = report.semantic_report
    assert semantic["global_semantic_objective"]["maximize"]["reasoning_utility"] > 0.0
    assert semantic["causality_graph"]["edges"]
    assert semantic["meaning_consistency"]["contradictory_semantic_memories"]
    assert outdated in semantic["meaning_consistency"]["outdated_high_weight_memories"]


def test_memory_world_model_maps_memories_to_cognitive_nodes(db):
    writer = MemoryWriter(db)
    decision = writer.write(
        project="demo",
        memory_type="decision",
        content="Decision: use SQLite for local memory storage.",
        entities=["SQLite", "memory storage"],
        tags=["storage", "local-first"],
        confidence=0.9,
        importance=0.85,
    ).memory_id
    event = writer.write(
        project="demo",
        memory_type="failure",
        content="Error: vector index failed during recall.",
        entities=["vector index", "recall"],
        tags=["retrieval"],
        confidence=0.8,
        importance=0.7,
    ).memory_id
    memories = [db.get_memory(decision), db.get_memory(event)]
    profiles = {memory.id: SemanticMemoryModel().profile(memory) for memory in memories}

    graph = MemoryWorldModel().build(memories=memories, profiles=profiles)

    assert "SQLite" in graph["entities"]
    assert "memory storage" in graph["concepts"]
    assert decision in graph["decisions"]
    assert event in graph["events"]
    assert graph["memory_node_map"][decision]
    assert graph["nodes"]


def test_semantic_fusion_merges_concepts_and_eliminates_world_redundancy(db):
    writer = MemoryWriter(db)
    first = writer.write(
        project="demo",
        memory_type="decision",
        content="Decision: use MemoryScheduler for lifecycle planning.",
        entities=["MemoryScheduler"],
        tags=["lifecycle"],
        confidence=0.9,
        importance=0.8,
    ).memory_id
    second = writer.write(
        project="demo",
        memory_type="technical_fact",
        content="Use MemoryScheduler for lifecycle planning.",
        entities=["memory scheduler"],
        tags=["lifecycle"],
        confidence=0.8,
        importance=0.7,
    ).memory_id
    memories = [db.get_memory(first), db.get_memory(second)]
    model = SemanticMemoryModel()
    profiles = {memory.id: model.profile(memory) for memory in memories}
    graph = MemoryWorldModel().build(memories=memories, profiles=profiles)

    fusion = SemanticFusionEngine().fuse(graph)

    assert fusion["merged_concepts"]
    assert fusion["redundancy_eliminated"] >= 1
    assert first in fusion["global_semantic_graph"]["memory_node_map"]


def test_cognitive_consistency_detects_belief_conflicts_and_unstable_chains(db):
    writer = MemoryWriter(db)
    positive = writer.write(
        project="demo",
        memory_type="decision",
        content="Decision: enable semantic cache.",
        confidence=0.9,
        importance=0.9,
    ).memory_id
    negative = writer.write(
        project="demo",
        memory_type="decision",
        content="Do not enable semantic cache.",
        confidence=0.85,
        importance=0.8,
    ).memory_id
    weak = writer.write(
        project="demo",
        memory_type="technical_fact",
        content="Semantic cache improves every task without tradeoffs.",
        confidence=0.25,
        importance=0.9,
    ).memory_id
    memories = [db.get_memory(positive), db.get_memory(negative), db.get_memory(weak)]
    profiles = {memory.id: SemanticMemoryModel().profile(memory) for memory in memories}
    graph = MemoryWorldModel().build(memories=memories, profiles=profiles)
    propagation = ReasoningPropagationEngine().propagate(graph, selected_memory_ids=[positive])

    consistency = CognitiveConsistencyEngine().evaluate(graph, propagation)

    assert consistency["contradictions"]
    assert consistency["inconsistent_beliefs"]
    assert consistency["unstable_reasoning_chains"]


def test_reasoning_propagation_and_global_cognitive_state(db):
    writer = MemoryWriter(db)
    first = writer.write(
        project="demo",
        memory_type="decision",
        content="Decision: recall policy influences scheduler policy.",
        entities=["recall policy", "scheduler policy"],
        confidence=0.9,
        importance=0.9,
    ).memory_id
    second = writer.write(
        project="demo",
        memory_type="technical_fact",
        content="Insight: scheduler policy affects long-term task performance.",
        entities=["scheduler policy", "long-term task performance"],
        confidence=0.75,
        importance=0.75,
    ).memory_id
    memories = [db.get_memory(first), db.get_memory(second)]
    profiles = {memory.id: SemanticMemoryModel().profile(memory) for memory in memories}
    graph = MemoryWorldModel().build(memories=memories, profiles=profiles)

    propagation = ReasoningPropagationEngine().propagate(graph, selected_memory_ids=[first])
    cognitive_state = MemoryWorldModel().cognitive_state(graph)

    assert propagation["propagation_chains"]
    assert propagation["ripple_effects"]
    assert cognitive_state["belief_state"]
    assert cognitive_state["confidence_distribution"]["high"] >= 1
    assert cognitive_state["uncertainty_tracking"]["mean_uncertainty"] >= 0.0


def test_scheduler_semantic_report_includes_world_model(db, tmp_path):
    selected = (
        MemoryWriter(db)
        .write(
            project="demo",
            memory_type="decision",
            content="Decision: semantic memories form a world model.",
            entities=["semantic memories", "world model"],
            confidence=0.9,
            importance=0.85,
        )
        .memory_id
    )

    report = MemoryScheduler(db, state_path=tmp_path / "scheduler.json").run_cycle(
        project="demo",
        selected_memory_ids=[selected],
    )

    world = report.semantic_report["world_model"]
    assert world["graph"]["nodes"]
    assert world["semantic_fusion"]["global_semantic_graph"]
    assert "cognitive_consistency" in world
    assert "reasoning_propagation" in world
    assert "global_cognitive_state" in world


def test_cognitive_drive_engine_generates_goals_attention_and_loops(db):
    writer = MemoryWriter(db)
    positive = writer.write(
        project="demo",
        memory_type="decision",
        content="Decision: enable semantic cache.",
        entities=["semantic cache"],
        confidence=0.9,
        importance=0.9,
    ).memory_id
    negative = writer.write(
        project="demo",
        memory_type="decision",
        content="Do not enable semantic cache.",
        entities=["semantic cache"],
        confidence=0.85,
        importance=0.8,
    ).memory_id
    uncertain = writer.write(
        project="demo",
        memory_type="technical_fact",
        content="Semantic cache always improves every task without tradeoffs.",
        entities=["semantic cache"],
        confidence=0.25,
        importance=0.85,
    ).memory_id
    memories = [db.get_memory(positive), db.get_memory(negative), db.get_memory(uncertain)]
    model = SemanticMemoryModel()
    profiles = {memory.id: model.profile(memory) for memory in memories}
    graph = MemoryWorldModel().build(memories=memories, profiles=profiles)
    fusion = SemanticFusionEngine().fuse(graph)
    propagation = ReasoningPropagationEngine().propagate(
        fusion["global_semantic_graph"],
        selected_memory_ids=[positive],
    )
    consistency = CognitiveConsistencyEngine().evaluate(
        fusion["global_semantic_graph"],
        propagation,
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

    drive = CognitiveDriveEngine().evaluate(world_report=world_report, memories=memories)

    assert drive["generated_goals"]
    assert drive["generated_goals"][0]["trigger"] == "cognitive_conflict"
    assert any(
        item["reason"] == "conflict_driven_attention_spike"
        for item in drive["attention_allocation"]
    )
    assert drive["memory_driven_task_selection"]["selected_task"] == "resolve_cognitive_conflict"
    assert drive["self_triggering_loops"]
    assert any(
        loop["loop_type"] == "consistency_resolution" for loop in drive["self_triggering_loops"]
    )
    assert drive["unified_cognitive_objective"]["minimize"]["contradiction_density"] > 0.0
    assert drive["continuous_cognitive_flow"]["stream"]


def test_scheduler_semantic_report_includes_cognitive_drive(db, tmp_path):
    writer = MemoryWriter(db)
    selected = writer.write(
        project="demo",
        memory_type="decision",
        content="Decision: keep API routing through orchestrator.",
        entities=["API routing", "orchestrator"],
        confidence=0.9,
        importance=0.85,
    ).memory_id
    writer.write(
        project="demo",
        memory_type="technical_fact",
        content="API routing depends on orchestrator trace continuity.",
        entities=["API routing", "orchestrator"],
        confidence=0.7,
        importance=0.7,
    )

    report = MemoryScheduler(db, state_path=tmp_path / "scheduler.json").run_cycle(
        project="demo",
        selected_memory_ids=[selected],
    )

    drive = report.semantic_report["cognitive_drive"]
    assert drive["generated_goals"]
    assert drive["attention_allocation"]
    assert drive["memory_driven_task_selection"]["selected_task"]
    assert drive["unified_cognitive_objective"]["maximize"]["memory_utility_alignment"] >= 0.0
