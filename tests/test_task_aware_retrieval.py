from __future__ import annotations

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.models import SearchResult
from hippocampus_memory.orchestrator.memory_relevance_router import MemoryRelevanceRouter
from hippocampus_memory.orchestrator.task_intent import classify_task_intent
from hippocampus_memory.orchestrator.turn_orchestrator import TurnOrchestrator


def _result(
    memory_id: str,
    content: str,
    *,
    score: float,
    memory_type: str = "technical_fact",
    tags: list[str] | None = None,
    entities: list[str] | None = None,
) -> SearchResult:
    return SearchResult(
        memory_id=memory_id,
        content=content,
        summary=None,
        memory_type=memory_type,
        project="demo",
        importance=0.8,
        confidence=0.8,
        status="active",
        visibility="project",
        score=score,
        matched_reason="test fixture",
        tags=tags or [],
        entities=entities or [],
        score_details={"base": score},
    )


def test_task_intent_detection() -> None:
    assert (
        classify_task_intent(
            "Refactor memory_scheduler and policy orchestration to decouple semantic layers"
        ).intent
        == "architecture_refactor"
    )
    assert classify_task_intent("Fix failing recall bug in CLI route").intent == "bug_fix"
    assert classify_task_intent("Debug why MCP trace returns empty results").intent == "debugging"
    assert classify_task_intent(
        "Change search ranking but avoid leaking private sensitive memories"
    ).intent == "bug_fix"
    assert classify_task_intent("Explain how the world model semantic graph works").intent == (
        "code_understanding"
    )
    assert classify_task_intent("thanks").intent == "general_query"


def test_memory_reranking_improves_relevance() -> None:
    memories = [
        _result(
            "ui-token",
            "Token savings status bar UI history.",
            score=0.9,
            tags=["ui-history", "token-ui"],
        ),
        _result(
            "scheduler-policy",
            "MemoryScheduler policy arbiter semantic world model coupling issue.",
            score=0.45,
            memory_type="decision",
            tags=["scheduler", "policy", "architecture"],
            entities=["MemoryScheduler", "PolicyArbiter"],
        ),
    ]

    routed = MemoryRelevanceRouter().rerank(
        "Refactor memory_scheduler and orchestration logic to decouple policy and semantic layers",
        memories,
    )

    assert routed.memories[0].memory_id == "scheduler-policy"
    assert routed.report.boosted_memories == ["scheduler-policy"]
    assert "ui-token" in routed.report.suppressed_memories
    assert routed.report.scores_after["scheduler-policy"] > routed.report.scores_before[
        "scheduler-policy"
    ]


def test_memory_reranking_clamps_scores_to_one() -> None:
    routed = MemoryRelevanceRouter().rerank(
        "Refactor memory_scheduler policy semantic world model orchestration boundaries.",
        [
            _result(
                "strong-architecture",
                "MemoryScheduler policy orchestrator semantic world model boundary decision.",
                score=0.9,
                memory_type="decision",
                tags=["scheduler", "policy", "architecture"],
            )
        ],
    )

    assert routed.memories[0].score == 1.0
    assert routed.report.scores_after["strong-architecture"] == 1.0
    assert routed.report.adjustments["strong-architecture"]["after"] == 1.0


def test_irrelevant_memory_suppression() -> None:
    routed = MemoryRelevanceRouter().rerank(
        "Architecture refactor for scheduler policy semantic world model separation",
        [
            _result("token-ui", "Token savings UI status line issue.", score=0.8),
            _result(
                "world-model",
                "World model semantic graph influences memory scheduler decisions.",
                score=0.5,
                tags=["world_model", "semantic"],
            ),
        ],
    )

    token = next(memory for memory in routed.memories if memory.memory_id == "token-ui")
    world = next(memory for memory in routed.memories if memory.memory_id == "world-model")
    assert token.score < routed.report.scores_before["token-ui"]
    assert world.score > routed.report.scores_before["world-model"]
    assert routed.report.suppressed_memories == ["token-ui"]


def test_architecture_memory_suppressed_for_non_architecture_bugfix() -> None:
    routed = MemoryRelevanceRouter().rerank(
        "Fix YAML parsing error in config loader",
        [
            _result(
                "architecture-runtime",
                "Architecture Runtime Profile: MemoryScheduler policy orchestrator "
                "semantic world_model boundaries.",
                score=0.85,
                memory_type="constraint",
                tags=["architecture", "scheduler", "policy"],
                entities=["MemoryScheduler", "PolicyArbiter"],
            ),
            _result(
                "failure-trace",
                "Error trace: YAML config loader failed when parsing empty values.",
                score=0.45,
                memory_type="failure",
                tags=["error", "yaml"],
            ),
        ],
    )

    assert routed.memories[0].memory_id == "failure-trace"
    assert "architecture-runtime" in routed.report.suppressed_memories


def test_turn_orchestrator_interface_memory_suppressed_for_general_query() -> None:
    routed = MemoryRelevanceRouter().rerank(
        "thanks",
        [
            _result(
                "orchestrator-interface",
                "Architectural decision: TurnOrchestrator remains the CLI/MCP/API "
                "entry point and owns turn orchestration for compatibility.",
                score=0.85,
                memory_type="decision",
                tags=["architecture", "orchestrator"],
                entities=["TurnOrchestrator", "CLI", "MCP", "API"],
            ),
            _result(
                "ui-history",
                "Decision: Token savings status bar did not show.",
                score=0.75,
                memory_type="decision",
                tags=["ui-history"],
            ),
        ],
    )

    assert "orchestrator-interface" in routed.report.suppressed_memories
    assert "ui-history" in routed.report.suppressed_memories


def test_memory_pack_suppresses_architecture_profile_for_general_query(db) -> None:
    MemoryWriter(db).write(
        project="demo",
        memory_type="constraint",
        content="MemoryScheduler policy orchestrator semantic world_model architecture boundary.",
        tags=["architecture", "scheduler", "policy"],
        entities=["MemoryScheduler", "PolicyArbiter"],
        importance=1.0,
        metadata={
            "architecture_runtime_profile": {
                "layers": ["scheduler", "policy"],
                "interfaces": [],
                "boundary_signals": ["ownership"],
                "canonical_entities": ["MemoryScheduler", "PolicyArbiter"],
            }
        },
    )

    result = TurnOrchestrator(db).run_turn(
        "thanks",
        context={"project": "demo", "operation": "memory_pack", "writeback": False},
        mode="preview",
    )

    assert "Architecture Runtime Profile" not in result.injected_context
    assert "MemoryScheduler policy" not in result.injected_context


def test_unrelated_ui_history_suppressed_for_architecture_tasks() -> None:
    routed = MemoryRelevanceRouter().rerank(
        "Architecture refactor for scheduler policy semantic world model separation",
        [
            _result(
                "ui-history",
                "Prompt-only token UI memory instructions were insufficient.",
                score=0.7,
                memory_type="decision",
                tags=["ui-history"],
            ),
            _result(
                "scheduler-design",
                "MemoryScheduler policy semantic world model boundaries should be separated.",
                score=0.45,
                memory_type="decision",
                tags=["scheduler", "policy", "semantic"],
            ),
        ],
    )

    assert routed.memories[0].memory_id == "scheduler-design"
    assert "ui-history" in routed.report.suppressed_memories


def test_orchestrator_trace_includes_task_aware_relevance(db) -> None:
    writer = MemoryWriter(db)
    writer.write(
        project="demo",
        memory_type="constraint",
        content=(
            "Token savings status bar UI history should not steer scheduler refactors."
        ),
        tags=["ui-history", "token-ui"],
        importance=1.0,
    )
    writer.write(
        project="demo",
        memory_type="decision",
        content="MemoryScheduler policy arbiter semantic world model coupling must be decoupled.",
        tags=["scheduler", "policy", "semantic", "architecture"],
        entities=["MemoryScheduler", "PolicyArbiter"],
        importance=0.5,
    )

    result = TurnOrchestrator(db).run_turn(
        (
            "Refactor memory_scheduler and related orchestration logic so policy and semantic "
            "layers are separated."
        ),
        context={"project": "demo", "writeback": False, "top_k": 8, "min_rank_confidence": 0.0},
        mode="preview",
    )

    rank_event = next(event for event in result.trace if event.node_id == "rank_memories")
    relevance = rank_event.output_state["task_relevance"]
    assert relevance["detected_task_intent"] == "architecture_refactor"
    assert relevance["boosted_memories"]
    assert relevance["suppressed_memories"]
    assert result.turn_context.selected_memories[0].memory_id in relevance["boosted_memories"]


def test_no_regression_on_existing_routes(db) -> None:
    MemoryWriter(db).write(
        project="demo",
        memory_type="failure",
        content="Error trace: CLI route failed because context payload was empty.",
        tags=["error", "trace"],
        importance=0.8,
    )

    result = TurnOrchestrator(db).run_turn(
        "Fix failing CLI route bug",
        context={"project": "demo", "operation": "memory_search", "writeback": False},
        mode="preview",
    )

    assert result.injected_context.startswith("Memory Search Results:")
    assert result.recall_payload["results"]
    assert result.trace


def test_memory_pack_uses_task_aware_selection_for_final_text(db) -> None:
    token_memory = MemoryWriter(db).write(
        project="demo",
        memory_type="constraint",
        content=(
            "Token savings status bar UI history should not appear in "
            "architecture refactor packs."
        ),
        tags=["ui-history", "token-ui"],
        importance=1.0,
    ).memory_id
    architecture_memory = MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content=(
            "MemoryScheduler policy arbiter semantic world model coupling must be "
            "decoupled."
        ),
        tags=["scheduler", "policy", "semantic", "architecture"],
        entities=["MemoryScheduler", "PolicyArbiter"],
        importance=0.5,
    ).memory_id

    result = TurnOrchestrator(db).run_turn(
        "Refactor memory_scheduler policy semantic world model orchestration boundaries.",
        context={
            "project": "demo",
            "operation": "memory_pack",
            "writeback": False,
            "source_chunk_limit": 0,
            "compact": True,
            "top_k": 8,
        },
        mode="preview",
    )

    assert architecture_memory in result.recall_payload["included_memory_ids"]
    assert token_memory not in result.recall_payload["included_memory_ids"]
    assert "MemoryScheduler policy arbiter" in result.injected_context
    assert "Token savings" not in result.injected_context


def test_memory_search_hides_suppressed_memories(db) -> None:
    MemoryWriter(db).write(
        project="demo",
        memory_type="constraint",
        content=(
            "Token savings status bar UI history should not appear in "
            "architecture memory search output."
        ),
        tags=["ui-history", "token-ui"],
        importance=1.0,
    )
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content=(
            "MemoryScheduler policy arbiter semantic world model coupling must be "
            "decoupled."
        ),
        tags=["scheduler", "policy", "semantic", "architecture"],
        entities=["MemoryScheduler", "PolicyArbiter"],
        importance=0.5,
    )

    result = TurnOrchestrator(db).run_turn(
        "Refactor memory_scheduler policy semantic world model orchestration boundaries.",
        context={
            "project": "demo",
            "operation": "memory_search",
            "writeback": False,
            "top_k": 8,
        },
        mode="preview",
    )

    assert "MemoryScheduler policy arbiter" in result.injected_context
    assert "Token savings" not in result.injected_context
    assert all(
        "Token savings" not in item["content"]
        for item in result.recall_payload["results"]
    )


def test_context_bundle_uses_task_aware_selection_for_final_text(db) -> None:
    MemoryWriter(db).write(
        project="demo",
        memory_type="constraint",
        content=(
            "Token savings status bar UI history should not appear in "
            "architecture context bundles."
        ),
        tags=["ui-history", "token-ui"],
        importance=1.0,
    )
    MemoryWriter(db).write(
        project="demo",
        memory_type="decision",
        content=(
            "MemoryScheduler policy arbiter semantic world model coupling must be "
            "decoupled."
        ),
        tags=["scheduler", "policy", "semantic", "architecture"],
        entities=["MemoryScheduler", "PolicyArbiter"],
        importance=0.5,
    )

    result = TurnOrchestrator(db).run_turn(
        "Refactor memory_scheduler policy semantic world model orchestration boundaries.",
        context={
            "project": "demo",
            "operation": "context_bundle",
            "strategy": "lean",
            "include_code_map": False,
            "writeback": False,
            "top_k": 8,
        },
        mode="preview",
    )

    assert "MemoryScheduler policy arbiter" in result.injected_context
    assert "Token savings" not in result.injected_context

