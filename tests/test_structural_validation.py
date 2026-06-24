from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from hippocampus_memory.memory_writer import MemoryWriter
from hippocampus_memory.orchestrator.turn_orchestrator import (
    DecisionPolicyEngine,
    PolicyWeights,
    TurnOrchestrator,
    TurnResult,
)

PROJECT = "structural-validation"

ENTRYPOINTS = [
    Path("hippocampus_memory/cli.py"),
    Path("hippocampus_memory/mcp_server.py"),
    Path("hippocampus_memory/api.py"),
]

FORBIDDEN_ENTRYPOINT_MODULES = {
    "hippocampus_memory.recall_policy",
    "hippocampus_memory.ranker",
    "hippocampus_memory.memory_policy",
    "hippocampus_memory.orchestrator.memory_scheduler",
}

FORBIDDEN_ENTRYPOINT_CALLS = {
    "build_auto_context",
    "decide_recall",
    "explain_memory_score",
    "auto_store_memories",
    "MemoryScheduler",
    "SemanticMemoryModel",
    "SemanticCompressionEngine",
    "MemoryWorldModel",
    "CognitiveDriveEngine",
}

REQUIRED_TRACE_NODES = {"should_recall", "rank_memories", "execute"}
LIFECYCLE_ACTIONS = {"decay", "promote", "compress", "evict", "demote"}


def test_short_deterministic_structural_validation(db, tmp_path) -> None:
    """Validate one complete memory-runtime turn without changing production code."""

    _seed_structural_memories(db)
    report: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {}

    def run_group(name: str, check: Callable[[], dict[str, Any]]) -> None:
        try:
            report[name] = {"status": "PASS", **check()}
        except Exception as exc:  # pragma: no cover - failure path is reported through pytest
            report[name] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def entrypoint_integrity() -> dict[str, Any]:
        checked: list[str] = []
        for path in ENTRYPOINTS:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports_orchestrator = False
            calls_run_turn = False
            violations: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    names = {alias.name for alias in node.names}
                    if node.module == "hippocampus_memory.orchestrator":
                        imports_orchestrator = "TurnOrchestrator" in names
                    if node.module in FORBIDDEN_ENTRYPOINT_MODULES:
                        violations.append(f"forbidden import from {node.module}")
                elif isinstance(node, ast.Call):
                    call_name = _call_name(node.func)
                    if call_name == "run_turn":
                        calls_run_turn = True
                    if call_name in FORBIDDEN_ENTRYPOINT_CALLS:
                        violations.append(f"forbidden direct call {call_name}()")
            assert imports_orchestrator, f"{path} must import TurnOrchestrator"
            assert calls_run_turn, f"{path} must call TurnOrchestrator.run_turn()"
            assert not violations, f"{path} bypasses orchestrator: {violations}"
            checked.append(str(path))
        return {"checked_entrypoints": checked}

    def turn_execution_smoke() -> dict[str, Any]:
        result = TurnOrchestrator(db).run_turn(
            "test memory recall and context injection",
            context={
                "project": PROJECT,
                "session_key": "qa-structural",
                "max_tokens": 1800,
                "min_rank_confidence": 0.0,
            },
            mode="preview",
        )
        state["result"] = result
        assert isinstance(result, TurnResult)
        assert isinstance(result.injected_context, str)
        assert result.injected_context
        assert isinstance(result.trace, list)
        assert hasattr(result.turn_context, "retrieved_memories")
        assert hasattr(result.turn_context, "selected_memories")
        return {
            "retrieved_memories": len(result.turn_context.retrieved_memories),
            "selected_memories": len(result.turn_context.selected_memories),
            "trace_nodes": _trace_nodes(result),
        }

    def decision_graph_routing() -> dict[str, Any]:
        result = _turn_result(state)
        trace = [event.to_dict() for event in result.trace]
        trace_nodes = {event["node_id"] for event in trace}
        assert REQUIRED_TRACE_NODES <= trace_nodes
        assert "writeback" in trace_nodes or "fallback_lightweight" in trace_nodes
        assert any(node in trace_nodes for node in {"execute", "fallback_lightweight"})
        assert all({"node_id", "decision"} <= event.keys() for event in trace)
        return {
            "trace_node_count": len(trace),
            "routing_nodes": _trace_nodes(result),
        }

    def policy_engine_validation() -> dict[str, Any]:
        result = _turn_result(state)
        engine = DecisionPolicyEngine(tmp_path / "structural-policy.json")
        signals = engine.update_from_trace(PROJECT, result.trace)
        policy_state = engine.state.to_dict()
        json.dumps(policy_state, ensure_ascii=False)
        execute_weights = engine.state.weights_for(PROJECT, "execute")
        assert isinstance(execute_weights, PolicyWeights)
        _assert_policy_weights_are_finite(engine.state.project_weights)
        _assert_policy_weights_are_finite({"__global__": engine.state.global_node_weights})
        assert math.isfinite(engine.state.global_memory_routing_bias)
        return {
            "feedback_signals": signals,
            "policy_turn_count": engine.state.turn_count,
            "execute_weight": execute_weights.to_dict(),
        }

    def scheduler_basic_integration() -> dict[str, Any]:
        scheduler = _scheduler_report(state)
        actions = scheduler["lifecycle_actions"]
        action_types = {action["action_type"] for action in actions}
        assert action_types & LIFECYCLE_ACTIONS, action_types
        assert scheduler["state_version"]
        assert "global_objective" in scheduler
        return {
            "state_version": scheduler["state_version"],
            "lifecycle_actions": sorted(action_types),
        }

    def semantic_report_validation() -> dict[str, Any]:
        semantic = _semantic_report(state)
        required = {
            "profiles",
            "semantic_compression",
            "global_semantic_objective",
            "causality_graph",
            "meaning_consistency",
        }
        assert required <= semantic.keys()
        profiles = semantic["profiles"]
        assert profiles
        semantic_types = {
            profile["semantic_type"]
            for profile in profiles.values()
            if isinstance(profile, Mapping) and profile.get("semantic_type")
        }
        assert semantic_types
        assert semantic["semantic_compression"]
        return {
            "profile_count": len(profiles),
            "semantic_types": sorted(semantic_types),
        }

    def world_model_basic() -> dict[str, Any]:
        world = _semantic_report(state)["world_model"]
        graph = world["graph"]
        assert graph
        assert graph["nodes"]
        assert graph["edges"]
        return {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
        }

    def cognitive_drive_validation() -> dict[str, Any]:
        drive = _semantic_report(state)["cognitive_drive"]
        required = {
            "generated_goals",
            "attention_allocation",
            "memory_driven_task_selection",
        }
        assert required <= drive.keys()
        assert drive["generated_goals"]
        return {
            "goal_count": len(drive["generated_goals"]),
            "selected_task": drive["memory_driven_task_selection"]["selected_task"],
        }

    def integration_consistency() -> dict[str, Any]:
        result = _turn_result(state)
        budget = result.turn_context.context_budget
        scheduler = budget["memory_scheduler_report"]
        semantic = scheduler["semantic_report"]
        world = semantic["world_model"]
        missing_layers = [
            layer
            for layer, present in {
                "orchestrator": bool(result.trace),
                "decision_graph": REQUIRED_TRACE_NODES <= set(_trace_nodes(result)),
                "policy": bool(budget.get("multi_policy_decision_history")),
                "scheduler": bool(scheduler),
                "semantic": bool(semantic),
                "world_model": bool(world.get("graph", {}).get("nodes")),
                "cognitive_drive": bool(semantic.get("cognitive_drive")),
                "output": bool(result.injected_context),
            }.items()
            if not present
        ]
        assert not missing_layers, missing_layers
        assert "system_optimization_report" in budget
        assert "policy_feedback_signals" in budget
        return {
            "missing_layers": missing_layers,
            "context_budget_keys": sorted(budget.keys()),
        }

    def failure_safety() -> dict[str, Any]:
        result = TurnOrchestrator(db).run_turn(
            "@@@invalid memory command###",
            context={"project": PROJECT, "session_key": "qa-invalid", "writeback": False},
            mode="preview",
        )
        assert isinstance(result, TurnResult)
        assert result.injected_context
        assert any(
            marker in result.injected_context
            for marker in {
                "Lightweight Context Fallback",
                "No external memory recall recommended",
                "Memory Pack",
                "Hippocampus Context Bundle",
            }
        )
        return {
            "trace_nodes": _trace_nodes(result),
            "selected_memories": len(result.turn_context.selected_memories),
        }

    run_group("1_entrypoint_integrity", entrypoint_integrity)
    run_group("2_turn_execution_smoke", turn_execution_smoke)
    run_group("3_decision_graph_routing", decision_graph_routing)
    run_group("4_policy_engine_basic_validation", policy_engine_validation)
    run_group("5_scheduler_basic_integration", scheduler_basic_integration)
    run_group("6_semantic_report_validation", semantic_report_validation)
    run_group("7_world_model_basic", world_model_basic)
    run_group("8_cognitive_drive_validation", cognitive_drive_validation)
    run_group("9_integration_consistency", integration_consistency)
    run_group("10_failure_safety", failure_safety)

    failures = {name: result for name, result in report.items() if result["status"] != "PASS"}
    if failures:
        pytest.fail(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _seed_structural_memories(db) -> None:
    writer = MemoryWriter(db)
    writer.write(
        project=PROJECT,
        memory_type="decision",
        content="Decision: test memory recall and context injection must use TurnOrchestrator.",
        entities=["TurnOrchestrator", "context injection"],
        tags=["recall"],
        confidence=0.95,
        importance=0.9,
    )
    old_low = writer.write(
        project=PROJECT,
        memory_type="technical_fact",
        content="Old low confidence implementation note about structural validation.",
        confidence=0.2,
        importance=0.1,
    ).memory_id
    high_value = writer.write(
        project=PROJECT,
        memory_type="decision",
        content="Decision: enable semantic cache.",
        entities=["semantic cache"],
        tags=["cache"],
        confidence=0.95,
        importance=0.9,
    ).memory_id
    writer.write(
        project=PROJECT,
        memory_type="decision",
        content="Do not enable semantic cache.",
        entities=["semantic cache"],
        tags=["cache"],
        confidence=0.85,
        importance=0.8,
    )
    long_memory = writer.write(
        project=PROJECT,
        memory_type="source_chunk",
        content="Long structural validation trace detail. " * 180,
        confidence=0.8,
        importance=0.7,
    ).memory_id
    task_state = writer.write(
        project=PROJECT,
        memory_type="task_state",
        content="Temporary structural validation task state.",
        confidence=0.65,
        importance=0.45,
    ).memory_id
    _age_memory(db, old_low, days=120, usage_count=0)
    _age_memory(db, high_value, days=45, usage_count=8)
    _age_memory(db, long_memory, days=20, usage_count=1)
    _age_memory(db, task_state, days=25, usage_count=0)


def _age_memory(db, memory_id: str, *, days: int, usage_count: int) -> None:
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


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _turn_result(state: dict[str, Any]) -> TurnResult:
    result = state.get("result")
    assert isinstance(result, TurnResult), "turn smoke test did not produce TurnResult"
    return result


def _trace_nodes(result: TurnResult) -> list[str]:
    return [event.node_id for event in result.trace]


def _scheduler_report(state: dict[str, Any]) -> dict[str, Any]:
    result = _turn_result(state)
    scheduler = result.turn_context.context_budget.get("memory_scheduler_report")
    assert isinstance(scheduler, dict), "memory_scheduler_report missing from context_budget"
    return scheduler


def _semantic_report(state: dict[str, Any]) -> dict[str, Any]:
    semantic = _scheduler_report(state).get("semantic_report")
    assert isinstance(semantic, dict) and semantic, "semantic_report missing or empty"
    return semantic


def _assert_policy_weights_are_finite(
    project_weights: Mapping[str, Mapping[str, PolicyWeights]],
) -> None:
    for nodes in project_weights.values():
        for weights in nodes.values():
            assert weights.confidence_score >= 0.0 and math.isfinite(weights.confidence_score)
            assert weights.historical_success_rate >= 0.0 and math.isfinite(
                weights.historical_success_rate
            )
            assert weights.adaptive_routing_weight >= 0.0 and math.isfinite(
                weights.adaptive_routing_weight
            )
            assert all(
                value >= 0.0 and math.isfinite(value)
                for value in weights.decision_weights.values()
            )
