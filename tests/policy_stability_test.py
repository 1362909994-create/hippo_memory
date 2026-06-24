from __future__ import annotations

from hippocampus_memory.orchestrator.turn_orchestrator import (
    DecisionPolicyEngine,
    TurnTraceEvent,
)


def test_repeated_fallback_penalty_remains_bounded_and_detects_drift(tmp_path) -> None:
    engine = DecisionPolicyEngine(tmp_path / "policy.json", max_update_magnitude=0.04)
    success = [TurnTraceEvent("execute", "executed", {}, {"selected_count": 1}, "writeback")]
    trace = [
        TurnTraceEvent(
            "fallback_lightweight",
            "fallback",
            {},
            {"fallback_reason": "stress"},
            "writeback",
        )
    ]

    for _ in range(4):
        engine.update_from_trace("policy", success)
    for _ in range(4):
        engine.update_from_trace("policy", trace)

    weight = engine.edge_weight("policy", "fallback_lightweight", "fallback")
    drift = engine.detect_policy_drift("policy")

    assert 0.1 <= weight <= 3.0
    assert drift["current"]["fallback_rate"] == 1.0
    assert "fallback_rate" in drift["warnings"]


def test_noisy_reward_signals_do_not_create_should_recall_oscillation(tmp_path) -> None:
    engine = DecisionPolicyEngine(
        tmp_path / "policy.json",
        max_update_magnitude=0.05,
        stability_window=4,
    )
    engine.record_decision("policy", "should_recall", "recall")

    selected_decisions = []
    for index in range(3):
        decision = "skip" if index % 2 else "recall"
        engine.adjust_edge("policy", "should_recall", decision, reward=1.0)
        selected = engine.select_decision(
            "policy",
            "should_recall",
            decision,
            ["recall", "skip"],
        )
        selected_decisions.append(selected)
        engine.record_decision("policy", "should_recall", selected)

    assert selected_decisions == ["recall", "recall", "recall"]
    assert engine.guardrail_warnings
    assert engine.guardrail_warnings[-1]["guardrail"] == "stability_window"


def test_conflicting_policy_feedback_creates_explainable_version_history(tmp_path) -> None:
    engine = DecisionPolicyEngine(tmp_path / "policy.json", snapshot_interval=2)
    success = [TurnTraceEvent("execute", "executed", {}, {"selected_count": 1}, "writeback")]
    fallback = [
        TurnTraceEvent(
            "fallback_lightweight",
            "fallback",
            {},
            {"fallback_reason": "noise"},
            "writeback",
        )
    ]

    engine.update_from_trace("policy", success)
    engine.update_from_trace("policy", fallback)
    history = engine.show_policy_history("policy")
    diff = engine.diff_between_policy_versions(
        history[0]["policy_version_id"], history[1]["policy_version_id"]
    )

    assert len(history) == 2
    assert engine.state.snapshots
    assert diff["found"] is True
    assert "fallback_lightweight" in diff["changed_nodes"]
