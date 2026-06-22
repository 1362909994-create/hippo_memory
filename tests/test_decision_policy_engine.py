from __future__ import annotations

from hippocampus_memory.orchestrator.turn_orchestrator import (
    DecisionPolicyEngine,
    MemoryCostModel,
    PolicyArbiter,
    PolicyDecision,
    PolicyState,
    PolicyWeights,
    TurnOrchestrator,
    TurnTraceEvent,
)


class _StaticPolicy:
    def __init__(
        self,
        name: str,
        decision: str,
        *,
        weight: float = 1.0,
        confidence: float = 1.0,
        safe_mode: bool = False,
    ) -> None:
        self.name = name
        self.decision = decision
        self.weight = weight
        self.confidence = confidence
        self.safe_mode = safe_mode

    def evaluate(self, *, node_id, proposed, allowed, state, cost, context):
        return PolicyDecision(
            policy_name=self.name,
            node_id=node_id,
            preferred_decision=self.decision if self.decision in allowed else proposed,
            confidence=self.confidence,
            weight=self.weight,
            reason=f"{self.name} fixture",
            safe_mode=self.safe_mode,
        )


def test_policy_engine_updates_and_persists_project_weights(tmp_path):
    policy_path = tmp_path / "policy.json"
    engine = DecisionPolicyEngine(policy_path)
    trace = [
        TurnTraceEvent(
            node_id="should_recall",
            decision="recall",
            input_state={},
            output_state={"selected_count": 1},
            next_node="use_cache",
        ),
        TurnTraceEvent(
            node_id="execute",
            decision="executed",
            input_state={},
            output_state={"selected_count": 1},
            next_node="writeback",
        ),
        TurnTraceEvent(
            node_id="complete",
            decision="done",
            input_state={},
            output_state={"selected_count": 1},
            next_node=None,
        ),
    ]

    signals = engine.update_from_trace("demo", trace)
    engine.save()

    assert {"signal": "successful_recall", "reward": 1.0} in signals
    execute_weights = engine.state.weights_for("demo", "execute")
    assert execute_weights.historical_success_rate > 0.5
    assert execute_weights.adaptive_routing_weight > 1.0
    loaded = DecisionPolicyEngine(policy_path)
    assert loaded.state.weights_for("demo", "execute").visits == execute_weights.visits


def test_policy_engine_penalizes_fallback_and_rewards_skip(tmp_path):
    engine = DecisionPolicyEngine(tmp_path / "policy.json")
    trace = [
        TurnTraceEvent(
            node_id="skip_memory",
            decision="skip",
            input_state={},
            output_state={"skip_reason": "low_rank_confidence"},
            next_node="fallback_lightweight",
        ),
        TurnTraceEvent(
            node_id="fallback_lightweight",
            decision="fallback",
            input_state={},
            output_state={"fallback_reason": "low_rank_confidence"},
            next_node="writeback",
        ),
    ]

    signals = engine.update_from_trace("demo", trace)

    assert {"signal": "skip_memory_correctness", "reward": 0.3} in signals
    assert {"signal": "fallback_usage", "reward": -0.2} in signals
    skip_weights = engine.state.weights_for("demo", "skip_memory")
    fallback_weights = engine.state.weights_for("demo", "fallback_lightweight")
    assert skip_weights.adaptive_routing_weight > 1.0
    assert fallback_weights.adaptive_routing_weight < 1.0


def test_policy_bias_can_route_ambiguous_auto_context_to_recall(db, tmp_path):
    engine = DecisionPolicyEngine(tmp_path / "policy.json")
    engine.adjust_edge("demo", "should_recall", "recall", reward=2.0)
    engine.adjust_edge("demo", "should_recall", "skip", reward=-1.0)

    result = TurnOrchestrator(db, policy_engine=engine).run_turn(
        "thanks",
        context={"project": "demo", "writeback": False},
        mode="preview",
    )

    should_recall = [event for event in result.trace if event.node_id == "should_recall"][0]
    assert should_recall.decision == "recall"
    assert should_recall.output_state["policy"]["adaptive_routing_weight"] > 1.0
    assert should_recall.output_state["edge_weight"] > 1.0


def test_policy_state_round_trips_plain_data():
    state = PolicyState()
    state.weights_for("demo", "should_recall").decision_weights["recall"] = 1.7
    state.global_memory_routing_bias = 0.2

    restored = PolicyState.from_dict(state.to_dict())

    assert isinstance(restored.weights_for("demo", "should_recall"), PolicyWeights)
    assert restored.weights_for("demo", "should_recall").decision_weights["recall"] == 1.7
    assert restored.global_memory_routing_bias == 0.2


def test_policy_update_creates_version_chain_and_explainable_delta(tmp_path):
    engine = DecisionPolicyEngine(tmp_path / "policy.json")
    trace = [
        TurnTraceEvent(
            node_id="execute",
            decision="executed",
            input_state={},
            output_state={"selected_count": 1},
            next_node="writeback",
        )
    ]

    engine.update_from_trace("demo", trace)
    engine.update_from_trace("demo", trace)

    history = engine.show_policy_history("demo")
    assert len(history) == 2
    assert history[0]["policy_version_id"]
    assert history[1]["parent_version"] == history[0]["policy_version_id"]
    explanation = engine.explain_policy_change(history[1]["policy_version_id"])
    assert explanation["reason"][0]["signal"] == "successful_recall"
    assert (
        explanation["delta_change"]["execute.executed"]["after"]
        > explanation["delta_change"]["execute.executed"]["before"]
    )


def test_policy_drift_detection_warns_when_rates_move_too_far(tmp_path):
    engine = DecisionPolicyEngine(tmp_path / "policy.json", drift_threshold=0.25)
    fallback_trace = [
        TurnTraceEvent(
            node_id="fallback_lightweight",
            decision="fallback",
            input_state={},
            output_state={"fallback_reason": "test"},
            next_node="writeback",
        )
    ]
    for _ in range(4):
        engine.update_from_trace("demo", fallback_trace)

    drift = engine.detect_policy_drift("demo")

    assert "fallback_rate" in drift["warnings"]
    assert drift["current"]["fallback_rate"] == 1.0


def test_policy_snapshots_and_rollback_restore_previous_state(tmp_path):
    engine = DecisionPolicyEngine(tmp_path / "policy.json", snapshot_interval=2)
    first_trace = [TurnTraceEvent("execute", "executed", {}, {"selected_count": 1}, "writeback")]
    second_trace = [
        TurnTraceEvent(
            "fallback_lightweight", "fallback", {}, {"fallback_reason": "test"}, "writeback"
        )
    ]

    engine.update_from_trace("demo", first_trace)
    first_version = engine.show_policy_history("demo")[0]["policy_version_id"]
    first_weight = engine.state.weights_for("demo", "execute").adaptive_routing_weight
    engine.update_from_trace("demo", second_trace)

    assert engine.state.snapshots
    assert engine.rollback_to_version(first_version)["rolled_back"] is True
    assert engine.state.current_version_id == first_version
    assert engine.state.weights_for("demo", "execute").adaptive_routing_weight == first_weight


def test_policy_diff_between_versions_reports_node_changes(tmp_path):
    engine = DecisionPolicyEngine(tmp_path / "policy.json")
    engine.update_from_trace(
        "demo",
        [TurnTraceEvent("execute", "executed", {}, {"selected_count": 1}, "writeback")],
    )
    first = engine.show_policy_history("demo")[0]["policy_version_id"]
    engine.update_from_trace(
        "demo",
        [
            TurnTraceEvent(
                "fallback_lightweight", "fallback", {}, {"fallback_reason": "x"}, "writeback"
            )
        ],
    )
    second = engine.show_policy_history("demo")[1]["policy_version_id"]

    diff = engine.diff_between_policy_versions(first, second)

    assert "fallback_lightweight" in diff["changed_nodes"]
    assert diff["from_version"] == first
    assert diff["to_version"] == second


def test_policy_guardrails_limit_update_magnitude_and_oscillation(tmp_path):
    engine = DecisionPolicyEngine(
        tmp_path / "policy.json",
        max_update_magnitude=0.05,
        stability_window=3,
    )

    engine.adjust_edge("demo", "should_recall", "recall", reward=100.0)
    assert engine.edge_weight("demo", "should_recall", "recall") <= 1.05
    engine.record_decision("demo", "should_recall", "recall")
    engine.adjust_edge("demo", "should_recall", "skip", reward=100.0)

    selected = engine.select_decision(
        "demo",
        "should_recall",
        "skip",
        ["recall", "skip"],
    )

    assert selected == "recall"
    assert engine.guardrail_warnings[-1]["guardrail"] == "stability_window"


def test_memory_cost_model_estimates_tradeoff_costs():
    cost = MemoryCostModel().estimate(
        node_id="use_full_bundle",
        proposed="full",
        state={
            "input": "fix ranking bug",
            "retrieved_count": 12,
            "selected_count": 5,
            "max_tokens": 1000,
            "injected_tokens": 800,
        },
    )

    assert cost["token_cost"] > 0
    assert cost["memory_retrieval_cost"] == 12.0
    assert cost["context_injection_budget_cost"] == 0.8
    assert 0.0 <= cost["compute_tradeoff_score"] <= 1.0


def test_policy_arbiter_uses_safe_mode_when_policies_conflict():
    arbiter = PolicyArbiter(
        policies=[
            _StaticPolicy("recall", "recall", weight=2.0),
            _StaticPolicy("safety", "skip", weight=1.0, safe_mode=True),
        ]
    )

    result = arbiter.arbitrate(
        project="demo",
        node_id="should_recall",
        proposed="recall",
        allowed=["recall", "skip"],
        state={"input": "read private memory"},
    )

    assert result.final_decision == "skip"
    assert result.safe_mode is True
    assert result.conflicts
    assert result.conflict_trace[0]["resolution"] == "safe_mode"


def test_turn_trace_includes_multi_policy_arbitration(db):
    result = TurnOrchestrator(db).run_turn(
        "fix ranking bug",
        context={"project": "demo", "writeback": False},
        mode="preview",
    )

    should_recall = [event for event in result.trace if event.node_id == "should_recall"][0]
    arbitration = should_recall.output_state["policy_arbitration"]

    assert arbitration["final_decision"] == should_recall.decision
    assert {
        "recall",
        "compression",
        "latency",
        "safety",
        "cost",
    } <= {output["policy_name"] for output in arbitration["policy_outputs"]}
    assert result.turn_context.context_budget["multi_policy_decision_history"]
    assert result.turn_context.context_budget["system_optimization_report"]["objective"]


def test_turn_orchestrator_safe_mode_arbiter_can_override_recall(db):
    arbiter = PolicyArbiter(
        policies=[
            _StaticPolicy("recall", "recall", weight=2.0),
            _StaticPolicy("safety", "skip", weight=1.0, safe_mode=True),
        ]
    )

    result = TurnOrchestrator(db, policy_arbiter=arbiter).run_turn(
        "fix ranking bug",
        context={"project": "demo", "writeback": False},
        mode="preview",
    )

    should_recall = [event for event in result.trace if event.node_id == "should_recall"][0]

    assert should_recall.decision == "skip"
    assert should_recall.output_state["policy_arbitration"]["safe_mode"] is True
    assert result.turn_context.selected_memories == []
