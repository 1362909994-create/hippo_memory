from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from hippocampus_memory.callback import callback_pack
from hippocampus_memory.codegraph_bootstrap import attach_codegraph_bootstrap_suggestion
from hippocampus_memory.context_bundle import ContextBundleBuilder
from hippocampus_memory.db import Database
from hippocampus_memory.memory_policy import auto_store_memories
from hippocampus_memory.models import MemoryRecord, SearchResult
from hippocampus_memory.orchestrator.memory_relevance_router import MemoryRelevanceRouter
from hippocampus_memory.orchestrator.memory_scheduler import MemoryScheduler
from hippocampus_memory.packer import MemoryPacker
from hippocampus_memory.ranker import RANKER_VERSION, explain_memory_score
from hippocampus_memory.recall_policy import RecallDecision, build_auto_context, decide_recall
from hippocampus_memory.retriever import Retriever
from hippocampus_memory.utils import estimate_tokens, normalize_text, tokenize, utc_now

TurnMode = Literal["preview", "execute", "write"]


@dataclass(frozen=True, slots=True)
class DecisionNode:
    node_id: str
    label: str
    description: str
    confidence_score: float = 0.5
    historical_success_rate: float = 0.5
    adaptive_routing_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class DecisionEdge:
    source: str
    decision: str
    target: str | None
    base_weight: float = 1.0


@dataclass(slots=True)
class TurnTraceEvent:
    node_id: str
    decision: str
    input_state: dict[str, Any]
    output_state: dict[str, Any]
    next_node: str | None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "decision": self.decision,
            "input_state": self.input_state,
            "output_state": self.output_state,
            "next_node": self.next_node,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class PolicyWeights:
    confidence_score: float = 0.5
    historical_success_rate: float = 0.5
    adaptive_routing_weight: float = 1.0
    visits: int = 0
    reward_total: float = 0.0
    decision_weights: dict[str, float] = field(default_factory=dict)

    def apply_reward(
        self,
        reward: float,
        decision: str | None = None,
        *,
        max_update_magnitude: float = 0.1,
    ) -> None:
        bounded_reward = _clamp_float(reward, -max_update_magnitude, max_update_magnitude)
        previous_visits = self.visits
        self.visits += 1
        self.reward_total += reward
        success_value = 1.0 if reward > 0 else 0.0
        self.historical_success_rate = _clamp_float(
            (self.historical_success_rate * previous_visits + success_value) / self.visits
        )
        self.confidence_score = _clamp_float(
            (self.confidence_score * previous_visits + min(1.0, abs(reward))) / self.visits
        )
        self.adaptive_routing_weight = _clamp_float(
            self.adaptive_routing_weight + bounded_reward,
            0.1,
            3.0,
        )
        if decision:
            current = self.decision_weights.get(decision, 1.0)
            self.decision_weights[decision] = _clamp_float(
                current + bounded_reward,
                0.1,
                3.0,
            )

    def edge_weight(self, decision: str) -> float:
        return self.decision_weights.get(decision, self.adaptive_routing_weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_score": self.confidence_score,
            "historical_success_rate": self.historical_success_rate,
            "adaptive_routing_weight": self.adaptive_routing_weight,
            "visits": self.visits,
            "reward_total": self.reward_total,
            "decision_weights": self.decision_weights,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyWeights:
        return cls(
            confidence_score=float(data.get("confidence_score", 0.5)),
            historical_success_rate=float(data.get("historical_success_rate", 0.5)),
            adaptive_routing_weight=float(data.get("adaptive_routing_weight", 1.0)),
            visits=int(data.get("visits", 0)),
            reward_total=float(data.get("reward_total", 0.0)),
            decision_weights={
                str(key): float(value)
                for key, value in dict(data.get("decision_weights", {})).items()
            },
        )


@dataclass(slots=True)
class PolicyState:
    project_weights: dict[str, dict[str, PolicyWeights]] = field(default_factory=dict)
    global_node_weights: dict[str, PolicyWeights] = field(default_factory=dict)
    global_memory_routing_bias: float = 0.0
    policy_versions: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    current_version_id: str | None = None
    turn_count: int = 0
    recent_decisions: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    outcome_history: dict[str, list[dict[str, float]]] = field(default_factory=dict)

    def weights_for(self, project: str | None, node_id: str) -> PolicyWeights:
        project_key = project or "__global__"
        project_nodes = self.project_weights.setdefault(project_key, {})
        return project_nodes.setdefault(node_id, PolicyWeights())

    def global_weights_for(self, node_id: str) -> PolicyWeights:
        return self.global_node_weights.setdefault(node_id, PolicyWeights())

    def to_dict(self, *, include_history: bool = True) -> dict[str, Any]:
        payload = {
            "project_weights": {
                project: {node: weights.to_dict() for node, weights in nodes.items()}
                for project, nodes in self.project_weights.items()
            },
            "global_node_weights": {
                node: weights.to_dict() for node, weights in self.global_node_weights.items()
            },
            "global_memory_routing_bias": self.global_memory_routing_bias,
            "current_version_id": self.current_version_id,
            "turn_count": self.turn_count,
            "recent_decisions": self.recent_decisions,
            "outcome_history": self.outcome_history,
        }
        if include_history:
            payload["policy_versions"] = self.policy_versions
            payload["snapshots"] = self.snapshots
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyState:
        state = cls(global_memory_routing_bias=float(data.get("global_memory_routing_bias", 0.0)))
        for project, nodes in dict(data.get("project_weights", {})).items():
            state.project_weights[str(project)] = {
                str(node): PolicyWeights.from_dict(weights) for node, weights in dict(nodes).items()
            }
        state.global_node_weights = {
            str(node): PolicyWeights.from_dict(weights)
            for node, weights in dict(data.get("global_node_weights", {})).items()
        }
        state.policy_versions = list(data.get("policy_versions", []))
        state.snapshots = list(data.get("snapshots", []))
        state.current_version_id = data.get("current_version_id")
        state.turn_count = int(data.get("turn_count", 0))
        state.recent_decisions = {
            str(project): {
                str(node): [str(item) for item in decisions]
                for node, decisions in dict(nodes).items()
            }
            for project, nodes in dict(data.get("recent_decisions", {})).items()
        }
        state.outcome_history = {
            str(project): [
                {str(key): float(value) for key, value in dict(row).items()} for row in rows
            ]
            for project, rows in dict(data.get("outcome_history", {})).items()
        }
        return state


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    policy_name: str
    node_id: str
    preferred_decision: str
    confidence: float
    weight: float
    reason: str
    safe_mode: bool = False
    objective_scores: dict[str, float] = field(default_factory=dict)

    def weighted_score(self) -> float:
        return _clamp_float(self.confidence, 0.0, 1.0) * max(0.0, self.weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "node_id": self.node_id,
            "preferred_decision": self.preferred_decision,
            "confidence": self.confidence,
            "weight": self.weight,
            "weighted_score": self.weighted_score(),
            "reason": self.reason,
            "safe_mode": self.safe_mode,
            "objective_scores": self.objective_scores,
        }


@dataclass(frozen=True, slots=True)
class PolicyArbitrationResult:
    node_id: str
    proposed_decision: str
    final_decision: str
    policy_outputs: list[PolicyDecision]
    decision_scores: dict[str, float]
    conflicts: list[str]
    conflict_trace: list[dict[str, Any]]
    safe_mode: bool
    cost: dict[str, float]
    objective_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "proposed_decision": self.proposed_decision,
            "final_decision": self.final_decision,
            "policy_outputs": [decision.to_dict() for decision in self.policy_outputs],
            "decision_scores": self.decision_scores,
            "conflicts": self.conflicts,
            "conflict_trace": self.conflict_trace,
            "safe_mode": self.safe_mode,
            "cost": self.cost,
            "objective_report": self.objective_report,
        }


class MemoryCostModel:
    def estimate(
        self,
        *,
        node_id: str,
        proposed: str,
        state: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        context = context or {}
        input_tokens = float(estimate_tokens(str(state.get("input") or "")))
        retrieved_count = float(state.get("retrieved_count", 0) or 0)
        selected_count = float(state.get("selected_count", 0) or 0)
        max_tokens = max(1.0, float(state.get("max_tokens") or context.get("max_tokens") or 3500))
        bundle_overhead = 600.0 if proposed == "full" else 160.0 if proposed == "lean" else 40.0
        estimated_injection_tokens = float(
            state.get("injected_tokens") or input_tokens + selected_count * 160.0 + bundle_overhead
        )
        token_cost = (
            input_tokens + retrieved_count * 12.0 + selected_count * 160.0 + bundle_overhead
        )
        retrieval_cost = retrieved_count
        budget_cost = _clamp_float(estimated_injection_tokens / max_tokens, 0.0, 10.0)
        latency_cost = _clamp_float((retrieved_count * 0.015) + (selected_count * 0.02), 0.0, 1.0)
        memory_cost = _clamp_float((retrieved_count + selected_count * 2.0) / 50.0, 0.0, 1.0)
        tradeoff_penalty = _clamp_float(
            (budget_cost * 0.6) + (latency_cost * 0.25) + (memory_cost * 0.15)
        )
        return {
            "token_cost": token_cost,
            "memory_retrieval_cost": retrieval_cost,
            "context_injection_budget_cost": budget_cost,
            "latency_cost": latency_cost,
            "memory_cost": memory_cost,
            "compute_tradeoff_score": _clamp_float(1.0 - tradeoff_penalty),
        }


class RecallPolicy:
    name = "recall"
    weight = 1.25

    def evaluate(
        self,
        *,
        node_id: str,
        proposed: str,
        allowed: list[str],
        state: Mapping[str, Any],
        cost: Mapping[str, float],
        context: Mapping[str, Any],
    ) -> PolicyDecision:
        preferred = proposed
        confidence = 0.7
        reason = "follow base recall decision"
        if node_id == "should_recall" and "recall" in allowed:
            if state.get("decision_action") not in {None, "none"} or state.get(
                "retrieved_count", 0
            ):
                preferred = "recall"
                confidence = 0.82
                reason = "memory signal exists"
        elif node_id == "skip_memory" and "inject" in allowed:
            ranker_confidence = float(state.get("ranker_confidence", 0.0) or 0.0)
            if ranker_confidence >= float(context.get("min_rank_confidence", 0.35)):
                preferred = "inject"
                confidence = _clamp_float(0.55 + ranker_confidence)
                reason = "ranker confidence supports injection"
        return _policy_decision(self.name, node_id, preferred, confidence, self.weight, reason)


class CompressionPolicy:
    name = "compression"
    weight = 0.85

    def evaluate(
        self,
        *,
        node_id: str,
        proposed: str,
        allowed: list[str],
        state: Mapping[str, Any],
        cost: Mapping[str, float],
        context: Mapping[str, Any],
    ) -> PolicyDecision:
        preferred = proposed
        confidence = 0.62
        reason = "compression neutral"
        if node_id == "use_full_bundle" and "lean" in allowed:
            if cost["context_injection_budget_cost"] >= 0.55 or cost["token_cost"] > 1200:
                preferred = "lean"
                confidence = 0.88
                reason = "lean bundle preserves context budget"
        return _policy_decision(
            self.name,
            node_id,
            preferred,
            confidence,
            self.weight,
            reason,
            efficiency=cost["compute_tradeoff_score"],
        )


class LatencyPolicy:
    name = "latency"
    weight = 0.7

    def evaluate(
        self,
        *,
        node_id: str,
        proposed: str,
        allowed: list[str],
        state: Mapping[str, Any],
        cost: Mapping[str, float],
        context: Mapping[str, Any],
    ) -> PolicyDecision:
        preferred = proposed
        confidence = 0.58
        reason = "latency within normal range"
        if node_id == "use_full_bundle" and "lean" in allowed and cost["latency_cost"] > 0.2:
            preferred = "lean"
            confidence = 0.76
            reason = "lean path lowers retrieval latency"
        return _policy_decision(
            self.name,
            node_id,
            preferred,
            confidence,
            self.weight,
            reason,
            efficiency=1.0 - cost["latency_cost"],
        )


class SafetyPolicy:
    name = "safety"
    weight = 1.4

    def evaluate(
        self,
        *,
        node_id: str,
        proposed: str,
        allowed: list[str],
        state: Mapping[str, Any],
        cost: Mapping[str, float],
        context: Mapping[str, Any],
    ) -> PolicyDecision:
        preferred = proposed
        confidence = 0.72
        reason = "no safety risk detected"
        safe_mode = False
        sensitive_requested = bool(context.get("include_sensitive")) and not bool(
            context.get("allow_sensitive")
        )
        if (
            sensitive_requested
            and node_id in {"should_recall", "skip_memory"}
            and "skip" in allowed
        ):
            preferred = "skip"
            confidence = 1.0
            reason = "sensitive recall requires explicit allow_sensitive"
            safe_mode = True
        return _policy_decision(
            self.name,
            node_id,
            preferred,
            confidence,
            self.weight,
            reason,
            safe_mode=safe_mode,
            safety=1.0 if not safe_mode else 0.95,
        )


class CostPolicy:
    name = "cost"
    weight = 0.8

    def evaluate(
        self,
        *,
        node_id: str,
        proposed: str,
        allowed: list[str],
        state: Mapping[str, Any],
        cost: Mapping[str, float],
        context: Mapping[str, Any],
    ) -> PolicyDecision:
        preferred = proposed
        confidence = 0.6
        reason = "cost within budget"
        if node_id == "use_full_bundle" and "lean" in allowed:
            if cost["context_injection_budget_cost"] >= 0.75:
                preferred = "lean"
                confidence = 0.9
                reason = "full bundle exceeds preferred budget cost"
        elif node_id == "skip_memory" and "skip" in allowed and cost["memory_cost"] >= 0.8:
            preferred = "skip"
            confidence = 0.72
            reason = "retrieval cost is too high for this turn"
        return _policy_decision(
            self.name,
            node_id,
            preferred,
            confidence,
            self.weight,
            reason,
            efficiency=cost["compute_tradeoff_score"],
        )


class PolicyArbiter:
    def __init__(
        self,
        policies: list[Any] | None = None,
        cost_model: MemoryCostModel | None = None,
    ) -> None:
        self.policies = policies or [
            RecallPolicy(),
            CompressionPolicy(),
            LatencyPolicy(),
            SafetyPolicy(),
            CostPolicy(),
        ]
        self.cost_model = cost_model or MemoryCostModel()

    def arbitrate(
        self,
        *,
        project: str | None,
        node_id: str,
        proposed: str,
        allowed: list[str],
        state: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
        deterministic: bool = False,
    ) -> PolicyArbitrationResult:
        context = context or {}
        cost = self.cost_model.estimate(
            node_id=node_id, proposed=proposed, state=state, context=context
        )
        outputs = [
            self._evaluate_policy(policy, node_id, proposed, allowed, state, cost, context)
            for policy in self.policies
        ]
        scores = {decision: 0.0 for decision in allowed}
        scores[proposed] = scores.get(proposed, 0.0) + 0.01
        for output in outputs:
            scores[output.preferred_decision] = (
                scores.get(output.preferred_decision, 0.0) + output.weighted_score()
            )
        preferred = {output.preferred_decision for output in outputs}
        conflicts = sorted(preferred) if len(preferred) > 1 else []
        safe_outputs = [
            output
            for output in outputs
            if output.safe_mode and output.preferred_decision in allowed
        ]
        conflict_trace: list[dict[str, Any]] = []
        safe_mode = False
        if deterministic:
            final = proposed
            resolution = "deterministic"
        elif safe_outputs and conflicts:
            final = safe_outputs[0].preferred_decision
            safe_mode = True
            resolution = "safe_mode"
        else:
            final = max(
                allowed, key=lambda decision: (scores.get(decision, 0.0), decision == proposed)
            )
            resolution = "weighted_arbitration" if conflicts else "consensus"
        if conflicts:
            conflict_trace.append(
                {
                    "node_id": node_id,
                    "project": project or "__global__",
                    "proposed_decision": proposed,
                    "conflicting_decisions": conflicts,
                    "resolution": resolution,
                    "final_decision": final,
                    "policy_names": [output.policy_name for output in outputs],
                }
            )
        return PolicyArbitrationResult(
            node_id=node_id,
            proposed_decision=proposed,
            final_decision=final,
            policy_outputs=outputs,
            decision_scores=scores,
            conflicts=conflicts,
            conflict_trace=conflict_trace,
            safe_mode=safe_mode,
            cost=cost,
            objective_report=self._objective_report(outputs, cost, bool(conflicts)),
        )

    def optimization_report(
        self,
        history: list[PolicyArbitrationResult],
    ) -> dict[str, Any]:
        if not history:
            return {"objective": self._objective_text(), "turns": 0, "average_scores": {}}
        metrics = ("recall_accuracy", "execution_efficiency", "stability", "safety")
        averages = {
            metric: sum(float(item.objective_report["scores"].get(metric, 0.0)) for item in history)
            / len(history)
            for metric in metrics
        }
        return {
            "objective": self._objective_text(),
            "turns": len(history),
            "average_scores": averages,
            "conflict_count": sum(1 for item in history if item.conflicts),
            "safe_mode_count": sum(1 for item in history if item.safe_mode),
            "total_token_cost": sum(item.cost["token_cost"] for item in history),
        }

    def _evaluate_policy(
        self,
        policy: Any,
        node_id: str,
        proposed: str,
        allowed: list[str],
        state: Mapping[str, Any],
        cost: Mapping[str, float],
        context: Mapping[str, Any],
    ) -> PolicyDecision:
        decision = policy.evaluate(
            node_id=node_id,
            proposed=proposed,
            allowed=allowed,
            state=state,
            cost=cost,
            context=context,
        )
        if decision.preferred_decision not in allowed:
            return PolicyDecision(
                policy_name=decision.policy_name,
                node_id=node_id,
                preferred_decision=proposed,
                confidence=0.1,
                weight=decision.weight,
                reason="policy returned decision outside allowed set",
                safe_mode=decision.safe_mode,
                objective_scores=decision.objective_scores,
            )
        return decision

    def _objective_report(
        self,
        outputs: list[PolicyDecision],
        cost: Mapping[str, float],
        has_conflict: bool,
    ) -> dict[str, Any]:
        recall_outputs = [item for item in outputs if item.policy_name == "recall"]
        safety_outputs = [item for item in outputs if item.policy_name == "safety"]
        recall_accuracy = recall_outputs[0].confidence if recall_outputs else 0.5
        safety = safety_outputs[0].confidence if safety_outputs else 0.5
        execution_efficiency = cost["compute_tradeoff_score"]
        stability = 0.65 if has_conflict else 1.0
        return {
            "objective": self._objective_text(),
            "scores": {
                "recall_accuracy": _clamp_float(recall_accuracy),
                "execution_efficiency": _clamp_float(execution_efficiency),
                "stability": stability,
                "safety": _clamp_float(safety),
                "latency_minimization": _clamp_float(1.0 - cost["latency_cost"]),
                "memory_cost_minimization": _clamp_float(1.0 - cost["memory_cost"]),
                "policy_drift_minimization": stability,
            },
        }

    def _objective_text(self) -> str:
        return (
            "maximize recall accuracy, execution efficiency, stability, and safety; "
            "minimize latency, memory cost, and policy drift"
        )


class DecisionPolicyEngine:
    def __init__(
        self,
        policy_path: str | Path | None = None,
        state: PolicyState | None = None,
        *,
        drift_threshold: float = 0.25,
        snapshot_interval: int = 10,
        max_update_magnitude: float = 0.1,
        stability_window: int = 3,
        drift_window: int = 4,
    ) -> None:
        self.policy_path = Path(policy_path) if policy_path is not None else None
        self.state = state or self._load_state()
        self.drift_threshold = max(0.0, float(drift_threshold))
        self.snapshot_interval = max(1, int(snapshot_interval))
        self.max_update_magnitude = max(0.0, float(max_update_magnitude))
        self.stability_window = max(1, int(stability_window))
        self.drift_window = max(1, int(drift_window))
        self.guardrail_warnings: list[dict[str, Any]] = []

    @classmethod
    def for_database(cls, db: Database) -> DecisionPolicyEngine:
        return cls(Path(db.path).with_suffix(".policy.json"))

    def _load_state(self) -> PolicyState:
        if self.policy_path is None or not self.policy_path.exists():
            return PolicyState()
        try:
            return PolicyState.from_dict(json.loads(self.policy_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return PolicyState()

    def save(self) -> None:
        if self.policy_path is None:
            return
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_path.write_text(
            json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def select_decision(
        self,
        project: str | None,
        node_id: str,
        proposed: str,
        allowed: list[str],
        *,
        deterministic: bool = False,
    ) -> str:
        if deterministic or proposed not in allowed:
            return proposed
        weights = self.state.weights_for(project, node_id)
        proposed_weight = self.edge_weight(project, node_id, proposed)
        best = max(allowed, key=lambda decision: self.edge_weight(project, node_id, decision))
        best_weight = self.edge_weight(project, node_id, best)
        selected = proposed
        if best != proposed and best_weight >= proposed_weight + 0.15 and weights.visits > 0:
            selected = best
        if node_id == "should_recall":
            recent = self._recent_decisions(project, node_id)
            if recent and selected != recent[-1] and len(recent) < self.stability_window:
                self.guardrail_warnings.append(
                    {
                        "guardrail": "stability_window",
                        "project": project or "__global__",
                        "node_id": node_id,
                        "proposed": selected,
                        "selected": recent[-1],
                    }
                )
                return recent[-1]
        return selected

    def edge_weight(self, project: str | None, node_id: str, decision: str) -> float:
        weights = self.state.weights_for(project, node_id)
        value = weights.edge_weight(decision)
        if node_id == "should_recall":
            if decision == "recall":
                value += self.state.global_memory_routing_bias
            elif decision == "skip":
                value -= self.state.global_memory_routing_bias
        return _clamp_float(value, 0.1, 3.0)

    def policy_snapshot(self, project: str | None, node_id: str, decision: str) -> dict[str, Any]:
        weights = self.state.weights_for(project, node_id)
        return {
            "confidence_score": weights.confidence_score,
            "historical_success_rate": weights.historical_success_rate,
            "adaptive_routing_weight": weights.adaptive_routing_weight,
            "visits": weights.visits,
            "reward_total": weights.reward_total,
            "decision_weight": self.edge_weight(project, node_id, decision),
        }

    def adjust_edge(
        self,
        project: str | None,
        node_id: str,
        decision: str,
        *,
        reward: float,
    ) -> dict[str, float]:
        weights = self.state.weights_for(project, node_id)
        before = weights.edge_weight(decision)
        preserve_node_weight = node_id == "should_recall" and decision == "skip" and reward < 0
        project_adaptive_weight = weights.adaptive_routing_weight
        weights.apply_reward(
            reward,
            decision,
            max_update_magnitude=self.max_update_magnitude,
        )
        if preserve_node_weight:
            weights.adaptive_routing_weight = project_adaptive_weight
        global_weights = self.state.global_weights_for(node_id)
        global_adaptive_weight = global_weights.adaptive_routing_weight
        global_weights.apply_reward(
            reward,
            decision,
            max_update_magnitude=self.max_update_magnitude,
        )
        if preserve_node_weight:
            global_weights.adaptive_routing_weight = global_adaptive_weight
        after = weights.edge_weight(decision)
        return {"before": before, "after": after, "reward": reward}

    def record_decision(self, project: str | None, node_id: str, decision: str) -> None:
        project_key = project or "__global__"
        project_nodes = self.state.recent_decisions.setdefault(project_key, {})
        decisions = project_nodes.setdefault(node_id, [])
        decisions.append(decision)
        max_items = max(self.stability_window, self.drift_window, 4)
        del decisions[:-max_items]

    def _recent_decisions(self, project: str | None, node_id: str) -> list[str]:
        return self.state.recent_decisions.get(project or "__global__", {}).get(node_id, [])

    def update_from_trace(
        self,
        project: str | None,
        trace: list[TurnTraceEvent],
    ) -> list[dict[str, float | str]]:
        signals: list[dict[str, float | str]] = []
        delta_change: dict[str, dict[str, float]] = {}
        for event in trace:
            if event.node_id == "execute" and event.decision == "executed":
                if int(event.output_state.get("selected_count", 0)) > 0:
                    signals.append({"signal": "successful_recall", "reward": 1.0})
                    delta_change[f"{event.node_id}.{event.decision}"] = self.adjust_edge(
                        project, event.node_id, event.decision, reward=1.0
                    )
            elif event.node_id == "fallback_lightweight":
                signals.append({"signal": "fallback_usage", "reward": -0.2})
                delta_change[f"{event.node_id}.{event.decision}"] = self.adjust_edge(
                    project, event.node_id, event.decision, reward=-0.2
                )
            elif event.node_id == "skip_memory" and event.decision in {"skip", "policy_skip"}:
                signals.append({"signal": "skip_memory_correctness", "reward": 0.3})
                delta_change[f"{event.node_id}.{event.decision}"] = self.adjust_edge(
                    project, event.node_id, event.decision, reward=0.3
                )
        self._record_outcome(project, signals)
        self.state.turn_count += 1
        if signals:
            self._record_policy_version(project, delta_change, signals)
        self._maybe_snapshot()
        return signals

    def show_policy_history(self, project: str | None = None) -> list[dict[str, Any]]:
        project_key = project or None
        return [
            self._public_policy_version(version)
            for version in self.state.policy_versions
            if project_key is None or version.get("project") == project_key
        ]

    def explain_policy_change(self, policy_version_id: str) -> dict[str, Any]:
        version = self._find_policy_version(policy_version_id)
        if version is None:
            return {"policy_version_id": policy_version_id, "found": False}
        payload = self._public_policy_version(version)
        payload["found"] = True
        return payload

    def diff_between_policy_versions(
        self,
        from_version: str,
        to_version: str,
    ) -> dict[str, Any]:
        left = self._find_policy_version(from_version)
        right = self._find_policy_version(to_version)
        if left is None or right is None:
            return {
                "from_version": from_version,
                "to_version": to_version,
                "found": False,
                "changed_nodes": [],
                "node_diffs": {},
            }
        left_nodes = self._version_weight_map(left)
        right_nodes = self._version_weight_map(right)
        changed_nodes: set[str] = set()
        node_diffs: dict[str, dict[str, Any]] = {}
        for key in sorted(set(left_nodes) | set(right_nodes)):
            before = left_nodes.get(key)
            after = right_nodes.get(key)
            if before == after:
                continue
            node_id = key.rsplit(".", 1)[-1]
            changed_nodes.add(node_id)
            node_diffs[key] = {"before": before, "after": after}
        return {
            "from_version": from_version,
            "to_version": to_version,
            "found": True,
            "changed_nodes": sorted(changed_nodes),
            "node_diffs": node_diffs,
        }

    def detect_policy_drift(self, project: str | None = None) -> dict[str, Any]:
        project_key = project or "__global__"
        history = self.state.outcome_history.get(project_key, [])
        metrics = ("recall_rate", "fallback_rate", "skip_memory_rate")
        current = self._average_outcomes(history[-self.drift_window :], metrics)
        if len(history) > self.drift_window:
            baseline = self._average_outcomes(
                history[-self.drift_window * 2 : -self.drift_window], metrics
            )
        else:
            baseline = {metric: 0.0 for metric in metrics}
        drift = {metric: current[metric] - baseline[metric] for metric in metrics}
        warnings = {
            metric: value for metric, value in drift.items() if abs(value) >= self.drift_threshold
        }
        return {
            "project": project_key,
            "baseline": baseline,
            "current": current,
            "drift": drift,
            "warnings": warnings,
            "threshold": self.drift_threshold,
        }

    def rollback_to_version(self, policy_version_id: str) -> dict[str, Any]:
        version = self._find_policy_version(policy_version_id)
        state_after = version.get("state_after") if version else None
        if version is None or not isinstance(state_after, Mapping):
            return {"rolled_back": False, "policy_version_id": policy_version_id}
        versions = list(self.state.policy_versions)
        snapshots = list(self.state.snapshots)
        restored = PolicyState.from_dict(state_after)
        restored.policy_versions = versions
        restored.snapshots = snapshots
        restored.current_version_id = policy_version_id
        self.state = restored
        self.save()
        return {
            "rolled_back": True,
            "policy_version_id": policy_version_id,
            "turn_count": self.state.turn_count,
        }

    def _record_outcome(
        self,
        project: str | None,
        signals: list[dict[str, float | str]],
    ) -> None:
        signal_names = {str(signal.get("signal")) for signal in signals}
        row = {
            "recall_rate": 1.0 if "successful_recall" in signal_names else 0.0,
            "fallback_rate": 1.0 if "fallback_usage" in signal_names else 0.0,
            "skip_memory_rate": 1.0 if "skip_memory_correctness" in signal_names else 0.0,
        }
        history = self.state.outcome_history.setdefault(project or "__global__", [])
        history.append(row)
        del history[:-200]

    def _record_policy_version(
        self,
        project: str | None,
        delta_change: dict[str, dict[str, float]],
        reason: list[dict[str, float | str]],
    ) -> None:
        parent_version = self.state.current_version_id
        policy_version_id = self._next_policy_version_id()
        self.state.current_version_id = policy_version_id
        version = {
            "policy_version_id": policy_version_id,
            "parent_version": parent_version,
            "project": project,
            "created_at": utc_now(),
            "turn_count": self.state.turn_count,
            "delta_change": delta_change,
            "reason": [dict(item) for item in reason],
            "state_after": self.state.to_dict(include_history=False),
        }
        self.state.policy_versions.append(version)

    def _maybe_snapshot(self) -> None:
        if self.state.turn_count % self.snapshot_interval != 0:
            return
        self.state.snapshots.append(
            {
                "snapshot_id": f"snapshot-{self.state.turn_count:06d}",
                "policy_version_id": self.state.current_version_id,
                "created_at": utc_now(),
                "turn_count": self.state.turn_count,
                "state": self.state.to_dict(include_history=False),
            }
        )
        del self.state.snapshots[:-50]

    def _next_policy_version_id(self) -> str:
        return f"policy-{len(self.state.policy_versions) + 1:06d}"

    def _find_policy_version(self, policy_version_id: str) -> dict[str, Any] | None:
        for version in self.state.policy_versions:
            if version.get("policy_version_id") == policy_version_id:
                return version
        return None

    def _public_policy_version(self, version: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in version.items() if key != "state_after"}

    def _version_weight_map(self, version: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        state_after = version.get("state_after")
        if not isinstance(state_after, Mapping):
            return {}
        weights: dict[str, dict[str, Any]] = {}
        for project, nodes in dict(state_after.get("project_weights", {})).items():
            for node_id, node_weights in dict(nodes).items():
                weights[f"project.{project}.{node_id}"] = dict(node_weights)
        for node_id, node_weights in dict(state_after.get("global_node_weights", {})).items():
            weights[f"global.{node_id}"] = dict(node_weights)
        return weights

    def _average_outcomes(
        self,
        rows: list[dict[str, float]],
        metrics: tuple[str, ...],
    ) -> dict[str, float]:
        if not rows:
            return {metric: 0.0 for metric in metrics}
        return {
            metric: sum(float(row.get(metric, 0.0)) for row in rows) / len(rows)
            for metric in metrics
        }


class TurnDecisionGraph:
    def __init__(self, nodes: list[DecisionNode], edges: list[DecisionEdge]) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.edges = edges
        self._edge_index = {(edge.source, edge.decision): edge.target for edge in edges}

    @classmethod
    def default(cls) -> TurnDecisionGraph:
        return cls(
            nodes=[
                DecisionNode("start", "Start", "Initialize turn state."),
                DecisionNode("should_recall", "should_recall?", "Decide whether recall is needed."),
                DecisionNode(
                    "use_cache", "use_cache?", "Decide whether cached context can be reused."
                ),
                DecisionNode(
                    "rank_memories", "Rank Memories", "Retrieve and rank candidate memories."
                ),
                DecisionNode(
                    "skip_memory",
                    "skip_memory?",
                    "Decide whether memory injection should be skipped.",
                ),
                DecisionNode(
                    "use_full_bundle", "use_full_bundle?", "Choose full or lean context path."
                ),
                DecisionNode("execute", "Execute", "Run the selected runtime operation."),
                DecisionNode(
                    "fallback_lightweight",
                    "Fallback",
                    "Build lightweight context after failure or low confidence.",
                ),
                DecisionNode("writeback", "Writeback", "Plan or apply memory write-back."),
                DecisionNode("complete", "Complete", "Return final TurnResult."),
            ],
            edges=[
                DecisionEdge("start", "begin", "should_recall"),
                DecisionEdge("should_recall", "recall", "use_cache"),
                DecisionEdge("should_recall", "skip", "skip_memory"),
                DecisionEdge("should_recall", "writeback_only", "execute"),
                DecisionEdge("use_cache", "hit", "execute"),
                DecisionEdge("use_cache", "miss", "rank_memories"),
                DecisionEdge("rank_memories", "ranked", "skip_memory"),
                DecisionEdge("rank_memories", "recall_failed", "fallback_lightweight"),
                DecisionEdge("skip_memory", "inject", "use_full_bundle"),
                DecisionEdge("skip_memory", "skip", "fallback_lightweight"),
                DecisionEdge("skip_memory", "policy_skip", "execute"),
                DecisionEdge("use_full_bundle", "full", "execute"),
                DecisionEdge("use_full_bundle", "lean", "execute"),
                DecisionEdge("execute", "executed", "writeback"),
                DecisionEdge("execute", "recall_failed", "fallback_lightweight"),
                DecisionEdge("fallback_lightweight", "fallback", "writeback"),
                DecisionEdge("writeback", "preview", "complete"),
                DecisionEdge("writeback", "write", "complete"),
                DecisionEdge("writeback", "skipped", "complete"),
                DecisionEdge("complete", "done", None),
            ],
        )

    def next_node(self, node_id: str, decision: str) -> str | None:
        try:
            return self._edge_index[(node_id, decision)]
        except KeyError as exc:
            raise ValueError(f"no decision edge for {node_id!r} -> {decision!r}") from exc


@dataclass(slots=True)
class TurnContext:
    input: str
    retrieved_memories: list[SearchResult] = field(default_factory=list)
    selected_memories: list[SearchResult] = field(default_factory=list)
    context_budget: dict[str, Any] = field(default_factory=dict)
    execution_trace: list[TurnTraceEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "retrieved_memories": [
                _search_result_to_dict(item) for item in self.retrieved_memories
            ],
            "selected_memories": [_search_result_to_dict(item) for item in self.selected_memories],
            "context_budget": self.context_budget,
            "execution_trace": [event.to_dict() for event in self.execution_trace],
        }


@dataclass(slots=True)
class TurnResult:
    injected_context: str
    trace: list[TurnTraceEvent]
    turn_context: TurnContext
    recall_payload: dict[str, Any]
    memory_writeback: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "injected_context": self.injected_context,
            "trace": [event.to_dict() for event in self.trace],
            "turn_context": self.turn_context.to_dict(),
            "recall_payload": self.recall_payload,
            "memory_writeback": self.memory_writeback,
        }

    def runtime_payload(self) -> dict[str, Any]:
        payload = dict(self.recall_payload)
        payload.setdefault("text", self.injected_context)
        payload.setdefault("injected_context", self.injected_context)
        payload["injected_context"] = self.injected_context
        payload["execution_trace"] = [event.to_dict() for event in self.trace]
        payload["retrieved_memories"] = [
            _search_result_to_dict(item) for item in self.turn_context.retrieved_memories
        ]
        payload["selected_memories"] = [
            _search_result_to_dict(item) for item in self.turn_context.selected_memories
        ]
        payload["context_budget"] = self.turn_context.context_budget
        if self.memory_writeback is not None:
            payload["memory_writeback"] = self.memory_writeback
        return payload


class TurnOrchestrator:
    """OS-level turn lifecycle coordinator for runtime memory execution."""

    def __init__(
        self,
        db: Database,
        retriever: Retriever | None = None,
        decision_graph: TurnDecisionGraph | None = None,
        policy_engine: DecisionPolicyEngine | None = None,
        policy_arbiter: PolicyArbiter | None = None,
        memory_scheduler: MemoryScheduler | None = None,
        scheduler_state_path: str | Path | None = None,
        memory_relevance_router: MemoryRelevanceRouter | None = None,
    ) -> None:
        self.db = db
        self.retriever = retriever or Retriever(db)
        self.decision_graph = decision_graph or TurnDecisionGraph.default()
        self.policy_engine = policy_engine or DecisionPolicyEngine.for_database(db)
        self.policy_arbiter = policy_arbiter or PolicyArbiter()
        self.memory_scheduler = memory_scheduler or (
            MemoryScheduler(db, state_path=scheduler_state_path)
            if scheduler_state_path is not None
            else MemoryScheduler.for_database(db)
        )
        self.memory_relevance_router = memory_relevance_router or MemoryRelevanceRouter()
        self._active_policy_project: str | None = None
        self._pending_policy_arbitrations: dict[str, PolicyArbitrationResult] = {}
        self._multi_policy_decision_history: list[PolicyArbitrationResult] = []
        self._suppress_legacy_trace = False

    def run_turn(
        self,
        input: str,
        context: Mapping[str, Any] | None = None,
        mode: TurnMode | str = "preview",
    ) -> TurnResult:
        config = _TurnConfig.from_context(context, mode)
        turn_context = self._new_turn_context(input, config)
        self._active_policy_project = config.project
        self._pending_policy_arbitrations = {}
        self._multi_policy_decision_history = []
        try:
            return self._run_decision_graph(input, config, turn_context)
        finally:
            self._active_policy_project = None
            self._pending_policy_arbitrations = {}
            self._multi_policy_decision_history = []

    def _run_decision_graph(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ) -> TurnResult:
        state: dict[str, Any] = {
            "input": input,
            "operation": config.operation,
            "mode": config.mode,
            "project": config.project,
            "max_tokens": config.max_tokens,
        }
        self._trace_decision(turn_context, "start", "begin", {}, self._trace_state(state))

        decision = self._decide_for_graph(input, config)
        should_recall = _operation_requests_memory(config.operation) or decision.action != "none"
        if config.operation == "memory_auto_store":
            proposed_should_decision = "writeback_only"
            should_recall = False
        else:
            proposed_should_decision = "recall" if should_recall else "skip"
        governed_should_decision = self.policy_engine.select_decision(
            config.project,
            "should_recall",
            proposed_should_decision,
            ["recall", "skip", "writeback_only"],
            deterministic=config.operation != "auto_context",
        )
        state["decision_action"] = decision.action
        should_decision = self._arbitrate_decision(
            config,
            "should_recall",
            governed_should_decision,
            ["recall", "skip", "writeback_only"],
            state,
            deterministic=config.operation != "auto_context",
        )
        should_recall = should_decision == "recall"
        state.update({"recall_decision": decision, "should_recall": should_recall})
        self._trace_decision(
            turn_context,
            "should_recall",
            should_decision,
            {"operation": config.operation},
            self._trace_state(state),
        )

        if should_decision == "recall":
            cache_decision = "hit" if config.cached_context else "miss"
            state["cache_hit"] = bool(config.cached_context)
            self._trace_decision(
                turn_context,
                "use_cache",
                cache_decision,
                self._trace_state(state),
                {"cache_hit": state["cache_hit"]},
            )
            if not config.cached_context:
                rank_decision = self._rank_for_graph(input, config, turn_context, state)
                if rank_decision == "recall_failed":
                    return self._run_graph_fallback(input, config, turn_context, state)
            skip_decision = self._skip_memory_for_graph(config, turn_context, state)
            if skip_decision == "skip":
                return self._run_graph_fallback(input, config, turn_context, state)
            if skip_decision == "inject":
                proposed_bundle_decision = (
                    "full" if _uses_full_bundle(input, config, decision) else "lean"
                )
                governed_bundle_decision = self.policy_engine.select_decision(
                    config.project,
                    "use_full_bundle",
                    proposed_bundle_decision,
                    ["full", "lean"],
                )
                bundle_decision = self._arbitrate_decision(
                    config,
                    "use_full_bundle",
                    governed_bundle_decision,
                    ["full", "lean"],
                    state,
                )
                self._trace_decision(
                    turn_context,
                    "use_full_bundle",
                    bundle_decision,
                    self._trace_state(state),
                    {"use_full_bundle": bundle_decision == "full"},
                )
        else:
            state.update({"skip_memory": True, "skip_reason": "policy_no_recall"})
            self._trace_decision(
                turn_context,
                "skip_memory",
                "policy_skip",
                self._trace_state(state),
                {"skip_memory": True, "skip_reason": "policy_no_recall", "selected_count": 0},
            )

        return self._execute_graph_operation(input, config, turn_context, state)

    def _decide_for_graph(self, input: str, config: _TurnConfig) -> RecallDecision:
        return decide_recall(
            input,
            project=config.project,
            max_tokens=config.max_tokens,
            include_code_map=config.include_code_map,
        )

    def _rank_for_graph(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
        state: dict[str, Any],
    ) -> str:
        try:
            results = self._retrieve(input, config, turn_context)
        except Exception as exc:  # pragma: no cover - defensive fallback boundary
            state.update({"recall_error": str(exc), "retrieved_count": 0})
            self._trace_decision(
                turn_context,
                "rank_memories",
                "recall_failed",
                self._trace_state(state),
                {"recall_error": str(exc), "retrieved_count": 0},
            )
            return "recall_failed"
        confidence = _max_score(results)
        state.update(
            {
                "retrieved_memories": results,
                "retrieved_count": len(results),
                "ranker_confidence": confidence,
            }
        )
        self._trace_decision(
            turn_context,
            "rank_memories",
            "ranked",
            {"operation": config.operation, "should_recall": state.get("should_recall")},
            {
                "retrieved_count": len(results),
                "ranker_confidence": confidence,
                "task_relevance": turn_context.context_budget.get("task_relevance", {}),
            },
        )
        return "ranked"

    def _skip_memory_for_graph(
        self,
        config: _TurnConfig,
        turn_context: TurnContext,
        state: dict[str, Any],
    ) -> str:
        if self._should_skip_low_confidence(config, turn_context.retrieved_memories):
            skip_decision = self._arbitrate_decision(
                config,
                "skip_memory",
                "skip",
                ["inject", "skip"],
                state,
                deterministic=True,
            )
            turn_context.selected_memories = []
            state.update(
                {
                    "skip_memory": True,
                    "skip_reason": "low_rank_confidence",
                    "selected_count": 0,
                }
            )
            self._trace_decision(
                turn_context,
                "skip_memory",
                skip_decision,
                self._trace_state(state),
                {
                    "skip_memory": True,
                    "skip_reason": "low_rank_confidence",
                    "ranker_confidence": state.get("ranker_confidence", 0.0),
                    "selected_count": 0,
                },
            )
            return skip_decision
        turn_context.selected_memories = turn_context.retrieved_memories[: config.selected_k]
        selected_ids = [memory.memory_id for memory in turn_context.selected_memories]
        state.update(
            {
                "skip_memory": False,
                "skip_reason": None,
                "selected_count": len(turn_context.selected_memories),
                "selected_memory_ids": selected_ids,
            }
        )
        inject_decision = self._arbitrate_decision(
            config,
            "skip_memory",
            "inject",
            ["inject", "skip"],
            state,
        )
        if inject_decision == "skip":
            turn_context.selected_memories = []
            state.update(
                {
                    "skip_memory": True,
                    "skip_reason": "multi_policy_arbitration",
                    "selected_count": 0,
                    "selected_memory_ids": [],
                }
            )
            self._trace_decision(
                turn_context,
                "skip_memory",
                "skip",
                self._trace_state(state),
                {
                    "skip_memory": True,
                    "skip_reason": "multi_policy_arbitration",
                    "selected_count": 0,
                },
            )
            return "skip"
        self._trace_decision(
            turn_context,
            "skip_memory",
            inject_decision,
            self._trace_state(state),
            {
                "skip_memory": False,
                "selected_count": len(turn_context.selected_memories),
                "selected_memory_ids": selected_ids,
            },
        )
        return inject_decision

    def _execute_graph_operation(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
        state: dict[str, Any],
    ) -> TurnResult:
        if config.cached_context:
            result = self._finish_with_graph(
                turn_context,
                config,
                injected_context=config.cached_context,
                recall_payload={
                    "decision": _decision_payload(state),
                    "text": config.cached_context,
                    "cache_hit": True,
                },
                decision_action="cached_context",
            )
            self._trace_decision(
                turn_context,
                "execute",
                "executed",
                self._trace_state(state),
                {"cache_hit": True, "selected_count": len(turn_context.selected_memories)},
            )
            self._trace_graph_completion(turn_context, config, result)
            return result

        try:
            self._suppress_legacy_trace = True
            result = self._run_operation_linear(input, config, turn_context)
        except ValueError:
            raise
        except Exception as exc:
            if config.operation != "auto_context":
                raise
            state["recall_error"] = str(exc)
            self._trace_decision(
                turn_context,
                "execute",
                "recall_failed",
                self._trace_state(state),
                {"recall_error": str(exc), "selected_count": len(turn_context.selected_memories)},
            )
            return self._run_graph_fallback(input, config, turn_context, state)
        finally:
            self._suppress_legacy_trace = False

        self._trace_decision(
            turn_context,
            "execute",
            "executed",
            self._trace_state(state),
            {
                "selected_count": len(turn_context.selected_memories),
                "injected_tokens": estimate_tokens(result.injected_context),
            },
        )
        self._trace_graph_completion(turn_context, config, result)
        return result

    def _run_operation_linear(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ) -> TurnResult:
        if config.operation == "memory_search":
            return self._run_memory_search(input, config, turn_context)
        if config.operation == "memory_pack":
            return self._run_memory_pack(input, config, turn_context)
        if config.operation == "context_bundle":
            return self._run_context_bundle(input, config, turn_context)
        if config.operation == "context_callback":
            return self._run_context_callback(input, config, turn_context)
        if config.operation == "memory_auto_store":
            return self._run_memory_auto_store(input, config, turn_context)
        if config.operation == "memory_explain":
            return self._run_memory_explain(input, config, turn_context)
        return self._run_auto_context(input, config, turn_context)

    def _run_graph_fallback(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
        state: dict[str, Any],
    ) -> TurnResult:
        reason = str(state.get("skip_reason") or state.get("recall_error") or "fallback")
        text = _lightweight_context(input, config, reason)
        turn_context.selected_memories = []
        self._trace_decision(
            turn_context,
            "fallback_lightweight",
            "fallback",
            self._trace_state(state),
            {"fallback_reason": reason, "selected_count": 0},
        )
        self._suppress_legacy_trace = True
        try:
            result = self._finish(
                turn_context,
                config,
                injected_context=text,
                recall_payload={
                    "decision": _decision_payload(state),
                    "text": text,
                    "token_count": estimate_tokens(text),
                    "fallback_reason": reason,
                    "included_memory_ids": [],
                    "excluded_memory_ids": [],
                },
                decision_action="fallback_lightweight",
            )
        finally:
            self._suppress_legacy_trace = False
        self._trace_graph_completion(turn_context, config, result)
        return result

    def _finish_with_graph(
        self,
        turn_context: TurnContext,
        config: _TurnConfig,
        *,
        injected_context: str,
        recall_payload: dict[str, Any],
        decision_action: str,
    ) -> TurnResult:
        self._suppress_legacy_trace = True
        try:
            return self._finish(
                turn_context,
                config,
                injected_context=injected_context,
                recall_payload=recall_payload,
                decision_action=decision_action,
            )
        finally:
            self._suppress_legacy_trace = False

    def _run_auto_context(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ) -> TurnResult:
        decision = self._decide(input, config, turn_context)
        retrieved_memories = self._retrieve(input, config, turn_context)
        recall_payload = build_auto_context(
            self.db,
            intent=input,
            project=config.project,
            session_key=config.session_key,
            max_tokens=config.max_tokens,
            include_code_map=config.include_code_map,
            track_token_savings=config.track_token_savings,
            token_model=config.token_model,
            include_savings_in_text=config.include_savings_in_text,
        )
        self._select(
            turn_context,
            retrieved_memories,
            _as_string_list(recall_payload.get("included_memory_ids")),
            config,
        )
        return self._finish(
            turn_context,
            config,
            injected_context=str(recall_payload.get("text") or ""),
            recall_payload=recall_payload,
            decision_action=decision.action,
        )

    def _run_memory_search(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ) -> TurnResult:
        decision = self._decide(input, config, turn_context)
        retrieved_memories = self._retrieve(input, config, turn_context)
        self._select(turn_context, retrieved_memories, [], config)
        payload = {
            "decision": decision.to_dict(),
            "text": _format_search_results(turn_context.selected_memories),
            "results": [_search_result_to_dict(item) for item in turn_context.selected_memories],
        }
        return self._finish(
            turn_context,
            config,
            injected_context=str(payload["text"]),
            recall_payload=payload,
            decision_action=decision.action,
        )

    def _run_memory_pack(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ) -> TurnResult:
        decision = self._decide(input, config, turn_context)
        retrieved_memories = self._retrieve(input, config, turn_context)
        preferred_memory_ids, exclude_memory_ids = self._task_relevance_memory_filters(
            turn_context,
            config,
        )
        packer = MemoryPacker(self.db)
        pack = packer.pack(
            input,
            project=config.project,
            max_tokens=config.max_tokens,
            source_chunk_limit=config.source_chunk_limit,
            compact=config.compact,
            exclude_memory_ids=exclude_memory_ids or None,
            preferred_memory_ids=preferred_memory_ids or None,
            session_dedupe=config.session_dedupe,
        )
        self._select(turn_context, retrieved_memories, packer.last_included_memory_ids, config)
        payload = {
            "decision": decision.to_dict(),
            "text": pack,
            "pack": pack,
            "included_memory_ids": packer.last_included_memory_ids,
        }
        return self._finish(
            turn_context,
            config,
            injected_context=pack,
            recall_payload=payload,
            decision_action=decision.action,
        )

    def _run_context_bundle(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ) -> TurnResult:
        decision = self._decide(input, config, turn_context)
        retrieved_memories = self._retrieve(input, config, turn_context)
        if not config.project:
            raise ValueError("project is required for context_bundle")
        preferred_memory_ids, exclude_memory_ids = self._task_relevance_memory_filters(
            turn_context,
            config,
        )
        bundle = ContextBundleBuilder(self.db).build(
            project=config.project,
            intent=input,
            max_tokens=config.max_tokens,
            include_code_map=config.include_code_map,
            strategy=config.bundle_strategy,
            exclude_memory_ids=exclude_memory_ids or None,
            preferred_memory_ids=preferred_memory_ids or None,
        )
        self._select(turn_context, retrieved_memories, [], config)
        payload = {"decision": decision.to_dict(), "text": bundle, "bundle": bundle}
        return self._finish(
            turn_context,
            config,
            injected_context=bundle,
            recall_payload=payload,
            decision_action=decision.action,
        )

    def _run_context_callback(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ) -> TurnResult:
        decision = self._decide(input, config, turn_context)
        retrieved_memories = self._retrieve(input, config, turn_context)
        if not config.project:
            raise ValueError("project is required for context_callback")
        payload = callback_pack(
            self.db,
            project=config.project,
            intent=input,
            session_key=config.session_key,
            max_tokens=config.max_tokens,
            source_chunk_limit=config.source_chunk_limit,
            compact=config.compact,
        )
        self._select(
            turn_context,
            retrieved_memories,
            _as_string_list(payload.get("included_memory_ids")),
            config,
        )
        return self._finish(
            turn_context,
            config,
            injected_context=str(payload.get("text") or payload.get("pack") or ""),
            recall_payload=dict(payload),
            decision_action=decision.action,
        )

    def _run_memory_auto_store(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ) -> TurnResult:
        decision = self._decide(input, config, turn_context)
        retrieved_memories = self._retrieve(input, config, turn_context)
        self._select(turn_context, retrieved_memories, [], config)
        dry_run = config.dry_run or config.mode != "write" or config.store_mode == "preview"
        memory_writeback = auto_store_memories(
            self.db,
            input,
            project=config.project,
            source=config.source,
            mode=config.store_mode,
            max_candidates=config.max_candidates,
            allow_sensitive=config.allow_sensitive,
            dry_run=dry_run,
        )
        self._trace(
            turn_context,
            "memory.writeback.plan",
            dry_run=memory_writeback["dry_run"],
            written=memory_writeback["written"],
            queued=memory_writeback["queued"],
            skipped=memory_writeback["skipped"],
            duplicates=memory_writeback["duplicates"],
        )
        payload = {"decision": decision.to_dict(), **memory_writeback}
        return self._finish(
            turn_context,
            config,
            injected_context=str(memory_writeback),
            recall_payload=payload,
            decision_action=decision.action,
            memory_writeback=memory_writeback,
            writeback_already_planned=True,
        )

    def _run_memory_explain(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ) -> TurnResult:
        decision = self._decide(input, config, turn_context)
        retrieved_memories = self._retrieve(input, config, turn_context)
        memory_id = config.memory_id or input
        memory = self.db.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"memory not found: {memory_id}")
        explanation = explain_memory_score(
            memory,
            keyword_score=_memory_keyword_score(config.query or input, memory),
            project=config.project or memory.project,
        )
        explained = _memory_to_search_result(memory, explanation.score, explanation.reason)
        turn_context.retrieved_memories = retrieved_memories
        turn_context.selected_memories = [explained]
        self._trace_ranker(turn_context)
        payload = {
            "decision": decision.to_dict(),
            "memory_id": memory.id,
            "project": memory.project,
            "memory_type": memory.memory_type,
            "status": memory.status,
            "visibility": memory.visibility,
            "importance": memory.importance,
            "confidence": memory.confidence,
            "score": explanation.score,
            "why_recalled": explanation.reason,
            "score_details": explanation.factors,
            "ranker_version": RANKER_VERSION,
        }
        return self._finish(
            turn_context,
            config,
            injected_context=str(payload),
            recall_payload=payload,
            decision_action=decision.action,
        )

    def _decide(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ):
        decision = decide_recall(
            input,
            project=config.project,
            max_tokens=config.max_tokens,
            include_code_map=config.include_code_map,
        )
        self._trace(turn_context, "recall.decision", decision=decision.to_dict())
        return decision

    def _retrieve(
        self,
        input: str,
        config: _TurnConfig,
        turn_context: TurnContext,
    ) -> list[SearchResult]:
        results = self.retriever.search(
            input,
            project=config.project,
            memory_types=config.memory_types,
            visibility_scope=config.visibility_scope,
            entities=config.entities,
            tags=config.tags,
            top_k=config.candidate_k,
            include_archived=config.include_archived,
            include_private=config.include_private,
            include_sensitive=config.include_sensitive,
            search_mode=config.search_mode,
            dedupe_results=config.dedupe_results,
        )
        routed = self.memory_relevance_router.rerank(input, results)
        turn_context.context_budget["task_relevance"] = routed.report.to_dict()
        turn_context.retrieved_memories = routed.memories
        return routed.memories

    def _task_relevance_memory_filters(
        self,
        turn_context: TurnContext,
        config: _TurnConfig,
    ) -> tuple[list[str], list[str]]:
        report = turn_context.context_budget.get("task_relevance", {})
        suppressed = []
        if isinstance(report, Mapping):
            suppressed = _as_string_list(report.get("suppressed_memories"))
        excluded = _unique_list([*config.exclude_memory_ids, *suppressed])
        excluded_set = set(excluded)
        preferred = _unique_list(
            [
                memory.memory_id
                for memory in turn_context.retrieved_memories
                if memory.memory_id not in excluded_set
            ]
        )
        return preferred, excluded

    def _select(
        self,
        turn_context: TurnContext,
        retrieved_memories: list[SearchResult],
        selected_memory_ids: list[str],
        config: _TurnConfig,
    ) -> None:
        selected_ids = set(selected_memory_ids)
        suppressed_ids = self._task_relevance_suppressed_ids(turn_context)
        turn_context.retrieved_memories = retrieved_memories
        if selected_ids:
            selected = [
                memory
                for memory in retrieved_memories
                if memory.memory_id in selected_ids and memory.memory_id not in suppressed_ids
            ]
        else:
            selected = [
                memory for memory in retrieved_memories if memory.memory_id not in suppressed_ids
            ][: config.selected_k]
        turn_context.selected_memories = selected
        self._trace_ranker(turn_context)

    def _task_relevance_suppressed_ids(self, turn_context: TurnContext) -> set[str]:
        report = turn_context.context_budget.get("task_relevance", {})
        if not isinstance(report, Mapping):
            return set()
        return set(_as_string_list(report.get("suppressed_memories")))

    def _trace_ranker(self, turn_context: TurnContext) -> None:
        self._trace(
            turn_context,
            "ranker.retrieve",
            retrieved_count=len(turn_context.retrieved_memories),
            selected_count=len(turn_context.selected_memories),
            selected_memory_ids=[memory.memory_id for memory in turn_context.selected_memories],
            score_chain=[
                {
                    "memory_id": memory.memory_id,
                    "score": memory.score,
                    "reason": memory.matched_reason,
                    "score_details": memory.score_details,
                }
                for memory in turn_context.selected_memories
            ],
            task_relevance=turn_context.context_budget.get("task_relevance", {}),
        )

    def _finish(
        self,
        turn_context: TurnContext,
        config: _TurnConfig,
        *,
        injected_context: str,
        recall_payload: dict[str, Any],
        decision_action: str,
        memory_writeback: dict[str, Any] | None = None,
        writeback_already_planned: bool = False,
    ) -> TurnResult:
        turn_context.context_budget.update(
            {
                "operation": config.operation,
                "max_tokens": config.max_tokens,
                "actual_tokens": estimate_tokens(injected_context),
                "candidate_k": config.candidate_k,
                "selected_k": config.selected_k,
                "decision_action": decision_action,
            }
        )
        attach_codegraph_bootstrap_suggestion(
            self.db,
            project=config.project,
            operation=config.operation,
            context_budget=turn_context.context_budget,
            recall_payload=recall_payload,
        )
        self._trace(
            turn_context,
            "context.build",
            token_count=turn_context.context_budget["actual_tokens"],
            budget=turn_context.context_budget,
        )

        if not writeback_already_planned:
            if config.writeback:
                memory_writeback = auto_store_memories(
                    self.db,
                    turn_context.input,
                    project=config.project,
                    source="turn_orchestrator",
                    mode="auto" if config.mode == "write" else "preview",
                    dry_run=config.mode != "write",
                )
                self._trace(
                    turn_context,
                    "memory.writeback.plan",
                    dry_run=memory_writeback["dry_run"],
                    written=memory_writeback["written"],
                    queued=memory_writeback["queued"],
                    skipped=memory_writeback["skipped"],
                    duplicates=memory_writeback["duplicates"],
                )
            else:
                self._trace(turn_context, "memory.writeback.plan", skipped=True)

        self._trace(
            turn_context,
            "turn.complete",
            injected_tokens=turn_context.context_budget["actual_tokens"],
        )
        return TurnResult(
            injected_context=injected_context,
            trace=turn_context.execution_trace,
            turn_context=turn_context,
            recall_payload=recall_payload,
            memory_writeback=memory_writeback,
        )

    def _new_turn_context(self, input: str, config: _TurnConfig) -> TurnContext:
        return TurnContext(
            input=input,
            context_budget={
                "mode": config.mode,
                "operation": config.operation,
                "max_tokens": config.max_tokens,
            },
        )

    def _arbitrate_decision(
        self,
        config: _TurnConfig,
        node_id: str,
        proposed: str,
        allowed: list[str],
        state: Mapping[str, Any],
        *,
        deterministic: bool = False,
    ) -> str:
        result = self.policy_arbiter.arbitrate(
            project=config.project,
            node_id=node_id,
            proposed=proposed,
            allowed=allowed,
            state=state,
            context=self._policy_context(config),
            deterministic=deterministic,
        )
        self._pending_policy_arbitrations[node_id] = result
        self._multi_policy_decision_history.append(result)
        return result.final_decision

    def _policy_context(self, config: _TurnConfig) -> dict[str, Any]:
        return {
            "project": config.project,
            "operation": config.operation,
            "mode": config.mode,
            "max_tokens": config.max_tokens,
            "include_private": config.include_private,
            "include_sensitive": config.include_sensitive,
            "allow_sensitive": config.allow_sensitive,
            "min_rank_confidence": config.min_rank_confidence,
        }

    def _trace_graph_completion(
        self,
        turn_context: TurnContext,
        config: _TurnConfig,
        result: TurnResult,
    ) -> None:
        writeback = result.memory_writeback
        if writeback is None:
            writeback_decision = "skipped"
            writeback_output = {"memory_writeback": None}
        else:
            writeback_decision = "preview" if writeback.get("dry_run") else "write"
            writeback_output = {
                "writeback_dry_run": writeback.get("dry_run"),
                "written": writeback.get("written", 0),
                "queued": writeback.get("queued", 0),
                "skipped": writeback.get("skipped", 0),
                "duplicates": writeback.get("duplicates", 0),
            }
        self._trace_decision(
            turn_context,
            "writeback",
            writeback_decision,
            {"operation": turn_context.context_budget.get("operation")},
            writeback_output,
        )
        self._trace_decision(
            turn_context,
            "complete",
            "done",
            {"operation": turn_context.context_budget.get("operation")},
            {
                "injected_tokens": estimate_tokens(result.injected_context),
                "retrieved_count": len(turn_context.retrieved_memories),
                "selected_count": len(turn_context.selected_memories),
            },
        )
        signals = self.policy_engine.update_from_trace(
            self._active_policy_project,
            turn_context.execution_trace,
        )
        policy_conflict_trace = self._policy_conflict_trace()
        scheduler_report = self.memory_scheduler.run_cycle(
            project=self._active_policy_project,
            trace=turn_context.execution_trace,
            policy_feedback_signals=signals,
            policy_conflict_trace=policy_conflict_trace,
            policy_arbiter=self.policy_arbiter,
            retrieved_memory_ids=[memory.memory_id for memory in turn_context.retrieved_memories],
            selected_memory_ids=[memory.memory_id for memory in turn_context.selected_memories],
            system_load=config.system_load,
        )
        persistence_report = scheduler_report.persistence_report
        if persistence_report.get("event") == "scheduler_state_save_failed":
            turn_context.execution_trace.append(
                TurnTraceEvent(
                    node_id="scheduler",
                    decision="state_save_failed",
                    input_state={"operation": turn_context.context_budget.get("operation")},
                    output_state=persistence_report,
                    next_node=None,
                )
            )
        turn_context.context_budget["multi_policy_decision_history"] = [
            item.to_dict() for item in self._multi_policy_decision_history
        ]
        turn_context.context_budget["system_optimization_report"] = (
            self.policy_arbiter.optimization_report(self._multi_policy_decision_history)
        )
        turn_context.context_budget["memory_scheduler_report"] = scheduler_report.to_dict()
        turn_context.context_budget["policy_feedback_signals"] = signals
        self.policy_engine.save()

    def _policy_conflict_trace(self) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        for result in self._multi_policy_decision_history:
            conflicts.extend(result.conflict_trace)
        return conflicts

    def _trace_decision(
        self,
        turn_context: TurnContext,
        node_id: str,
        decision: str,
        input_state: dict[str, Any],
        output_state: dict[str, Any],
    ) -> None:
        output_state = dict(output_state)
        output_state.setdefault(
            "policy",
            self.policy_engine.policy_snapshot(self._active_policy_project, node_id, decision),
        )
        output_state.setdefault(
            "edge_weight",
            self.policy_engine.edge_weight(self._active_policy_project, node_id, decision),
        )
        arbitration = self._pending_policy_arbitrations.pop(node_id, None)
        if arbitration is not None:
            output_state.setdefault("policy_arbitration", arbitration.to_dict())
        turn_context.execution_trace.append(
            TurnTraceEvent(
                node_id=node_id,
                decision=decision,
                input_state=input_state,
                output_state=output_state,
                next_node=self.decision_graph.next_node(node_id, decision),
            )
        )
        self.policy_engine.record_decision(self._active_policy_project, node_id, decision)

    def _trace_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key, value in state.items():
            if key == "recall_decision" and isinstance(value, RecallDecision):
                compact[key] = value.to_dict()
            elif key in {"retrieved_memories", "selected_memories"} and isinstance(value, list):
                compact[f"{key}_count"] = len(value)
            elif key in {
                "input",
                "operation",
                "mode",
                "project",
                "session_key",
                "should_recall",
                "decision_action",
                "cache_hit",
                "retrieved_count",
                "ranker_confidence",
                "skip_memory",
                "skip_reason",
                "selected_count",
                "selected_memory_ids",
                "use_full_bundle",
                "recall_error",
                "fallback_reason",
                "writeback_dry_run",
                "written",
                "queued",
                "skipped",
                "duplicates",
                "injected_tokens",
            }:
                compact[key] = value
        return compact

    def _should_skip_low_confidence(
        self,
        config: _TurnConfig,
        retrieved_memories: list[SearchResult],
    ) -> bool:
        if config.operation != "auto_context":
            return False
        if not retrieved_memories:
            return False
        return _max_score(retrieved_memories) < config.min_rank_confidence

    def _trace(self, turn_context: TurnContext, stage: str, **payload: Any) -> None:
        if self._suppress_legacy_trace:
            return
        turn_context.execution_trace.append(
            TurnTraceEvent(
                node_id=stage,
                decision="legacy",
                input_state={},
                output_state=payload,
                next_node=None,
            )
        )


@dataclass(frozen=True, slots=True)
class _TurnConfig:
    project: str | None = None
    session_key: str = "default"
    max_tokens: int = 3500
    include_code_map: bool = True
    candidate_k: int = 12
    selected_k: int = 5
    writeback: bool = True
    track_token_savings: bool = False
    token_model: str | None = None
    include_savings_in_text: bool = False
    mode: str = "preview"
    operation: str = "auto_context"
    memory_types: list[str] | None = None
    visibility_scope: list[str] | None = None
    entities: list[str] | None = None
    tags: list[str] | None = None
    include_archived: bool = False
    include_private: bool = False
    include_sensitive: bool = False
    search_mode: str = "hybrid"
    dedupe_results: bool = True
    source_chunk_limit: int = 2
    compact: bool = False
    exclude_memory_ids: list[str] = field(default_factory=list)
    session_dedupe: bool = False
    bundle_strategy: str = "auto"
    source: str = "auto_store"
    store_mode: str = "auto"
    max_candidates: int = 12
    allow_sensitive: bool = False
    dry_run: bool = False
    memory_id: str | None = None
    query: str | None = None
    min_rank_confidence: float = 0.35
    cached_context: str | None = None
    system_load: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context(cls, context: Mapping[str, Any] | None, mode: str) -> _TurnConfig:
        data = dict(context or {})
        selected_k = data.get("selected_k", data.get("top_k", 5))
        candidate_k = data.get("candidate_k", data.get("top_k", 12))
        return cls(
            project=_optional_str(data.get("project")),
            session_key=str(data.get("session_key") or data.get("session") or "default"),
            max_tokens=max(1, int(data.get("max_tokens", 3500))),
            include_code_map=bool(data.get("include_code_map", True)),
            candidate_k=max(1, min(100, int(candidate_k))),
            selected_k=max(1, min(50, int(selected_k))),
            writeback=bool(data.get("writeback", True)),
            track_token_savings=bool(data.get("track_token_savings", False)),
            token_model=_optional_str(data.get("token_model")),
            include_savings_in_text=bool(data.get("include_savings_in_text", False)),
            mode=str(mode or "preview"),
            operation=str(data.get("operation") or "auto_context"),
            memory_types=_string_list_or_none(data.get("memory_types")),
            visibility_scope=_string_list_or_none(data.get("visibility_scope")),
            entities=_string_list_or_none(data.get("entities")),
            tags=_string_list_or_none(data.get("tags")),
            include_archived=bool(data.get("include_archived", False)),
            include_private=bool(data.get("include_private", False)),
            include_sensitive=bool(data.get("include_sensitive", False)),
            search_mode=str(data.get("search_mode") or "hybrid"),
            dedupe_results=bool(data.get("dedupe_results", True)),
            source_chunk_limit=max(0, int(data.get("source_chunk_limit", 2))),
            compact=bool(data.get("compact", False)),
            exclude_memory_ids=_as_string_list(data.get("exclude_memory_ids")),
            session_dedupe=bool(data.get("session_dedupe", False)),
            bundle_strategy=str(data.get("bundle_strategy") or data.get("strategy") or "auto"),
            source=str(data.get("source") or "auto_store"),
            store_mode=str(data.get("store_mode") or data.get("memory_mode") or "auto"),
            max_candidates=max(1, int(data.get("max_candidates", 12))),
            allow_sensitive=bool(data.get("allow_sensitive", False)),
            dry_run=bool(data.get("dry_run", False)),
            memory_id=_optional_str(data.get("memory_id")),
            query=_optional_str(data.get("query")),
            min_rank_confidence=float(data.get("min_rank_confidence", 0.35)),
            cached_context=_optional_str(data.get("cached_context")),
            system_load=dict(data.get("system_load", {}))
            if isinstance(data.get("system_load"), Mapping)
            else {},
        )


def _search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    return asdict(result)


def _operation_requests_memory(operation: str) -> bool:
    return operation in {
        "memory_search",
        "memory_pack",
        "context_bundle",
        "context_callback",
        "memory_explain",
    }


def _uses_full_bundle(input: str, config: _TurnConfig, decision: RecallDecision) -> bool:
    if config.operation == "context_bundle":
        if config.bundle_strategy == "full":
            return True
        if config.bundle_strategy == "lean":
            return False
        return _looks_like_overview(input)
    return decision.action == "context_bundle" and decision.strategy in {"full", "auto:full"}


def _looks_like_overview(input: str) -> bool:
    lowered = input.casefold()
    return any(
        term in lowered
        for term in (
            "project overview",
            "understand project",
            "architecture",
            "onboard",
            "????",
            "????",
        )
    )


def _decision_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    decision = state.get("recall_decision")
    if isinstance(decision, RecallDecision):
        return decision.to_dict()
    return {}


def _lightweight_context(input: str, config: _TurnConfig, reason: str) -> str:
    return "\n".join(
        [
            "Lightweight Context Fallback",
            f"Project: {config.project or 'global'}",
            f"Intent: {input.strip()}",
            f"Reason: {reason}",
            "External memory injection was skipped for this turn.",
        ]
    )


def _max_score(results: list[SearchResult]) -> float:
    if not results:
        return 0.0
    return max(result.score for result in results)


def _policy_decision(
    policy_name: str,
    node_id: str,
    preferred_decision: str,
    confidence: float,
    weight: float,
    reason: str,
    *,
    safe_mode: bool = False,
    recall_accuracy: float | None = None,
    efficiency: float | None = None,
    safety: float | None = None,
) -> PolicyDecision:
    objective_scores = {
        "recall_accuracy": _clamp_float(confidence if recall_accuracy is None else recall_accuracy),
        "execution_efficiency": _clamp_float(confidence if efficiency is None else efficiency),
        "safety": _clamp_float(confidence if safety is None else safety),
    }
    return PolicyDecision(
        policy_name=policy_name,
        node_id=node_id,
        preferred_decision=preferred_decision,
        confidence=_clamp_float(confidence),
        weight=max(0.0, weight),
        reason=reason,
        safe_mode=safe_mode,
        objective_scores=objective_scores,
    )


def _clamp_float(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _memory_to_search_result(memory: MemoryRecord, score: float, reason: str) -> SearchResult:
    return SearchResult(
        memory_id=memory.id,
        content=memory.content,
        summary=memory.summary,
        memory_type=memory.memory_type,
        project=memory.project,
        importance=memory.importance,
        confidence=memory.confidence,
        status=memory.status,
        visibility=memory.visibility,
        score=score,
        matched_reason=reason,
        entities=memory.entities,
        tags=memory.tags,
    )


def _format_search_results(results: list[SearchResult], max_items: int = 8) -> str:
    lines = ["Memory Search Results:"]
    if not results:
        lines.append("1. No matching memory found.")
        return "\n".join(lines)
    seen_texts: set[str] = set()
    shown = 0
    omitted = 0
    for result in results:
        summary = result.summary or result.content
        key = normalize_text(summary).casefold()
        if key in seen_texts:
            omitted += 1
            continue
        seen_texts.add(key)
        shown += 1
        lines.append(
            f"{shown}. [{result.memory_type} score={result.score:.2f}] {_short_text(summary)}"
        )
        if shown >= max_items:
            break
    omitted += max(0, len(results) - shown - omitted)
    if omitted:
        lines.append(f"... {omitted} duplicate/extra result(s) omitted.")
    return "\n".join(lines)


def _short_text(text: str, limit: int = 260) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _memory_keyword_score(query: str, memory: MemoryRecord) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    haystack = " ".join(
        [
            memory.content,
            memory.summary or "",
            " ".join(memory.entities),
            " ".join(memory.tags),
            memory.project or "",
        ]
    )
    memory_tokens = set(tokenize(haystack))
    if not memory_tokens:
        return 0.0
    return min(1.0, len(query_tokens & memory_tokens) / max(1, len(query_tokens)))


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _unique_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _string_list_or_none(value: Any) -> list[str] | None:
    values = _as_string_list(value)
    return values or None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
