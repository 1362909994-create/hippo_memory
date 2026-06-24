from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hippocampus_memory.db import Database
from hippocampus_memory.models import MemoryRecord
from hippocampus_memory.utils import estimate_tokens, utc_now


@dataclass(frozen=True, slots=True)
class MemoryLifecycleAction:
    action_type: str
    memory_id: str
    priority: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "memory_id": self.memory_id,
            "priority": self.priority,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class MemoryTierAssignment:
    memory_id: str
    tier: str
    retrieval_weight: float
    reason: str
    strength: float
    age_days: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "tier": self.tier,
            "retrieval_weight": self.retrieval_weight,
            "reason": self.reason,
            "strength": self.strength,
            "age_days": self.age_days,
        }


@dataclass(frozen=True, slots=True)
class SemanticMemoryProfile:
    memory_id: str
    semantic_type: str
    importance_meaning: str
    context_role: str
    meaning_key: str
    reasoning_utility: float
    decision_improvement: float
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "semantic_type": self.semantic_type,
            "importance_meaning": self.importance_meaning,
            "context_role": self.context_role,
            "meaning_key": self.meaning_key,
            "reasoning_utility": self.reasoning_utility,
            "decision_improvement": self.decision_improvement,
            "risk_flags": self.risk_flags,
        }


class SemanticMemoryModel:
    def profile(self, memory: MemoryRecord) -> SemanticMemoryProfile:
        semantic_type = self.semantic_type(memory)
        importance_meaning = self.importance_meaning(semantic_type)
        context_role = self.context_role(semantic_type)
        meaning_key = self.meaning_key(memory)
        risk_flags = self.risk_flags(memory)
        return SemanticMemoryProfile(
            memory_id=memory.id,
            semantic_type=semantic_type,
            importance_meaning=importance_meaning,
            context_role=context_role,
            meaning_key=meaning_key,
            reasoning_utility=_clamp(
                _memory_strength(memory) + self._semantic_boost(semantic_type)
            ),
            decision_improvement=_clamp(
                memory.importance * (1.0 if semantic_type == "decision" else 0.6)
            ),
            risk_flags=risk_flags,
        )

    def semantic_type(self, memory: MemoryRecord) -> str:
        content = memory.content.casefold()
        if memory.memory_type == "decision" or content.startswith("decision:"):
            return "decision"
        if memory.memory_type == "failure" or any(
            term in content for term in ("error", "failed", "bug")
        ):
            return "error"
        if any(term in content for term in ("insight", "root cause", "learned")):
            return "insight"
        if memory.memory_type in {"constraint", "user_preference"} or any(
            term in content for term in ("always", "prefer", "pattern")
        ):
            return "pattern"
        return "fact"

    def importance_meaning(self, semantic_type: str) -> str:
        meanings = {
            "decision": "guides future implementation choices",
            "error": "prevents repeated failure",
            "insight": "improves reasoning about root causes",
            "pattern": "encodes reusable behavior",
            "fact": "anchors factual context",
        }
        return meanings.get(semantic_type, "anchors factual context")

    def context_role(self, semantic_type: str) -> str:
        roles = {
            "decision": "decision_memory",
            "error": "failure_avoidance",
            "insight": "reasoning_hint",
            "pattern": "behavioral_prior",
            "fact": "context_anchor",
        }
        return roles.get(semantic_type, "context_anchor")

    def meaning_key(self, memory: MemoryRecord) -> str:
        return _semantic_meaning_key(memory.content)

    def risk_flags(self, memory: MemoryRecord) -> list[str]:
        flags: list[str] = []
        if _age_days(memory) >= 120 and memory.importance >= 0.75:
            flags.append("outdated_high_weight")
        if memory.confidence < 0.45:
            flags.append("low_confidence")
        if _is_negative_memory(memory.content):
            flags.append("negative_semantic_claim")
        return flags

    def _semantic_boost(self, semantic_type: str) -> float:
        return {"decision": 0.12, "error": 0.1, "insight": 0.08, "pattern": 0.06}.get(
            semantic_type, 0.02
        )


class SemanticCompressionEngine:
    def compress(
        self,
        memories: list[MemoryRecord],
        *,
        model: SemanticMemoryModel | None = None,
    ) -> dict[str, Any]:
        model = model or SemanticMemoryModel()
        profiles = {memory.id: model.profile(memory) for memory in memories}
        grouped: dict[str, list[MemoryRecord]] = {}
        for memory in memories:
            grouped.setdefault(profiles[memory.id].meaning_key, []).append(memory)
        merged = [
            self._merge_meaning(key, group, profiles)
            for key, group in grouped.items()
            if len(group) > 1
        ]
        summaries = [self._semantic_summary(memory, profiles[memory.id]) for memory in memories]
        return {
            "semantic_summaries": summaries,
            "merged_meanings": merged,
            "semantic_redundancy": _clamp(len(merged) / max(1, len(memories))),
        }

    def _merge_meaning(
        self,
        meaning_key: str,
        memories: list[MemoryRecord],
        profiles: dict[str, SemanticMemoryProfile],
    ) -> dict[str, Any]:
        strongest = max(memories, key=lambda memory: profiles[memory.id].reasoning_utility)
        return {
            "meaning_key": meaning_key,
            "memory_ids": [memory.id for memory in memories],
            "semantic_summary": self._semantic_summary(strongest, profiles[strongest.id]),
            "meaning_preserved": True,
            "dominant_semantic_type": profiles[strongest.id].semantic_type,
        }

    def _semantic_summary(self, memory: MemoryRecord, profile: SemanticMemoryProfile) -> str:
        text = " ".join((memory.summary or memory.content).strip().split())
        if len(text) > 220:
            text = text[:217].rstrip() + "..."
        return f"[{profile.semantic_type}] {text}"


class MemoryCausalityGraph:
    def build(
        self,
        *,
        memories: list[MemoryRecord],
        selected_memory_ids: list[str],
        trace: list[Any],
        decision: str,
        profiles: dict[str, SemanticMemoryProfile],
    ) -> dict[str, Any]:
        selected = set(selected_memory_ids)
        nodes = [
            {"id": memory.id, "kind": "memory", "semantic_type": profiles[memory.id].semantic_type}
            for memory in memories
            if memory.id in selected
        ]
        decision_node = f"decision:{decision}"
        if selected_memory_ids:
            nodes.append({"id": decision_node, "kind": "decision", "trace_count": len(trace)})
        edges = []
        impact_chain = []
        explanations = []
        memory_map = {memory.id: memory for memory in memories}
        for memory_id in selected_memory_ids:
            memory = memory_map.get(memory_id)
            profile = profiles.get(memory_id)
            if memory is None or profile is None:
                continue
            edges.append(
                {
                    "from": memory_id,
                    "to": decision_node,
                    "influence": profile.reasoning_utility,
                    "semantic_type": profile.semantic_type,
                }
            )
            impact_chain.append(
                {
                    "memory_id": memory_id,
                    "decision": decision,
                    "causal_role": profile.context_role,
                    "importance_meaning": profile.importance_meaning,
                }
            )
            explanations.append(
                {
                    "memory_id": memory_id,
                    "why_recalled": (
                        f"{profile.context_role} used to support {decision}; "
                        f"{profile.importance_meaning}."
                    ),
                }
            )
        return {
            "nodes": nodes,
            "edges": edges,
            "impact_chain": impact_chain,
            "explanations": explanations,
        }


class MemoryWorldModel:
    def build(
        self,
        *,
        memories: list[MemoryRecord],
        profiles: dict[str, SemanticMemoryProfile],
    ) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        entities: dict[str, list[str]] = {}
        concepts: dict[str, list[str]] = {}
        decisions: dict[str, dict[str, Any]] = {}
        events: dict[str, dict[str, Any]] = {}
        patterns: dict[str, dict[str, Any]] = {}
        memory_node_map: dict[str, list[str]] = {}
        for memory in memories:
            profile = profiles[memory.id]
            mapped_nodes: list[str] = []
            memory_node = _world_node_id("memory", memory.id)
            nodes[memory_node] = {
                "id": memory_node,
                "kind": "memory",
                "memory_id": memory.id,
                "semantic_type": profile.semantic_type,
                "confidence": memory.confidence,
            }
            mapped_nodes.append(memory_node)
            for entity in memory.entities:
                node_id = _world_node_id("entity", entity)
                nodes.setdefault(node_id, {"id": node_id, "kind": "entity", "label": entity})
                entities.setdefault(entity, []).append(memory.id)
                concepts.setdefault(entity, []).append(memory.id)
                edges.append(
                    _world_edge(memory_node, node_id, "mentions_entity", memory.confidence)
                )
                mapped_nodes.append(node_id)
            concept_terms = [*memory.tags, profile.meaning_key]
            for concept in concept_terms:
                if not concept:
                    continue
                node_id = _world_node_id("concept", concept)
                nodes.setdefault(node_id, {"id": node_id, "kind": "concept", "label": concept})
                concepts.setdefault(concept, []).append(memory.id)
                edges.append(
                    _world_edge(
                        memory_node, node_id, "expresses_concept", profile.reasoning_utility
                    )
                )
                mapped_nodes.append(node_id)
            if profile.semantic_type == "decision":
                decisions[memory.id] = {"memory_id": memory.id, "meaning_key": profile.meaning_key}
            elif profile.semantic_type == "error":
                events[memory.id] = {"memory_id": memory.id, "event_type": "error"}
            elif profile.semantic_type == "pattern":
                patterns[memory.id] = {"memory_id": memory.id, "pattern": profile.meaning_key}
            memory_node_map[memory.id] = mapped_nodes
        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "entities": entities,
            "concepts": concepts,
            "decisions": decisions,
            "events": events,
            "patterns": patterns,
            "memory_node_map": memory_node_map,
            "profiles": {memory_id: profile.to_dict() for memory_id, profile in profiles.items()},
        }

    def cognitive_state(self, graph: Mapping[str, Any]) -> dict[str, Any]:
        profiles = {
            memory_id: profile
            for memory_id, profile in dict(graph.get("profiles", {})).items()
            if isinstance(profile, Mapping)
        }
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        uncertainties: list[float] = []
        belief_state: dict[str, dict[str, Any]] = {}
        for memory_id, profile in profiles.items():
            utility = float(profile.get("reasoning_utility", 0.0))
            if utility >= 0.75:
                confidence_distribution["high"] += 1
            elif utility >= 0.45:
                confidence_distribution["medium"] += 1
            else:
                confidence_distribution["low"] += 1
            uncertainties.append(1.0 - utility)
            belief_state[str(profile.get("meaning_key") or memory_id)] = {
                "memory_id": memory_id,
                "semantic_type": profile.get("semantic_type"),
                "confidence": utility,
            }
        return {
            "belief_state": belief_state,
            "confidence_distribution": confidence_distribution,
            "uncertainty_tracking": {
                "mean_uncertainty": _average(uncertainties),
                "uncertain_belief_count": confidence_distribution["low"],
            },
        }


class SemanticFusionEngine:
    def fuse(self, graph: Mapping[str, Any]) -> dict[str, Any]:
        seen: dict[str, set[str]] = {}
        for concept, memory_ids in dict(graph.get("concepts", {})).items():
            canonical = _canonical_concept(str(concept))
            if not canonical:
                continue
            seen.setdefault(canonical, set()).update(str(memory_id) for memory_id in memory_ids)

        merged = [
            {"concept": concept, "memory_ids": sorted(memory_ids)}
            for concept, memory_ids in seen.items()
            if len(memory_ids) > 1
        ]
        fused_graph = dict(graph)
        fused_graph["concepts"] = {concept: sorted(ids) for concept, ids in seen.items()}
        redundancy_eliminated = sum(max(0, len(item["memory_ids"]) - 1) for item in merged)
        return {
            "merged_concepts": merged,
            "global_semantic_graph": fused_graph,
            "redundancy_eliminated": redundancy_eliminated,
        }


class ReasoningPropagationEngine:
    def propagate(
        self,
        graph: Mapping[str, Any],
        *,
        selected_memory_ids: list[str],
    ) -> dict[str, Any]:
        selected = set(selected_memory_ids)
        semantic_groups = {
            **dict(graph.get("entities", {})),
            **dict(graph.get("concepts", {})),
        }
        concept_to_memories = {
            str(concept): [str(memory_id) for memory_id in memory_ids]
            for concept, memory_ids in semantic_groups.items()
        }
        memory_to_concepts: dict[str, list[str]] = {}
        for concept, memory_ids in concept_to_memories.items():
            for memory_id in memory_ids:
                memory_to_concepts.setdefault(memory_id, []).append(concept)
        chains: list[dict[str, Any]] = []
        ripples: list[dict[str, Any]] = []
        for source in selected:
            for concept in memory_to_concepts.get(source, []):
                for target in concept_to_memories.get(concept, []):
                    if target == source:
                        continue
                    chains.append({"from": source, "to": target, "via": concept})
                    ripples.append(
                        {
                            "source_memory_id": source,
                            "affected_memory_id": target,
                            "concept": concept,
                            "effect": "semantic_ripple",
                        }
                    )
        return {"propagation_chains": chains, "ripple_effects": ripples}


class CognitiveConsistencyEngine:
    def evaluate(
        self,
        graph: Mapping[str, Any],
        propagation: Mapping[str, Any],
    ) -> dict[str, Any]:
        profiles = {
            str(memory_id): dict(profile)
            for memory_id, profile in dict(graph.get("profiles", {})).items()
            if isinstance(profile, Mapping)
        }
        contradictions = self._contradictions(profiles)
        inconsistent = [
            memory_id
            for memory_id, profile in profiles.items()
            if float(profile.get("reasoning_utility", 0.0)) >= 0.45
            and "low_confidence" in list(profile.get("risk_flags", []))
        ]
        unstable = [
            chain
            for chain in list(propagation.get("propagation_chains", []))
            if chain.get("from") in inconsistent
            or chain.get("to") in inconsistent
            or contradictions
        ]
        return {
            "contradictions": contradictions,
            "inconsistent_beliefs": inconsistent,
            "unstable_reasoning_chains": unstable,
            "cognitively_consistent": not contradictions and not inconsistent and not unstable,
        }

    def _contradictions(self, profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        positives: dict[str, str] = {}
        negatives: dict[str, str] = {}
        for memory_id, profile in profiles.items():
            key = str(profile.get("meaning_key") or "")
            if "negative_semantic_claim" in list(profile.get("risk_flags", [])):
                negatives[key] = memory_id
            else:
                positives[key] = memory_id
        return [
            {"meaning_key": key, "memory_ids": [positives[key], negatives[key]]}
            for key in sorted(set(positives) & set(negatives))
        ]


class CognitiveDriveEngine:
    def evaluate(
        self,
        *,
        world_report: Mapping[str, Any],
        memories: list[MemoryRecord] | None = None,
        system_load: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        memories = memories or []
        system_load = system_load or {}
        metrics = self._metrics(world_report, memories)
        goals = self.generate_goals(world_report, metrics)
        attention = self.allocate_attention(world_report, metrics)
        task = self.select_task(goals, attention)
        objective = self.unified_cognitive_objective(metrics)
        flow = self.continuous_cognitive_flow(
            goals=goals,
            attention=attention,
            objective=objective,
            system_load=system_load,
        )
        loops = self.self_triggering_loops(goals, metrics)
        return {
            "generated_goals": goals,
            "attention_allocation": attention,
            "memory_driven_task_selection": task,
            "unified_cognitive_objective": objective,
            "continuous_cognitive_flow": flow,
            "self_triggering_loops": loops,
        }

    def generate_goals(
        self,
        world_report: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        consistency = dict(world_report.get("cognitive_consistency", {}))
        goals: list[dict[str, Any]] = []
        contradictions = list(consistency.get("contradictions", []))
        inconsistent = list(consistency.get("inconsistent_beliefs", []))
        stale = list(metrics.get("stale_memory_ids", []))
        if contradictions:
            goals.append(
                {
                    "goal_id": "resolve_cognitive_conflict",
                    "objective": "resolve contradictory beliefs before future recall",
                    "priority": 1.0,
                    "trigger": "cognitive_conflict",
                    "reason": "unresolved contradiction in world model",
                    "target_memory_ids": _flatten_conflict_memory_ids(contradictions),
                }
            )
        if inconsistent:
            goals.append(
                {
                    "goal_id": "clarify_uncertain_beliefs",
                    "objective": "increase confidence before the belief influences reasoning",
                    "priority": 0.85,
                    "trigger": "uncertainty",
                    "reason": "high-impact low-confidence belief",
                    "target_memory_ids": sorted(str(memory_id) for memory_id in inconsistent),
                }
            )
        if stale:
            goals.append(
                {
                    "goal_id": "reduce_stale_belief_influence",
                    "objective": "decay or refresh outdated high-weight memories",
                    "priority": 0.7,
                    "trigger": "stale_belief",
                    "reason": "outdated high-weight belief remains influential",
                    "target_memory_ids": stale,
                }
            )
        if not goals:
            goals.append(
                {
                    "goal_id": "maintain_reasoning_continuity",
                    "objective": "keep useful memories connected across reasoning cycles",
                    "priority": 0.35,
                    "trigger": "continuity_maintenance",
                    "reason": "no urgent conflict detected",
                    "target_memory_ids": list(metrics.get("salient_memory_ids", [])),
                }
            )
        return sorted(goals, key=lambda goal: float(goal["priority"]), reverse=True)

    def allocate_attention(
        self,
        world_report: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        consistency = dict(world_report.get("cognitive_consistency", {}))
        propagation = dict(world_report.get("reasoning_propagation", {}))
        attention: dict[str, dict[str, Any]] = {}
        for memory_id in _flatten_conflict_memory_ids(list(consistency.get("contradictions", []))):
            attention[memory_id] = _attention_item(
                memory_id,
                1.0,
                "conflict_driven_attention_spike",
            )
        for memory_id in list(consistency.get("inconsistent_beliefs", [])):
            _merge_attention(
                attention,
                _attention_item(str(memory_id), 0.85, "uncertainty_driven_recall_boost"),
            )
        for memory_id in list(metrics.get("stale_memory_ids", [])):
            _merge_attention(
                attention,
                _attention_item(str(memory_id), 0.65, "stale_belief_attention"),
            )
        for ripple in list(propagation.get("ripple_effects", [])):
            memory_id = str(ripple.get("affected_memory_id") or "")
            if memory_id:
                _merge_attention(
                    attention,
                    _attention_item(memory_id, 0.45, "continuous_memory_influence"),
                )
        if not attention:
            for memory_id in list(metrics.get("salient_memory_ids", []))[:3]:
                _merge_attention(attention, _attention_item(memory_id, 0.35, "memory_salience"))
        return sorted(attention.values(), key=lambda item: float(item["score"]), reverse=True)

    def select_task(
        self,
        goals: list[Mapping[str, Any]],
        attention: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not goals:
            return {"selected_task": "idle", "priority": 0.0, "memory_ids": [], "reason": "no_goal"}
        goal = dict(goals[0])
        memory_ids = list(goal.get("target_memory_ids", []))
        if not memory_ids:
            memory_ids = [
                str(item.get("target_id")) for item in attention[:3] if item.get("target_id")
            ]
        return {
            "selected_task": goal["goal_id"],
            "priority": float(goal["priority"]),
            "memory_ids": memory_ids,
            "reason": goal["reason"],
        }

    def unified_cognitive_objective(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        maximize = {
            "cognitive_coherence": _clamp(1.0 - float(metrics["contradiction_density"])),
            "reasoning_continuity": _clamp(float(metrics["continuity_score"])),
            "memory_utility_alignment": _clamp(float(metrics["memory_utility_alignment"])),
        }
        minimize = {
            "contradiction_density": _clamp(float(metrics["contradiction_density"])),
            "stale_belief_influence": _clamp(float(metrics["stale_belief_influence"])),
            "reasoning_fragmentation": _clamp(float(metrics["reasoning_fragmentation"])),
        }
        score = _clamp(_average(list(maximize.values())) - _average(list(minimize.values())))
        return {
            "objective": (
                "maximize cognitive coherence, reasoning continuity, and memory utility alignment; "
                "minimize contradiction density, stale belief influence, "
                "and reasoning fragmentation"
            ),
            "maximize": maximize,
            "minimize": minimize,
            "score": score,
        }

    def continuous_cognitive_flow(
        self,
        *,
        goals: list[Mapping[str, Any]],
        attention: list[Mapping[str, Any]],
        objective: Mapping[str, Any],
        system_load: Mapping[str, Any],
    ) -> dict[str, Any]:
        stream = [
            {
                "event": "cognitive_state_observed",
                "salience_count": len(attention),
                "goal_count": len(goals),
            },
            {
                "event": "attention_allocated",
                "top_target": attention[0]["target_id"] if attention else None,
            },
            {
                "event": "objective_updated",
                "score": objective.get("score", 0.0),
            },
        ]
        if goals:
            stream.append(
                {
                    "event": "rolling_reasoning_update",
                    "active_goal": goals[0].get("goal_id"),
                    "deferred_under_load": _is_high_load(system_load),
                }
            )
        return {
            "model": "rolling_state_stream",
            "stream": stream,
            "continuous_memory_influence": bool(attention),
        }

    def self_triggering_loops(
        self,
        goals: list[Mapping[str, Any]],
        metrics: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        loops: list[dict[str, Any]] = []
        goal_ids = {str(goal.get("goal_id")) for goal in goals}
        if "resolve_cognitive_conflict" in goal_ids:
            loops.append(
                {
                    "loop_type": "consistency_resolution",
                    "trigger": "cognitive_conflict",
                    "action": "review_conflicting_memories",
                    "priority": 1.0,
                    "deterministic": True,
                }
            )
        if "clarify_uncertain_beliefs" in goal_ids:
            loops.append(
                {
                    "loop_type": "uncertainty_review",
                    "trigger": "uncertain_belief",
                    "action": "boost_recall_for_low_confidence_memories",
                    "priority": 0.8,
                    "deterministic": True,
                }
            )
        if float(metrics.get("semantic_redundancy", 0.0)) > 0.0:
            loops.append(
                {
                    "loop_type": "consolidation_cycle",
                    "trigger": "semantic_redundancy",
                    "action": "schedule_memory_consolidation",
                    "priority": 0.65,
                    "deterministic": True,
                }
            )
        return loops

    def _metrics(
        self,
        world_report: Mapping[str, Any],
        memories: list[MemoryRecord],
    ) -> dict[str, Any]:
        graph = dict(world_report.get("graph", {}))
        profiles = {
            str(memory_id): dict(profile)
            for memory_id, profile in dict(graph.get("profiles", {})).items()
            if isinstance(profile, Mapping)
        }
        consistency = dict(world_report.get("cognitive_consistency", {}))
        propagation = dict(world_report.get("reasoning_propagation", {}))
        fusion = dict(world_report.get("semantic_fusion", {}))
        memory_count = max(1, len(memories) or len(profiles))
        contradictions = list(consistency.get("contradictions", []))
        stale = [
            memory_id
            for memory_id, profile in profiles.items()
            if "outdated_high_weight" in list(profile.get("risk_flags", []))
        ]
        utilities = [float(profile.get("reasoning_utility", 0.0)) for profile in profiles.values()]
        chains = list(propagation.get("propagation_chains", []))
        salient = sorted(
            profiles,
            key=lambda memory_id: float(profiles[memory_id].get("reasoning_utility", 0.0)),
            reverse=True,
        )
        return {
            "contradiction_density": len(contradictions) / memory_count,
            "stale_belief_influence": len(stale) / memory_count,
            "reasoning_fragmentation": 1.0 / (1.0 + len(chains)),
            "continuity_score": len(chains) / (1.0 + len(chains)),
            "memory_utility_alignment": _average(utilities),
            "semantic_redundancy": float(fusion.get("redundancy_eliminated", 0.0)) / memory_count,
            "stale_memory_ids": stale,
            "salient_memory_ids": salient,
        }


class MemoryHierarchy:
    tiers = {
        "L0": "working_memory",
        "L1": "short_term_memory",
        "L2": "long_term_memory",
        "L3": "archival_memory",
    }
    tier_weights = {"L0": 1.4, "L1": 1.0, "L2": 1.15, "L3": 0.35}

    def assign_tiers(
        self,
        memories: list[MemoryRecord],
        *,
        working_memory_ids: list[str] | None = None,
    ) -> dict[str, MemoryTierAssignment]:
        working = set(working_memory_ids or [])
        assignments: dict[str, MemoryTierAssignment] = {}
        for memory in memories:
            age_days = _age_days(memory)
            strength = _memory_strength(memory)
            tier, reason = self._tier_for(memory, age_days, strength, memory.id in working)
            assignments[memory.id] = MemoryTierAssignment(
                memory_id=memory.id,
                tier=tier,
                retrieval_weight=self.retrieval_weight(tier, strength),
                reason=reason,
                strength=strength,
                age_days=age_days,
            )
        return assignments

    def retrieval_weight(self, tier: str, strength: float) -> float:
        return _clamp(self.tier_weights.get(tier, 1.0) * (0.5 + strength), 0.05, 2.0)

    def promotion_target(self, assignment: MemoryTierAssignment) -> str:
        if assignment.tier == "L0":
            return "L1"
        if assignment.tier == "L1" and assignment.strength >= 0.75:
            return "L2"
        return assignment.tier

    def demotion_target(self, assignment: MemoryTierAssignment) -> str:
        if assignment.tier == "L2" and assignment.strength < 0.35:
            return "L1"
        if assignment.tier == "L1" and assignment.age_days >= 90 and assignment.strength < 0.35:
            return "L3"
        return assignment.tier

    def _tier_for(
        self,
        memory: MemoryRecord,
        age_days: int,
        strength: float,
        is_working_memory: bool,
    ) -> tuple[str, str]:
        if is_working_memory:
            return "L0", "selected in current turn"
        if strength >= 0.78 and memory.usage_count >= 5:
            return "L2", "stable and repeatedly useful"
        if age_days >= 120 or (strength < 0.25 and age_days >= 60):
            return "L3", "stale or low-strength memory"
        return "L1", "recent or not yet globally stable"


class GlobalConsistencyModel:
    def evaluate(
        self,
        memories: list[MemoryRecord],
        tier_assignments: dict[str, MemoryTierAssignment],
    ) -> dict[str, Any]:
        uniqueness = self._uniqueness_violations(memories, tier_assignments)
        contradictions = self._contradictions(memories, tier_assignments)
        return {
            "commit_allowed": not uniqueness and not contradictions,
            "uniqueness_violations": uniqueness,
            "contradictions": contradictions,
            "invariants": {
                "memory_uniqueness": not uniqueness,
                "no_cross_tier_contradictions": not contradictions,
                "consistency_enforced_before_commit": True,
            },
        }

    def _uniqueness_violations(
        self,
        memories: list[MemoryRecord],
        tier_assignments: dict[str, MemoryTierAssignment],
    ) -> list[dict[str, Any]]:
        buckets: dict[str, list[MemoryRecord]] = {}
        for memory in memories:
            key = _normalized_memory_key(memory.content)
            buckets.setdefault(key, []).append(memory)
        violations: list[dict[str, Any]] = []
        for key, grouped in buckets.items():
            if len(grouped) < 2:
                continue
            violations.append(
                {
                    "key": key,
                    "memory_ids": [memory.id for memory in grouped],
                    "tiers": [tier_assignments[memory.id].tier for memory in grouped],
                }
            )
        return violations

    def _contradictions(
        self,
        memories: list[MemoryRecord],
        tier_assignments: dict[str, MemoryTierAssignment],
    ) -> list[dict[str, Any]]:
        positives: dict[str, MemoryRecord] = {}
        negatives: dict[str, MemoryRecord] = {}
        for memory in memories:
            key = _contradiction_key(memory.content)
            if not key:
                continue
            if _is_negative_memory(memory.content):
                negatives[key] = memory
            else:
                positives[key] = memory
        contradictions: list[dict[str, Any]] = []
        for key in sorted(set(positives) & set(negatives)):
            left = positives[key]
            right = negatives[key]
            contradictions.append(
                {
                    "key": key,
                    "memory_ids": [left.id, right.id],
                    "tiers": [tier_assignments[left.id].tier, tier_assignments[right.id].tier],
                    "reason": "positive and negative statements target the same fact",
                }
            )
        return contradictions


class SystemStabilityController:
    def __init__(self, convergence_window: int = 3) -> None:
        self.convergence_window = max(1, convergence_window)
        self._last_transition: dict[str, dict[str, Any]] = {}

    def allow_transition(
        self,
        memory_id: str,
        from_tier: str,
        to_tier: str,
        *,
        turn: int,
    ) -> dict[str, Any]:
        previous = self._last_transition.get(memory_id)
        if previous and previous["to_tier"] == from_tier and previous["from_tier"] == to_tier:
            elapsed = turn - int(previous["turn"])
            if elapsed < self.convergence_window:
                return {
                    "allowed": False,
                    "reason": "convergence_window",
                    "last_transition": previous,
                }
        transition = {
            "memory_id": memory_id,
            "from_tier": from_tier,
            "to_tier": to_tier,
            "turn": turn,
        }
        self._last_transition[memory_id] = transition
        return {"allowed": True, "reason": "stable", "last_transition": transition}


@dataclass(slots=True)
class SchedulerState:
    turn_count: int = 0
    shared_reward_signal: float = 0.0
    policy_sync_weights: dict[str, float] = field(default_factory=dict)
    conflict_resolution_bias: dict[str, float] = field(default_factory=dict)
    system_parameters: dict[str, float] = field(default_factory=dict)
    lifecycle_history: list[dict[str, Any]] = field(default_factory=list)
    last_state_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_count": self.turn_count,
            "shared_reward_signal": self.shared_reward_signal,
            "policy_sync_weights": self.policy_sync_weights,
            "conflict_resolution_bias": self.conflict_resolution_bias,
            "system_parameters": self.system_parameters,
            "lifecycle_history": self.lifecycle_history,
            "last_state_version": self.last_state_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SchedulerState:
        return cls(
            turn_count=int(data.get("turn_count", 0)),
            shared_reward_signal=float(data.get("shared_reward_signal", 0.0)),
            policy_sync_weights={
                str(key): float(value)
                for key, value in dict(data.get("policy_sync_weights", {})).items()
            },
            conflict_resolution_bias={
                str(key): float(value)
                for key, value in dict(data.get("conflict_resolution_bias", {})).items()
            },
            system_parameters={
                str(key): float(value)
                for key, value in dict(data.get("system_parameters", {})).items()
            },
            lifecycle_history=list(data.get("lifecycle_history", [])),
            last_state_version=data.get("last_state_version"),
        )


@dataclass(frozen=True, slots=True)
class SchedulerReport:
    state_version: str
    lifecycle_actions: list[MemoryLifecycleAction]
    global_objective: dict[str, Any]
    shared_reward_signal: float
    policy_alignment: dict[str, Any]
    optimization_loop: dict[str, Any]
    hierarchy_report: dict[str, Any] = field(default_factory=dict)
    consistency_report: dict[str, Any] = field(default_factory=dict)
    stability_report: dict[str, Any] = field(default_factory=dict)
    load_report: dict[str, Any] = field(default_factory=dict)
    semantic_report: dict[str, Any] = field(default_factory=dict)
    persistence_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "lifecycle_actions": [action.to_dict() for action in self.lifecycle_actions],
            "global_objective": self.global_objective,
            "shared_reward_signal": self.shared_reward_signal,
            "policy_alignment": self.policy_alignment,
            "optimization_loop": self.optimization_loop,
            "hierarchy_report": self.hierarchy_report,
            "consistency_report": self.consistency_report,
            "stability_report": self.stability_report,
            "load_report": self.load_report,
            "semantic_report": self.semantic_report,
            "persistence_report": self.persistence_report,
        }


class MemoryScheduler:
    """Cross-turn scheduler for memory lifecycle and policy alignment planning."""

    def __init__(
        self,
        db: Database,
        state_path: str | Path | None = None,
        state: SchedulerState | None = None,
        hierarchy: MemoryHierarchy | None = None,
        consistency_model: GlobalConsistencyModel | None = None,
        stability_controller: SystemStabilityController | None = None,
        semantic_model: SemanticMemoryModel | None = None,
        semantic_compression: SemanticCompressionEngine | None = None,
        causality_graph: MemoryCausalityGraph | None = None,
        world_model: MemoryWorldModel | None = None,
        semantic_fusion: SemanticFusionEngine | None = None,
        reasoning_propagation: ReasoningPropagationEngine | None = None,
        cognitive_consistency: CognitiveConsistencyEngine | None = None,
        cognitive_drive: CognitiveDriveEngine | None = None,
    ) -> None:
        self.db = db
        self.state_path = (
            Path(state_path)
            if state_path is not None
            else Path(db.path).with_suffix(".scheduler.json")
        )
        self.state = state or self._load_state()
        self.hierarchy = hierarchy or MemoryHierarchy()
        self.consistency_model = consistency_model or GlobalConsistencyModel()
        self.stability_controller = stability_controller or SystemStabilityController()
        self.semantic_model = semantic_model or SemanticMemoryModel()
        self.semantic_compression = semantic_compression or SemanticCompressionEngine()
        self.causality_graph = causality_graph or MemoryCausalityGraph()
        self.world_model = world_model or MemoryWorldModel()
        self.semantic_fusion = semantic_fusion or SemanticFusionEngine()
        self.reasoning_propagation = reasoning_propagation or ReasoningPropagationEngine()
        self.cognitive_consistency = cognitive_consistency or CognitiveConsistencyEngine()
        self.cognitive_drive = cognitive_drive or CognitiveDriveEngine()

    @classmethod
    def for_database(cls, db: Database) -> MemoryScheduler:
        return cls(db)

    def run_cycle(
        self,
        *,
        project: str | None,
        trace: list[Any] | None = None,
        policy_feedback_signals: list[Mapping[str, Any]] | None = None,
        policy_conflict_trace: list[Mapping[str, Any]] | None = None,
        policy_arbiter: Any | None = None,
        retrieved_memory_ids: list[str] | None = None,
        selected_memory_ids: list[str] | None = None,
        system_load: Mapping[str, Any] | None = None,
    ) -> SchedulerReport:
        trace = trace or []
        policy_feedback_signals = policy_feedback_signals or []
        policy_conflict_trace = policy_conflict_trace or []
        retrieved_memory_ids = retrieved_memory_ids or []
        selected_memory_ids = selected_memory_ids or []
        system_load = system_load or {}
        memories = self._list_project_memories(project)
        tier_assignments = self.hierarchy.assign_tiers(
            memories,
            working_memory_ids=selected_memory_ids,
        )
        consistency_report = self.consistency_model.evaluate(memories, tier_assignments)
        load_report = self._load_profile(system_load, selected_memory_ids)
        lifecycle_actions, stability_report = self.schedule_lifecycle(
            project=project,
            memories=memories,
            tier_assignments=tier_assignments,
            system_load=system_load,
        )
        semantic_report = self.semantic_report(
            memories=memories,
            selected_memory_ids=selected_memory_ids,
            trace=trace,
            system_load=system_load,
        )
        conflict_frequency = self._conflict_frequency(policy_conflict_trace)
        shared_reward = self._shared_reward(policy_feedback_signals, conflict_frequency)
        self.state.turn_count += 1
        self.state.shared_reward_signal = shared_reward
        state_version = f"scheduler-{self.state.turn_count:06d}"
        self.state.last_state_version = state_version
        global_objective = self.global_objective(
            memories=memories,
            lifecycle_actions=lifecycle_actions,
            policy_feedback_signals=policy_feedback_signals,
            policy_conflict_frequency=conflict_frequency,
            retrieved_memory_ids=retrieved_memory_ids,
            selected_memory_ids=selected_memory_ids,
            trace=trace,
        )
        policy_alignment = self.synchronize_policies(
            policy_arbiter=policy_arbiter,
            shared_reward=shared_reward,
            conflict_trace=policy_conflict_trace,
        )
        optimization_loop = self.run_background_optimization(
            memories=memories,
            lifecycle_actions=lifecycle_actions,
            policy_conflict_frequency=conflict_frequency,
            system_load=system_load,
        )
        self.state.lifecycle_history.append(
            {
                "state_version": state_version,
                "project": project,
                "created_at": utc_now(),
                "action_count": len(lifecycle_actions),
                "shared_reward_signal": shared_reward,
                "policy_conflict_frequency": conflict_frequency,
            }
        )
        del self.state.lifecycle_history[:-200]
        report = SchedulerReport(
            state_version=state_version,
            lifecycle_actions=lifecycle_actions,
            global_objective=global_objective,
            shared_reward_signal=shared_reward,
            policy_alignment=policy_alignment,
            optimization_loop=optimization_loop,
            hierarchy_report=self._hierarchy_report(tier_assignments),
            consistency_report=consistency_report,
            stability_report=stability_report,
            load_report=load_report,
            semantic_report=semantic_report,
            persistence_report=self.save(),
        )
        return report

    def schedule_lifecycle(
        self,
        *,
        project: str | None,
        memories: list[MemoryRecord] | None = None,
        tier_assignments: dict[str, MemoryTierAssignment] | None = None,
        system_load: Mapping[str, Any] | None = None,
    ) -> tuple[list[MemoryLifecycleAction], dict[str, Any]]:
        memories = memories if memories is not None else self._list_project_memories(project)
        tier_assignments = tier_assignments or self.hierarchy.assign_tiers(memories)
        system_load = system_load or {}
        high_load = _is_high_load(system_load)
        low_confidence_state = bool(system_load.get("low_confidence_state", False))
        actions: list[MemoryLifecycleAction] = []
        stability_events: list[dict[str, Any]] = []
        for memory in memories:
            age_days = _age_days(memory)
            strength = _memory_strength(memory)
            token_count = estimate_tokens(memory.summary or memory.content)
            assignment = tier_assignments[memory.id]
            if age_days >= 30 and memory.usage_count == 0:
                actions.append(
                    MemoryLifecycleAction(
                        "decay",
                        memory.id,
                        priority=_clamp((age_days / 180.0) + (1.0 - strength) * 0.4),
                        reason="memory has aged without reuse",
                        metadata={"age_days": age_days, "strength": strength},
                    )
                )
            if strength <= 0.25 and age_days >= 60:
                deferred = low_confidence_state
                actions.append(
                    MemoryLifecycleAction(
                        "evict",
                        memory.id,
                        priority=_clamp((1.0 - strength) * (0.2 if deferred else 1.0)),
                        reason="low-strength stale memory should be reviewed for eviction",
                        metadata={
                            "hard_delete": False,
                            "age_days": age_days,
                            "source_tier": assignment.tier,
                            "deferred": deferred,
                        },
                    )
                )
            if memory.usage_count >= 5 and memory.importance >= 0.75 and memory.confidence >= 0.75:
                transition = self.stability_controller.allow_transition(
                    memory.id,
                    assignment.tier,
                    "L2",
                    turn=self.state.turn_count + 1,
                )
                stability_events.append(transition)
                if not transition["allowed"]:
                    continue
                actions.append(
                    MemoryLifecycleAction(
                        "promote",
                        memory.id,
                        priority=_clamp(strength),
                        reason="high-confidence memory is repeatedly useful",
                        metadata={
                            "target_scope": "long_term",
                            "source_tier": assignment.tier,
                            "target_tier": "L2",
                            "usage_count": memory.usage_count,
                        },
                    )
                )
            if token_count >= 180:
                deferred = high_load
                actions.append(
                    MemoryLifecycleAction(
                        "compress",
                        memory.id,
                        priority=_clamp((token_count / 1200.0) * (0.35 if deferred else 1.0)),
                        reason="memory is large enough to benefit from semantic compression",
                        metadata={
                            "token_estimate": token_count,
                            "source_tier": assignment.tier,
                            "deferred": deferred,
                        },
                    )
                )
            if memory.memory_type == "task_state" and memory.usage_count == 0 and age_days >= 14:
                transition = self.stability_controller.allow_transition(
                    memory.id,
                    assignment.tier,
                    "L1",
                    turn=self.state.turn_count + 1,
                )
                stability_events.append(transition)
                if not transition["allowed"]:
                    continue
                actions.append(
                    MemoryLifecycleAction(
                        "demote",
                        memory.id,
                        priority=_clamp(age_days / 90.0),
                        reason="unused task-state memory should remain short-term",
                        metadata={
                            "target_scope": "short_term",
                            "source_tier": assignment.tier,
                            "target_tier": "L1",
                            "age_days": age_days,
                        },
                    )
                )
        actions.sort(key=lambda action: (action.priority, action.action_type), reverse=True)
        return actions, {
            "convergence_window": self.stability_controller.convergence_window,
            "transition_events": stability_events,
            "blocked_transitions": [event for event in stability_events if not event["allowed"]],
        }

    def global_objective(
        self,
        *,
        memories: list[MemoryRecord],
        lifecycle_actions: list[MemoryLifecycleAction],
        policy_feedback_signals: list[Mapping[str, Any]],
        policy_conflict_frequency: float,
        retrieved_memory_ids: list[str],
        selected_memory_ids: list[str],
        trace: list[Any],
    ) -> dict[str, Any]:
        usefulness = _average([_memory_strength(memory) for memory in memories])
        recall_accuracy = _recall_accuracy(policy_feedback_signals, retrieved_memory_ids)
        task_success = _task_success_rate(policy_feedback_signals)
        cost = _clamp((len(memories) / 200.0) + (len(lifecycle_actions) / 100.0))
        latency = _clamp(len(retrieved_memory_ids) / 50.0)
        redundancy = self._redundancy_rate(memories)
        return {
            "objective": (
                "maximize memory usefulness, recall accuracy, and task success rate; "
                "minimize cost, latency, redundancy, and policy conflict frequency"
            ),
            "maximize": {
                "memory_usefulness": usefulness,
                "recall_accuracy": recall_accuracy,
                "task_success_rate": task_success,
            },
            "minimize": {
                "cost": cost,
                "latency": latency,
                "redundancy": redundancy,
                "policy_conflict_frequency": policy_conflict_frequency,
            },
            "trace_count": len(trace),
            "selected_memory_count": len(selected_memory_ids),
        }

    def semantic_report(
        self,
        *,
        memories: list[MemoryRecord],
        selected_memory_ids: list[str],
        trace: list[Any],
        system_load: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profiles = {memory.id: self.semantic_model.profile(memory) for memory in memories}
        compression = self.semantic_compression.compress(memories, model=self.semantic_model)
        decision = _trace_decision_label(trace)
        causality = self.causality_graph.build(
            memories=memories,
            selected_memory_ids=selected_memory_ids,
            trace=trace,
            decision=decision,
            profiles=profiles,
        )
        consistency = self.semantic_consistency(memories, profiles)
        world_graph = self.world_model.build(memories=memories, profiles=profiles)
        fusion = self.semantic_fusion.fuse(world_graph)
        propagation = self.reasoning_propagation.propagate(
            fusion["global_semantic_graph"],
            selected_memory_ids=selected_memory_ids,
        )
        cognitive_consistency = self.cognitive_consistency.evaluate(
            fusion["global_semantic_graph"],
            propagation,
        )
        cognitive_state = self.world_model.cognitive_state(fusion["global_semantic_graph"])
        world_report = {
            "graph": world_graph,
            "semantic_fusion": fusion,
            "cognitive_consistency": cognitive_consistency,
            "reasoning_propagation": propagation,
            "global_cognitive_state": cognitive_state,
        }
        cognitive_drive = self.cognitive_drive.evaluate(
            world_report=world_report,
            memories=memories,
            system_load=system_load,
        )
        return {
            "profiles": {memory_id: profile.to_dict() for memory_id, profile in profiles.items()},
            "semantic_compression": compression,
            "global_semantic_objective": self.global_semantic_objective(
                profiles=profiles,
                compression=compression,
                consistency=consistency,
            ),
            "causality_graph": causality,
            "meaning_consistency": consistency,
            "world_model": world_report,
            "cognitive_drive": cognitive_drive,
        }

    def global_semantic_objective(
        self,
        *,
        profiles: dict[str, SemanticMemoryProfile],
        compression: Mapping[str, Any],
        consistency: Mapping[str, Any],
    ) -> dict[str, Any]:
        reasoning_utility = _average([profile.reasoning_utility for profile in profiles.values()])
        decision_improvement = _average(
            [profile.decision_improvement for profile in profiles.values()]
        )
        long_term_task_performance = _average(
            [
                profile.reasoning_utility
                for profile in profiles.values()
                if profile.semantic_type != "fact"
            ]
        )
        misleading = len(consistency.get("misleading_memory_influence", [])) / max(1, len(profiles))
        outdated = len(consistency.get("outdated_high_weight_memories", [])) / max(1, len(profiles))
        return {
            "objective": (
                "maximize reasoning utility, decision improvement, and long-term task performance; "
                "minimize semantic redundancy, misleading memory influence, "
                "and outdated reasoning patterns"
            ),
            "maximize": {
                "reasoning_utility": reasoning_utility,
                "decision_improvement": decision_improvement,
                "long_term_task_performance": long_term_task_performance,
            },
            "minimize": {
                "semantic_redundancy": float(compression.get("semantic_redundancy", 0.0)),
                "misleading_memory_influence": _clamp(misleading),
                "outdated_reasoning_patterns": _clamp(outdated),
            },
        }

    def semantic_consistency(
        self,
        memories: list[MemoryRecord],
        profiles: dict[str, SemanticMemoryProfile],
    ) -> dict[str, Any]:
        contradictory = self._semantic_contradictions(memories, profiles)
        misleading = [
            memory.id
            for memory in memories
            if profiles[memory.id].reasoning_utility >= 0.65 and memory.confidence < 0.45
        ]
        outdated = [
            memory.id
            for memory in memories
            if "outdated_high_weight" in profiles[memory.id].risk_flags
        ]
        return {
            "contradictory_semantic_memories": contradictory,
            "misleading_memory_influence": misleading,
            "outdated_high_weight_memories": outdated,
            "meaning_consistent": not contradictory and not misleading and not outdated,
        }

    def _semantic_contradictions(
        self,
        memories: list[MemoryRecord],
        profiles: dict[str, SemanticMemoryProfile],
    ) -> list[dict[str, Any]]:
        positives: dict[str, MemoryRecord] = {}
        negatives: dict[str, MemoryRecord] = {}
        for memory in memories:
            key = profiles[memory.id].meaning_key
            if _is_negative_memory(memory.content):
                negatives[key] = memory
            else:
                positives[key] = memory
        return [
            {
                "meaning_key": key,
                "memory_ids": [positives[key].id, negatives[key].id],
                "semantic_types": [
                    profiles[positives[key].id].semantic_type,
                    profiles[negatives[key].id].semantic_type,
                ],
            }
            for key in sorted(set(positives) & set(negatives))
        ]

    def synchronize_policies(
        self,
        *,
        policy_arbiter: Any | None,
        shared_reward: float,
        conflict_trace: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        conflict_bias = self.learn_conflict_resolution(conflict_trace)
        synchronized: dict[str, float] = {}
        if policy_arbiter is not None:
            for policy in getattr(policy_arbiter, "policies", []):
                name = str(getattr(policy, "name", policy.__class__.__name__))
                current = float(getattr(policy, "weight", 1.0))
                delta = _clamp(shared_reward, -0.05, 0.05)
                if name == "safety" and conflict_bias.get("safe_mode", 0.0) > 0:
                    delta = max(delta + 0.1, 0.05)
                updated = _clamp(current + delta, 0.1, 3.0)
                policy.weight = updated
                synchronized[name] = updated
                self.state.policy_sync_weights[name] = updated
        return {
            "shared_reward_signal": shared_reward,
            "synchronized_policies": synchronized,
            "conflict_resolution_learning": conflict_bias,
        }

    def learn_conflict_resolution(
        self, conflict_trace: list[Mapping[str, Any]]
    ) -> dict[str, float]:
        for conflict in conflict_trace:
            resolution = str(conflict.get("resolution") or "weighted_arbitration")
            self.state.conflict_resolution_bias[resolution] = _clamp(
                self.state.conflict_resolution_bias.get(resolution, 0.0) + 0.1,
                0.0,
                1.0,
            )
        return dict(self.state.conflict_resolution_bias)

    def run_background_optimization(
        self,
        *,
        memories: list[MemoryRecord],
        lifecycle_actions: list[MemoryLifecycleAction],
        policy_conflict_frequency: float,
        system_load: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        system_load = system_load or {}
        high_load = _is_high_load(system_load)
        active_sessions = int(system_load.get("active_sessions", 0) or 0)
        memory_pressure = _clamp(len(memories) / 200.0)
        compression_pressure = _clamp(
            len([action for action in lifecycle_actions if action.action_type == "compress"]) / 20.0
        )
        interval = 1.0 if memory_pressure >= 0.5 else 3.0
        if high_load:
            interval = max(interval, 5.0)
        self.state.system_parameters.update(
            {
                "memory_pressure": memory_pressure,
                "compression_pressure": compression_pressure,
                "policy_conflict_frequency": policy_conflict_frequency,
                "scheduler_interval_turns": interval,
                "compression_budget": 0.35 if high_load else 1.0,
                "recall_priority": 1.0 + min(1.0, active_sessions / 3.0),
            }
        )
        return dict(self.state.system_parameters)

    def save(self) -> dict[str, Any]:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            return {
                "status": "failed",
                "event": "scheduler_state_save_failed",
                "state_path": str(self.state_path),
                "error": str(exc),
            }
        return {
            "status": "saved",
            "event": "scheduler_state_saved",
            "state_path": str(self.state_path),
        }

    def _load_state(self) -> SchedulerState:
        if not self.state_path.exists():
            return SchedulerState()
        try:
            return SchedulerState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return SchedulerState()

    def _list_project_memories(self, project: str | None) -> list[MemoryRecord]:
        return self.db.list_memories(
            project=project,
            include_archived=True,
            include_private=False,
            include_sensitive=False,
            limit=1000,
        )

    def _hierarchy_report(
        self,
        tier_assignments: dict[str, MemoryTierAssignment],
    ) -> dict[str, Any]:
        tier_counts: dict[str, int] = {tier: 0 for tier in self.hierarchy.tiers}
        for assignment in tier_assignments.values():
            tier_counts[assignment.tier] = tier_counts.get(assignment.tier, 0) + 1
        return {
            "tiers": self.hierarchy.tiers,
            "tier_counts": tier_counts,
            "assignments": {
                memory_id: assignment.to_dict()
                for memory_id, assignment in tier_assignments.items()
            },
            "retrieval_weighting": self.hierarchy.tier_weights,
        }

    def _load_profile(
        self,
        system_load: Mapping[str, Any],
        selected_memory_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "active_sessions": int(system_load.get("active_sessions", 0) or 0),
            "cpu_load": float(system_load.get("cpu_load", 0.0) or 0.0),
            "low_confidence_state": bool(system_load.get("low_confidence_state", False)),
            "high_load": _is_high_load(system_load),
            "active_recall": bool(selected_memory_ids),
        }

    def _conflict_frequency(self, conflict_trace: list[Mapping[str, Any]]) -> float:
        if not conflict_trace:
            return 0.0
        return _clamp(len(conflict_trace) / max(1, len(conflict_trace)))

    def _shared_reward(
        self,
        policy_feedback_signals: list[Mapping[str, Any]],
        conflict_frequency: float,
    ) -> float:
        if not policy_feedback_signals:
            reward = 0.0
        else:
            reward_total = sum(
                float(signal.get("reward", 0.0)) for signal in policy_feedback_signals
            )
            reward = reward_total / len(policy_feedback_signals)
        return _clamp(reward - conflict_frequency * 0.3, -1.0, 1.0)

    def _redundancy_rate(self, memories: list[MemoryRecord]) -> float:
        if not memories:
            return 0.0
        hashes = [memory.content_hash for memory in memories if memory.content_hash]
        if not hashes:
            return 0.0
        return _clamp((len(hashes) - len(set(hashes))) / len(hashes))


def _memory_strength(memory: MemoryRecord) -> float:
    usage_boost = min(1.0, memory.usage_count / 10.0) * 0.25
    return _clamp((memory.importance * 0.45) + (memory.confidence * 0.3) + usage_boost)


def _age_days(memory: MemoryRecord) -> int:
    source = memory.last_used_at or memory.updated_at or memory.created_at
    if not source:
        return 0
    try:
        stamp = datetime.fromisoformat(source)
    except ValueError:
        return 0
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return max(0, (datetime.now(UTC) - stamp).days)


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return _clamp(sum(values) / len(values))


def _recall_accuracy(
    policy_feedback_signals: list[Mapping[str, Any]],
    retrieved_memory_ids: list[str],
) -> float:
    names = {str(signal.get("signal")) for signal in policy_feedback_signals}
    if "successful_recall" in names:
        return 1.0
    if "fallback_usage" in names:
        return 0.0
    return 0.5 if retrieved_memory_ids else 0.0


def _task_success_rate(policy_feedback_signals: list[Mapping[str, Any]]) -> float:
    if not policy_feedback_signals:
        return 0.0
    positives = [float(signal.get("reward", 0.0)) for signal in policy_feedback_signals]
    return _clamp(sum(1 for reward in positives if reward > 0) / len(positives))


def _normalized_memory_key(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def _contradiction_key(text: str) -> str:
    normalized = _normalized_memory_key(text)
    for marker in ("do not ", "don't ", "never ", "no "):
        normalized = normalized.replace(marker, "")
    return normalized.rstrip(".")


def _semantic_meaning_key(text: str) -> str:
    normalized = _contradiction_key(text)
    prefixes = ("decision:", "error:", "fact:", "insight:", "pattern:")
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
    return normalized


def _flatten_conflict_memory_ids(contradictions: list[Any]) -> list[str]:
    memory_ids: set[str] = set()
    for contradiction in contradictions:
        if not isinstance(contradiction, Mapping):
            continue
        for memory_id in list(contradiction.get("memory_ids", [])):
            memory_ids.add(str(memory_id))
    return sorted(memory_ids)


def _attention_item(memory_id: str, score: float, reason: str) -> dict[str, Any]:
    return {
        "target_type": "memory",
        "target_id": memory_id,
        "score": _clamp(score),
        "reason": reason,
    }


def _merge_attention(attention: dict[str, dict[str, Any]], item: dict[str, Any]) -> None:
    memory_id = str(item["target_id"])
    current = attention.get(memory_id)
    if current is None or float(item["score"]) > float(current["score"]):
        attention[memory_id] = item


def _world_node_id(kind: str, value: str) -> str:
    return f"{kind}:{_canonical_concept(value)}"


def _world_edge(source: str, target: str, relation: str, weight: float) -> dict[str, Any]:
    return {"from": source, "to": target, "relation": relation, "weight": _clamp(weight)}


def _canonical_concept(value: str) -> str:
    expanded = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    normalized = expanded.casefold().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def _trace_decision_label(trace: list[Any]) -> str:
    for event in reversed(trace):
        if isinstance(event, Mapping):
            node_id = event.get("node_id")
            decision = event.get("decision")
        else:
            node_id = getattr(event, "node_id", None)
            decision = getattr(event, "decision", None)
        if node_id == "execute" and decision:
            return str(decision)
    return "unknown"


def _is_negative_memory(text: str) -> bool:
    normalized = _normalized_memory_key(text)
    return any(marker in normalized for marker in ("do not ", "don't ", "never ", "no "))


def _is_high_load(system_load: Mapping[str, Any]) -> bool:
    active_sessions = int(system_load.get("active_sessions", 0) or 0)
    cpu_load = float(system_load.get("cpu_load", 0.0) or 0.0)
    return active_sessions >= 3 or cpu_load >= 0.85


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
