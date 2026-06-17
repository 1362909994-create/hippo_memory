from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from hippocampus_memory.callback import callback_pack
from hippocampus_memory.change_planner import ChangePlanner
from hippocampus_memory.context_bundle import ContextBundleBuilder
from hippocampus_memory.db import Database
from hippocampus_memory.packer import MemoryPacker
from hippocampus_memory.project_profile import ProjectProfileBuilder
from hippocampus_memory.token_report import format_savings_line, record_context_savings
from hippocampus_memory.utils import estimate_tokens, normalize_text

RECALL_POLICY_VERSION = "auto-recall-v1"


@dataclass(slots=True)
class RecallDecision:
    action: str
    reason: str
    confidence: float
    project: str | None
    intent: str
    strategy: str = "auto"
    max_tokens: int = 0
    source_chunk_limit: int = 0
    include_code_map: bool = True
    use_session_dedupe: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_recall(
    intent: str,
    *,
    project: str | None = None,
    max_tokens: int = 3500,
    include_code_map: bool = True,
) -> RecallDecision:
    clean_intent = normalize_text(intent)
    lowered = clean_intent.casefold()
    max_tokens = max(0, max_tokens)
    if _explicit_no_recall(lowered):
        return _decision(
            "none",
            "user explicitly requested no external memory",
            0.98,
            project,
            clean_intent,
        )
    if _asks_to_continue(lowered):
        return _decision(
            "callback_pack",
            "continuation task should recall state while avoiding repeated injections",
            0.9,
            project,
            clean_intent,
            strategy="compact",
            max_tokens=min_positive(max_tokens, 1200),
            source_chunk_limit=1,
            use_session_dedupe=True,
        )
    if _is_low_signal_request(lowered):
        return _decision(
            "none",
            "request is too small or conversational to justify memory recall",
            0.86,
            project,
            clean_intent,
        )
    if _asks_memory_question(lowered):
        return _decision(
            "memory_pack",
            "user is asking about remembered context",
            0.86,
            project,
            clean_intent,
            max_tokens=min_positive(max_tokens, 1500),
            source_chunk_limit=0,
        )
    if _asks_project_overview(lowered):
        if project:
            return _decision(
                "context_bundle",
                "broad project understanding needs profile, memory and code map",
                0.9,
                project,
                clean_intent,
                strategy="full",
                max_tokens=min_positive(max_tokens, 4500),
                source_chunk_limit=2,
                include_code_map=include_code_map,
            )
        return _decision(
            "memory_pack",
            "overview requested but no project was resolved",
            0.64,
            project,
            clean_intent,
            max_tokens=min_positive(max_tokens, 1500),
        )
    if _asks_for_impact(lowered):
        if project:
            return _decision(
                "impact_pack",
                "impact/risk request should use the change planner",
                0.84,
                project,
                clean_intent,
                max_tokens=min_positive(max_tokens, 1400),
                source_chunk_limit=2,
            )
        return _decision(
            "memory_pack",
            "impact request has no project, falling back to memory recall",
            0.54,
            project,
            clean_intent,
            max_tokens=min_positive(max_tokens, 1200),
        )
    if _asks_for_risky_change(lowered):
        if project:
            return _decision(
                "context_bundle",
                "coding/debugging task needs memory plus impact-oriented code context",
                0.88,
                project,
                clean_intent,
                strategy="lean",
                max_tokens=min_positive(max_tokens, 3500),
                source_chunk_limit=2,
                include_code_map=include_code_map,
            )
        return _decision(
            "memory_pack",
            "task looks non-trivial but no project was resolved",
            0.58,
            project,
            clean_intent,
            max_tokens=min_positive(max_tokens, 1200),
        )
    if project and _looks_task_like(lowered):
        return _decision(
            "callback_pack",
            "non-trivial project request should get compact memory",
            0.62,
            project,
            clean_intent,
            strategy="compact",
            max_tokens=min_positive(max_tokens, 900),
            source_chunk_limit=1,
            use_session_dedupe=True,
        )
    return _decision(
        "none",
        "no strong signal that external memory would improve this request",
        0.62,
        project,
        clean_intent,
    )


def build_auto_context(
    db: Database,
    *,
    intent: str,
    project: str | None = None,
    session_key: str = "default",
    max_tokens: int = 3500,
    include_code_map: bool = True,
    track_token_savings: bool = False,
    token_model: str | None = None,
    include_savings_in_text: bool = False,
) -> dict[str, Any]:
    decision = decide_recall(
        intent,
        project=project,
        max_tokens=max_tokens,
        include_code_map=include_code_map,
    )
    payload: dict[str, Any] = {
        "policy_version": RECALL_POLICY_VERSION,
        "decision": decision.to_dict(),
        "text": "",
        "token_count": 0,
        "included_memory_ids": [],
        "excluded_memory_ids": [],
    }
    if decision.action == "none":
        payload["text"] = "No external memory recall recommended for this request."
        payload["token_count"] = estimate_tokens(str(payload["text"]))
        return payload

    if decision.action == "callback_pack":
        if decision.project:
            result = callback_pack(
                db,
                project=decision.project,
                intent=decision.intent,
                session_key=session_key,
                max_tokens=decision.max_tokens,
                source_chunk_limit=decision.source_chunk_limit,
                compact=True,
            )
            payload.update(
                {
                    "text": result["text"],
                    "included_memory_ids": result["included_memory_ids"],
                    "excluded_memory_ids": result["excluded_memory_ids"],
                    "seen_memory_ids": result["seen_memory_ids"],
                }
            )
        else:
            payload["text"] = MemoryPacker(db).pack(
                decision.intent,
                project=None,
                max_tokens=decision.max_tokens,
                source_chunk_limit=decision.source_chunk_limit,
                compact=True,
            )
    elif decision.action == "memory_pack":
        payload["text"] = MemoryPacker(db).pack(
            decision.intent,
            project=decision.project,
            max_tokens=decision.max_tokens,
            source_chunk_limit=decision.source_chunk_limit,
            compact=decision.strategy == "compact",
        )
    elif decision.action == "impact_pack":
        payload["text"] = ChangePlanner(db).plan(
            intent=decision.intent,
            project=str(decision.project),
            max_tokens=decision.max_tokens,
        )
    elif decision.action == "project_profile":
        payload["text"] = ProjectProfileBuilder(db).build(str(decision.project))
    elif decision.action == "context_bundle":
        payload["text"] = ContextBundleBuilder(db).build(
            project=str(decision.project),
            intent=decision.intent,
            max_tokens=decision.max_tokens,
            include_code_map=decision.include_code_map,
            strategy=decision.strategy,
        )
    else:
        raise ValueError(f"unsupported recall action: {decision.action}")

    if track_token_savings and decision.project:
        savings = record_context_savings(
            db,
            project=decision.project,
            intent=decision.intent,
            context_type=decision.action,
            output_text=str(payload["text"]),
            model=token_model,
            record=True,
        )
        savings_text = format_savings_line(savings)
        payload["token_savings"] = savings
        payload["token_savings_text"] = savings_text
        if include_savings_in_text:
            payload["text"] = (
                f"{payload['text']}\n\n"
                "Show this token savings line to the user:\n"
                f"{savings_text}"
            )

    payload["token_count"] = estimate_tokens(str(payload["text"]))
    return payload


def min_positive(value: int, cap: int) -> int:
    if value <= 0:
        return cap
    return max(1, min(value, cap))


def _decision(
    action: str,
    reason: str,
    confidence: float,
    project: str | None,
    intent: str,
    *,
    strategy: str = "auto",
    max_tokens: int = 0,
    source_chunk_limit: int = 0,
    include_code_map: bool = True,
    use_session_dedupe: bool = False,
) -> RecallDecision:
    return RecallDecision(
        action=action,
        reason=reason,
        confidence=confidence,
        project=project,
        intent=intent,
        strategy=strategy,
        max_tokens=max_tokens,
        source_chunk_limit=source_chunk_limit,
        include_code_map=include_code_map,
        use_session_dedupe=use_session_dedupe,
    )


def _explicit_no_recall(text: str) -> bool:
    return _has_any(
        text,
        (
            "do not use memory",
            "don't use memory",
            "no external memory",
            "without memory",
            "不要调用记忆",
            "不要用记忆",
            "不用外部记忆",
        ),
    )


def _is_low_signal_request(text: str) -> bool:
    if len(re.findall(r"[a-z0-9_\u4e00-\u9fff]", text)) < 6:
        return True
    return text in {
        "ok",
        "thanks",
        "thank you",
        "好的",
        "可以",
        "明白",
        "收到",
        "继续",
    }


def _asks_memory_question(text: str) -> bool:
    return _has_any(
        text,
        (
            "what do you remember",
            "recall",
            "memory pack",
            "search memory",
            "remembered",
            "记得什么",
            "查一下记忆",
            "回忆",
            "记忆里",
        ),
    )


def _asks_project_overview(text: str) -> bool:
    return _has_any(
        text,
        (
            "project overview",
            "understand project",
            "architecture",
            "onboard",
            "how does this project work",
            "整体项目",
            "项目概览",
            "项目架构",
            "解释这个项目",
            "这个项目怎么工作",
        ),
    )


def _asks_to_continue(text: str) -> bool:
    return _has_any(
        text,
        (
            "continue",
            "resume",
            "pick up",
            "last time",
            "next step",
            "继续",
            "接着",
            "恢复",
            "上次",
            "下一步",
        ),
    )


def _asks_for_risky_change(text: str) -> bool:
    return _has_any(
        text,
        (
            "fix",
            "bug",
            "implement",
            "add ",
            "change",
            "modify",
            "refactor",
            "test",
            "failing",
            "error",
            "debug",
            "优化",
            "修复",
            "实现",
            "增加",
            "修改",
            "重构",
            "测试",
            "报错",
            "调试",
        ),
    ) or bool(re.search(r"\b[\w.-]+\.(?:py|ts|tsx|js|json|toml|md)\b", text))


def _asks_for_impact(text: str) -> bool:
    return _has_any(
        text,
        (
            "impact",
            "risk",
            "what would break",
            "影响",
            "风险",
            "会破坏",
            "改动范围",
        ),
    )


def _looks_task_like(text: str) -> bool:
    return len(text) >= 18 and _has_any(
        text,
        (
            "please",
            "can you",
            "需要",
            "帮我",
            "看看",
            "做",
            "分析",
            "检查",
        ),
    )


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.casefold() in text for term in terms)
