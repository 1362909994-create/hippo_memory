from __future__ import annotations

import re
from pathlib import Path

from hippocampus_memory.llm_summarizer import summarize_with_openai_compatible
from hippocampus_memory.utils import normalize_text

SummaryCandidates = dict[str, list[dict[str, object]]]

KEYWORDS = {
    "constraint": ["必须", "不允许", "不能", "不要", "约束", "只接受", "不得", "避免"],
    "failure": ["失败", "不行", "不要再", "无效", "报错", "踩坑", "回退"],
    "decision": ["决定", "选择", "废弃", "采用", "结论", "改成", "方案是"],
    "task_state": ["当前", "下一步", "上次", "做到", "继续", "目标", "还差"],
    "user_preference": ["记住", "以后都", "从现在开始", "偏好", "我希望", "我不想", "我只想"],
    "unknown": ["未知", "待确认", "不确定", "需要验证", "风险", "可能影响"],
    "technical_fact": [
        "api",
        "sqlite",
        "fastapi",
        "python",
        "cli",
        "mcp",
        "daemon",
        "token",
        "函数",
        "接口",
        "调用",
        "影响",
        "文件",
        "schema",
    ],
}


def summarize_session_text(text: str, project: str | None = None) -> SummaryCandidates:
    del project
    candidates: dict[str, list[dict[str, object]]] = {
        "project_context": [],
        "decisions": [],
        "failures": [],
        "constraints": [],
        "task_state": [],
        "technical_facts": [],
        "open_questions": [],
    }
    seen: set[str] = set()
    for line in _candidate_lines(text):
        key = normalize_text(line).casefold()
        if key in seen:
            continue
        seen.add(key)
        bucket, memory_type, confidence, importance = _classify_line(line)
        if bucket is None:
            continue
        candidates[bucket].append(_candidate(line, memory_type, confidence, importance))
    return {bucket: items[:8] for bucket, items in candidates.items()}


def summarize_session_file(
    path: str | Path,
    project: str | None = None,
    use_llm: bool = False,
) -> SummaryCandidates:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return summarize_session(text, project=project, use_llm=use_llm)


def summarize_session(
    text: str,
    project: str | None = None,
    use_llm: bool = False,
) -> SummaryCandidates:
    if use_llm:
        llm_summary = summarize_with_openai_compatible(text, project)
        if llm_summary is not None:
            return llm_summary
    return summarize_session_text(text, project)


def _candidate_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        if not _is_candidate_worthy(line):
            continue
        lines.append(line)
    return lines


def _clean_line(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line.strip())
    line = re.sub(
        r"^(?:用户|user|assistant|ai|codex|deepseek|system)\s*[:：]\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )
    return normalize_text(line)


def _is_candidate_worthy(line: str) -> bool:
    if len(line) < 8:
        return False
    if line in {"好的", "继续", "可以", "明白", "谢谢"}:
        return False
    return True


def _classify_line(line: str) -> tuple[str | None, str, float, float]:
    lower = line.casefold()
    if any(keyword in line for keyword in KEYWORDS["user_preference"]):
        return "project_context", "user_preference", 0.82, 0.75
    if any(keyword in line for keyword in KEYWORDS["constraint"]):
        return "constraints", "constraint", 0.82, 0.78
    if any(keyword in line for keyword in KEYWORDS["failure"]):
        return "failures", "failure", 0.82, 0.78
    if any(keyword in line for keyword in KEYWORDS["decision"]):
        return "decisions", "decision", 0.78, 0.74
    if any(keyword in line for keyword in KEYWORDS["task_state"]):
        return "task_state", "task_state", 0.72, 0.7
    if "?" in line or "？" in line or any(keyword in line for keyword in KEYWORDS["unknown"]):
        return "open_questions", "task_state", 0.58, 0.62
    if any(term in lower for term in KEYWORDS["technical_fact"]):
        return "technical_facts", "technical_fact", 0.68, 0.66
    if "项目" in line or "实现" in line or "功能" in line:
        return "project_context", "project_context", 0.62, 0.6
    return None, "technical_fact", 0.5, 0.5


def _candidate(
    content: str,
    memory_type: str,
    confidence: float,
    importance: float,
) -> dict[str, object]:
    return {
        "content": content,
        "memory_type": memory_type,
        "confidence": confidence,
        "importance": importance,
    }
