from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

SUMMARY_SCHEMA_HINT = {
    "project_context": [],
    "decisions": [],
    "failures": [],
    "constraints": [],
    "task_state": [],
    "technical_facts": [],
    "open_questions": [],
}


def summarize_with_openai_compatible(
    text: str,
    project: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, list[dict[str, Any]]] | None:
    endpoint = os.getenv("HIPPO_LLM_ENDPOINT")
    api_key = os.getenv("HIPPO_LLM_API_KEY")
    model = os.getenv("HIPPO_LLM_MODEL")
    if not endpoint or not model:
        return None
    prompt = build_summary_prompt(text, project)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract durable AI coding memories. Return JSON only. "
                    "Do not turn guesses into facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    content = _extract_content(raw)
    if not content:
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return normalize_llm_summary(parsed)


def build_summary_prompt(text: str, project: str | None = None) -> str:
    return (
        f"Project: {project or 'global'}\n\n"
        "Extract only durable memories from this session. Categorize into these keys:\n"
        f"{json.dumps(SUMMARY_SCHEMA_HINT, ensure_ascii=False)}\n\n"
        "Each item must include content, memory_type, confidence, importance.\n"
        "Use lower confidence for guesses. "
        "Sensitive content should be avoided unless essential.\n\n"
        f"Session:\n{text[:12000]}"
    )


def normalize_llm_summary(value: Any) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {
        key: [] for key in SUMMARY_SCHEMA_HINT
    }
    if not isinstance(value, dict):
        return output
    for key in output:
        items = value.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str):
                output[key].append(_item(item, _type_for_bucket(key), 0.65, 0.6))
            elif isinstance(item, dict) and item.get("content"):
                output[key].append(
                    _item(
                        str(item["content"]),
                        str(item.get("memory_type") or _type_for_bucket(key)),
                        float(item.get("confidence", 0.65)),
                        float(item.get("importance", 0.6)),
                    )
                )
    return output


def _extract_content(raw: dict[str, Any]) -> str | None:
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            return message.get("content")
    return None


def _item(content: str, memory_type: str, confidence: float, importance: float) -> dict[str, Any]:
    return {
        "content": content,
        "memory_type": memory_type,
        "confidence": max(0.0, min(1.0, confidence)),
        "importance": max(0.0, min(1.0, importance)),
    }


def _type_for_bucket(bucket: str) -> str:
    return {
        "project_context": "project_context",
        "decisions": "decision",
        "failures": "failure",
        "constraints": "constraint",
        "task_state": "task_state",
        "technical_facts": "technical_fact",
        "open_questions": "task_state",
    }.get(bucket, "project_context")
